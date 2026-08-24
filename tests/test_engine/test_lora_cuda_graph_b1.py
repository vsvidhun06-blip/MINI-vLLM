"""
B1 investigation: CUDA-graph decode replay x per-request LoRA routing.

THE QUESTION
------------
`ContinuousBatchScheduler._decode_forward` can serve a decode step either
eagerly or by replaying a captured CUDA graph. Adapter routing, meanwhile, is
carried as PYTHON state: `LoRALinear._active` is read by a Python `if` inside
`LoRALinear.forward`, which selects between the base path, `_apply_single` and
`_apply_per_row`. A CUDA graph records the kernels launched during capture and
replays that recording; it does not re-execute Python. Those two facts are in
tension, and this module establishes experimentally whether the tension is real
and what it costs.

HOW THE EVIDENCE IS SPLIT (following test_live_graph.py's convention)
---------------------------------------------------------------------
  CPU (runs everywhere)
    * That LoRA routing is expressed as a DIFFERENT SEQUENCE OF TENSOR
      OPERATIONS, not as data flowing through a fixed sequence. This is the
      mechanism, and it is fully observable without a GPU.
    * That the static-shape decode path is exact for a base workload, and
      exact for a LoRA workload when replay re-reads the live routing. These
      are the controls that isolate routing as the cause of the divergence
      below rather than the staging path.
    * That a replay which executes the CAPTURE-TIME routing branch produces
      wrong tokens for every request in a mixed-adapter workload.

  CUDA (skipped on CPU-only hosts -- must be run on the GPU box)
    * Whether a real `torch.cuda.CUDAGraph` capture under an active per-row
      plan even succeeds, and whether a real replay honours live routing.

NOTHING HERE FABRICATES A CUDA RESULT. `_FrozenRoutingRunner` is a MODEL of one
specific, documented CUDA-graph property -- that the Python branch resolved at
capture is the branch whose kernels replay executes -- and is labelled as such
everywhere it appears. It is not evidence about CUDA execution; the CUDA-gated
tests are the only ones that can be.
"""
from __future__ import annotations

from collections import Counter

import pytest
import torch
from torch.overrides import TorchFunctionMode

from src.engine.kv_cache import PagedKVCache, PagedRequestCache
from src.engine.live_graph import DecodeGraphRouter, StaticDecodeBatch
from src.engine.lora import LoRALinear, LoRAManager
from src.engine.lora_model import (
    LoRALlamaModel,
    model_routing_plan,
    random_adapter_weights,
)
from src.engine.model import LlamaConfig, LlamaModel
from src.engine.scheduler import ContinuousBatchScheduler, RequestStatus

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="real CUDA graph capture/replay can only be verified on a GPU host; "
           "asserting it from CPU would be fabricating the result",
)

BLOCK_SIZE = 16
RANK = 8
ALPHA = 16.0
# See test_lora_scheduler_integration.py: random_adapter_weights' default 0.02
# is too gentle to flip a greedy argmax on a 256-token vocab, which would make
# every assertion here vacuous. 0.1 separates every adapter from base and from
# every other adapter without saturating. The negative-control test enforces it.
SCALE_INIT = 0.1

_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")


def _tiny_model() -> LlamaModel:
    """head_dim = 128/8 = 16, a power of two, so the CUDA FA2 decode kernel
    accepts this model too -- the CUDA-gated tests reuse it unchanged."""
    torch.manual_seed(1234)
    config = LlamaConfig(
        vocab_size=256, hidden_size=128, intermediate_size=256,
        num_hidden_layers=3, num_attention_heads=8, num_key_value_heads=4,
        max_position_embeddings=4096, rms_norm_eps=1e-5, rope_theta=10000.0,
        tie_word_embeddings=False,
    )
    return LlamaModel(config).eval()


def _lora_model(device="cpu") -> LoRALlamaModel:
    """Wrapped tiny model with adapters "a" and "b" resident."""
    model = LoRALlamaModel(_tiny_model().to(device), LoRAManager())
    for adapter_id, seed in (("a", 1), ("b", 2)):
        model.manager.load_adapter(
            adapter_id, rank=RANK, alpha=ALPHA,
            weights_dict=random_adapter_weights(
                model, rank=RANK, seed=seed, scale_init=SCALE_INIT),
        )
    return model


def _projections(model: LoRALlamaModel):
    for block in model.layers:
        for proj in _PROJECTIONS:
            yield getattr(block.attn, proj)


# ---------------------------------------------------------------------------
# CPU 1. The mechanism: routing selects an op SEQUENCE, not a data value.
# ---------------------------------------------------------------------------


class _OpRecorder(TorchFunctionMode):
    """Records every torch-level operation executed inside the context."""

    def __init__(self):
        self.ops: list[str] = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.ops.append(getattr(func, "__name__", str(func)))
        return func(*args, **(kwargs or {}))


def _decode_fixture(model: LoRALlamaModel, batch: int = 2, prompt_len: int = 8):
    """A pool + prefilled caches + a decode-shaped input, ready to forward."""
    cfg = model.config
    pool = PagedKVCache(
        num_layers=cfg.num_hidden_layers, num_blocks=64, block_size=BLOCK_SIZE,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.hidden_size // cfg.num_attention_heads,
        dtype=torch.float32, device="cpu", enable_prefix_cache=False,
    )
    caches = []
    g = torch.Generator().manual_seed(5)
    for i in range(batch):
        rid = f"fx{i}"
        pool.admit_request(request_id=rid, prefill_blocks_needed=4,
                           total_blocks_needed=4)
        cache = PagedRequestCache(pool, rid, num_layers=cfg.num_hidden_layers)
        with torch.no_grad():
            model.model(torch.randint(0, 256, (1, prompt_len), generator=g),
                        kv_cache=cache)
        caches.append(cache)
    return pool, caches, torch.randint(0, 256, (batch, 1), generator=g)


def _record_ops(model, caches, input_ids, plan) -> list[str]:
    """Ops executed by one decode forward under `plan`, cache state restored."""
    model.set_batch_adapters(plan) if isinstance(plan, list) else model.set_adapter(plan)
    saved = [list(c._seq_lens) for c in caches]            # noqa: SLF001
    recorder = _OpRecorder()
    with torch.no_grad(), recorder:
        model(input_ids, kv_cache=caches)
    for cache, seq_lens in zip(caches, saved):
        cache._seq_lens[:] = seq_lens                      # noqa: SLF001
        cache._bt_tensor = None                            # noqa: SLF001
        cache._seqlen_tensors.clear()                      # noqa: SLF001
        cache._seqlen_vals.clear()                         # noqa: SLF001
    return recorder.ops


def test_routing_selects_a_different_operation_sequence():
    """LoRA routing is CONTROL FLOW, not data -- the core of the B1 hazard.

    A mechanism that records operations once and replays the recording (which
    is exactly what a CUDA graph is) can only ever execute the sequence chosen
    at record time. This test shows the sequence genuinely differs, so such a
    mechanism cannot follow a routing change. It makes no claim about CUDA.
    """
    model = _lora_model()
    _pool, caches, input_ids = _decode_fixture(model)

    base_ops = _record_ops(model, caches, input_ids, [None, None])
    routed_ops = _record_ops(model, caches, input_ids, ["a", "b"])

    # The delta ops that only a routed forward performs.
    extra = Counter(routed_ops) - Counter(base_ops)
    assert extra["index_add"] > 0 and extra["index_select"] > 0, (
        f"expected per-row grouping ops to appear only under routing; "
        f"diff was {dict(extra)}"
    )
    assert extra["matmul"] > 0, "expected the LoRA delta GEMMs to be extra ops"

    # And the base-routed sequence contains NONE of them: a recording taken
    # while nothing is routed has no adapter kernels to replay, at all.
    assert not any("index_add" in op for op in base_ops), (
        "a base-routed forward unexpectedly contains grouping ops"
    )
    assert len(routed_ops) > len(base_ops), (
        f"routed forward should execute strictly more ops "
        f"({len(routed_ops)} vs {len(base_ops)})"
    )


def test_per_row_routing_constructs_tensors_from_python_lists():
    """`_apply_per_row` builds its index tensor from a Python list per layer.

    On CUDA that is a pageable host->device copy. attention.py:540-545 removed
    exactly this pattern from the decode path on the grounds that it is
    "illegal inside a CUDA-graph capture". Recording it here states the
    hypothesis precisely so the CUDA-gated test below can settle it; this test
    itself only establishes that the copies exist, which is CPU-observable.
    """
    model = _lora_model()
    _pool, caches, input_ids = _decode_fixture(model)

    base_ops = _record_ops(model, caches, input_ids, [None, None])
    routed_ops = _record_ops(model, caches, input_ids, ["a", "b"])

    assert base_ops.count("tensor") == 0, (
        "the base decode path should construct no tensors from host data"
    )
    assert routed_ops.count("tensor") > 0, (
        "expected _apply_per_row's torch.tensor(rows, ...) index construction"
    )


# ---------------------------------------------------------------------------
# CPU 2. Controls + the reproducer.
#
# `_StaticReplayRunner` runs the SAME StaticDecodeBatch staging a captured
# graph replays, through the real DecodeGraphRouter, exactly as the repo's
# existing `_CpuStaticRunner` (test_live_graph.py) does. The single added knob
# is `freeze_routing`:
#
#   freeze_routing=False -- replay re-reads live `_active`. This is what the
#                           existing CPU stand-in does, and it is why that
#                           stand-in cannot observe B1.
#   freeze_routing=True  -- replay executes the branch each LoRALinear took at
#                           CAPTURE time. This models the one CUDA-graph
#                           property under investigation and nothing else.
#
# Holding everything else identical between the two makes routing the only
# variable, which is what isolates it as the cause.
# ---------------------------------------------------------------------------


class _StaticReplayRunner(DecodeGraphRouter):
    """CPU stand-in for a captured decode graph. NOT a CUDA result.

    `guard_routing=False` reproduces the PRE-FIX runner: `_resolve` ignores the
    routing plan, exactly as it did before the B1 guard existed. It is how the
    hazard characterisation below can still demonstrate what a stale replay
    does now that production refuses to perform one.
    """

    def __init__(self, model, pool, *, freeze_routing: bool,
                 guard_routing: bool = True, **kwargs):
        super().__init__(pool, **kwargs)
        self.model = model
        self.freeze_routing = freeze_routing
        self.guard_routing = guard_routing
        self.replay_calls = 0
        self._captured_plans: dict[int, object] = {}
        self.graph_plans: dict[tuple[int, int], object] = {}
        for batch_size in self.batch_sizes:
            for n_blocks in self.buckets:
                self.graphs[(batch_size, n_blocks)] = (
                    StaticDecodeBatch(pool, batch_size, n_blocks),)

    def capture(self) -> None:
        """Freeze the routing branch every projection would take right now, and
        register the plan in `graph_plans` the way `LiveDecodeGraphRunner.
        _capture` does.

        Mirrors `capture_all()`, which the scheduler docstring instructs callers
        to run BEFORE admitting any request -- i.e. while nothing is routed.
        Setting a plan on the model before calling this captures under that plan
        instead, which is how the "a routed graph cannot serve a different plan"
        test is built.
        """
        for layer in _projections(self.model):
            if isinstance(layer, LoRALinear):                   # plain model: skip
                self._captured_plans[id(layer)] = layer._active  # noqa: SLF001
        plan = model_routing_plan(self.model)
        for key in self.graphs:
            self.graph_plans[key] = plan

    def _resolve(self, batch_size, caches, routing_plan=None):
        if not self.guard_routing:
            # Pre-fix behaviour: whatever the recording holds is accepted.
            routing_plan = next(iter(self.graph_plans.values()), None)
        return super()._resolve(batch_size, caches, routing_plan)

    def replay(self, input_ids, caches):
        # Resolve with the live plan, as LiveDecodeGraphRunner.replay does --
        # otherwise a graph legitimately captured under a routed plan could
        # never be replayed.
        key = self._resolve(len(caches), caches, model_routing_plan(self.model))
        assert key is not None, "replay() called without a routable key"
        self.replay_calls += 1
        (state,) = self.graphs[key]
        state.stage(input_ids, caches)
        if self.freeze_routing:
            out = self._run_with_captured_routing(state)
        else:
            out = state.run_eager(self.model)
        state.commit(caches)
        return out

    def _run_with_captured_routing(self, state):
        original = LoRALinear.forward
        captured = self._captured_plans

        def frozen_forward(layer, x, adapter_id=None):
            live = layer._active                                # noqa: SLF001
            layer._active = captured.get(id(layer))             # noqa: SLF001
            try:
                return original(layer, x, adapter_id)
            finally:
                layer._active = live                            # noqa: SLF001

        LoRALinear.forward = frozen_forward
        try:
            return state.run_eager(self.model)
        finally:
            LoRALinear.forward = original


# (request_id, adapter_id, max_new_tokens, admit_on_step).
#
# Tuned to produce BOTH kinds of composition change a frozen recording can fail
# on, because they are detectable by different failure modes:
#
#   * batch size changes (2 -> 1 -> 3): B retires early, C/D join mid-flight.
#   * batch size CONSTANT while the adapter at a row index changes:
#     D ("a") retires exactly as E ("b") joins, taking the batch from
#     ('a','b','a') to ('a','b','b') at size 3.
#
# The second is the only situation in which a plan that lags the request order
# by one step is observable -- without it, mutation M3 is undetectable because
# consecutive same-size steps always carry the identical plan. The reproducer
# asserts both properties hold so a future re-timing cannot silently drop them.
_WORKLOAD = [
    ("A", "a", 10, 0),
    ("B", "b",  3, 0),
    ("C", "b",  6, 4),
    ("D", "a",  3, 4),
    ("E", "b",  4, 6),
]


def _serve(model, *, adapters, runner_mode, ledger=None, guard_routing=True,
           capture_plan=None, out=None):
    """Drive the real scheduler over _WORKLOAD; return {request_id: tokens}.

    `runner_mode` is None (no runner -> eager), False (static replay that
    re-reads routing) or True (static replay frozen at capture time).
    `adapters` maps the workload's adapter column, so the same schedule can be
    served as an all-base control.
    """
    model.clear_adapters()
    sched = ContinuousBatchScheduler(
        model, max_batch_size=4, num_blocks=256, block_size=BLOCK_SIZE,
        use_cuda_graphs=True,
    )
    runner = None
    if runner_mode is not None:
        runner = _StaticReplayRunner(
            model, sched.pool, freeze_routing=runner_mode,
            guard_routing=guard_routing,
            max_batch_size=4, max_context_tokens=64,
        )
        if capture_plan is not None:
            model.set_batch_adapters(capture_plan)   # capture UNDER this plan
        runner.capture()          # before any request, per the documented usage
        model.clear_adapters()
        sched._graph_runner = runner
    if out is not None:
        out["runner"] = runner
        out["scheduler"] = sched

    original_decode = ContinuousBatchScheduler._decode_forward

    def spy(self, input_ids, caches):
        decode_reqs = [r for r in self.active if r.status is RequestStatus.DECODE]
        active = model._lora_layers[0]._active                  # noqa: SLF001
        replayed = (
            runner is not None and self.use_cuda_graphs
            and runner.can_replay(len(caches), caches)
        )
        if ledger is not None:
            ledger.append({
                "step": self._step_idx,
                "order": [r.request_id for r in decode_reqs],
                "adapters": [r.adapter_id for r in decode_reqs],
                "plan": list(active) if isinstance(active, list) else active,
                "batch_size": int(input_ids.shape[0]),
                "caches_aligned": [r.cache for r in decode_reqs] == caches,
                "path": "replay" if replayed else "eager",
            })
        return original_decode(self, input_ids, caches)

    tokens: dict[str, list[int]] = {}
    try:
        ContinuousBatchScheduler._decode_forward = spy
        g = torch.Generator().manual_seed(7)
        prompts = {rid: torch.randint(0, 256, (1, 6), generator=g)
                   for rid, _, _, _ in _WORKLOAD}
        for step in range(60):
            for rid, _adapter, max_new, admit_step in _WORKLOAD:
                if step == admit_step:
                    sched.add_request(
                        rid, prompts[rid], max_new_tokens=max_new,
                        eos_token_id=None, adapter_id=adapters[rid],
                    )
            last_admit = max(w[3] for w in _WORKLOAD)
            if not sched.has_work() and step > last_admit:
                break
            for rid, token_id in sched.step():
                tokens.setdefault(rid, []).append(token_id)
    finally:
        ContinuousBatchScheduler._decode_forward = original_decode
    return tokens


_LORA_ADAPTERS = {rid: adapter for rid, adapter, _, _ in _WORKLOAD}
_BASE_ADAPTERS = {rid: None for rid, _, _, _ in _WORKLOAD}


def test_negative_control_adapters_are_distinguishable():
    """base != a, base != b, a != b on this workload -- else all else is vacuous."""
    model = _lora_model()
    prompt = torch.randint(0, 256, (1, 6), generator=torch.Generator().manual_seed(7))
    outs = {}
    for adapter_id in (None, "a", "b"):
        out = model.generate(prompt, max_new_tokens=8, eos_token_id=None,
                             use_cache=True, adapter_id=adapter_id)
        outs[adapter_id] = tuple(out[0, prompt.shape[1]:].tolist())
    assert len(set(outs.values())) == 3, (
        f"adapters are not distinguishable on this input, so every routing "
        f"assertion in this module would pass vacuously: {outs}"
    )
    for adapter_id, seq in outs.items():
        assert len(set(seq)) > 1, (
            f"adapter {adapter_id!r} saturated to a single repeated token {seq}; "
            f"SCALE_INIT is too large"
        )


def test_control_static_replay_is_exact_for_a_base_workload():
    """CONTROL 1: with no adapters anywhere, frozen replay == eager exactly.

    Establishes that the StaticDecodeBatch staging path introduces no error of
    its own, so any divergence in the LoRA workload cannot be blamed on it.
    """
    model = _lora_model()
    eager = _serve(model, adapters=_BASE_ADAPTERS, runner_mode=None)
    frozen = _serve(model, adapters=_BASE_ADAPTERS, runner_mode=True)
    assert eager == frozen, (
        "the static-shape decode path is not exact even without LoRA; the B1 "
        "experiment below would be uninterpretable"
    )


def test_control_replay_that_rereads_routing_matches_eager():
    """CONTROL 2: same LoRA workload, same staging, routing re-read at replay.

    This is what the repo's existing `_CpuStaticRunner` does -- and it passes,
    which is precisely why that stand-in cannot detect B1. Together with the
    experiment below it isolates capture-time routing as the sole cause.
    """
    model = _lora_model()
    eager = _serve(model, adapters=_LORA_ADAPTERS, runner_mode=None)
    # guard_routing=False so the batch actually reaches replay: with the B1
    # guard on, a routed batch is refused and this control would prove nothing.
    rereading = _serve(model, adapters=_LORA_ADAPTERS, runner_mode=False,
                       guard_routing=False)
    assert eager == rereading, (
        "static replay diverged from eager even while re-reading live routing; "
        "the cause would then not be capture-time freezing"
    )


def test_replay_frozen_at_capture_time_routing_corrupts_every_request():
    """THE REPRODUCER: freezing the routing branch at capture corrupts output.

    Not a claim about CUDA execution -- a claim about the property CUDA graphs
    have. It stays true and worth pinning after any fix: it is the statement of
    WHY replay must not freeze routing.

    The scheduler's own plan is correct on every step (asserted below); the
    corruption is entirely downstream of it, which is why Phase 1's integration
    tests pass while the served tokens are wrong.
    """
    model = _lora_model()
    ledger: list[dict] = []
    eager = _serve(model, adapters=_LORA_ADAPTERS, runner_mode=None)
    # guard_routing=False reproduces the PRE-FIX runner, in which _resolve
    # ignored the routing plan. Production now refuses this replay; the point
    # of this test is to keep on record what it would do if it did not.
    frozen = _serve(model, adapters=_LORA_ADAPTERS, runner_mode=True,
                    ledger=ledger, guard_routing=False)

    # The scheduler did its job: plan == live decode order on every forward.
    assert ledger, "no decode forward was recorded"
    for row in ledger:
        assert row["plan"] == row["adapters"], (
            f"scheduler routing itself drifted, which is a different bug: {row}"
        )
        assert row["caches_aligned"], f"KV-cache ordering drifted: {row}"
    assert all(row["path"] == "replay" for row in ledger), (
        f"not every decode step went through replay: "
        f"{[r['path'] for r in ledger]}"
    )
    # The composition really did change under the frozen recording -- both ways.
    assert len({row["batch_size"] for row in ledger}) >= 2, (
        f"batch size never changed: {[r['batch_size'] for r in ledger]}"
    )
    same_size_reorder = [
        (a, b) for a, b in zip(ledger, ledger[1:])
        if a["batch_size"] == b["batch_size"] and a["adapters"] != b["adapters"]
    ]
    assert same_size_reorder, (
        "no two consecutive decode steps kept the batch size while changing the "
        "adapter at a row index; the workload no longer exercises a plan that "
        "lags the request order (mutation M3 would be undetectable). Observed: "
        f"{[(r['batch_size'], r['adapters']) for r in ledger]}"
    )

    corrupted = [rid for rid in eager if eager[rid] != frozen.get(rid)]
    assert corrupted, (
        "freezing capture-time routing did not change any output; the model "
        "of CUDA-graph behaviour is not exercising the routing branch"
    )
    assert set(corrupted) == set(eager), (
        f"expected every request to be corrupted, only {corrupted} were"
    )


def test_scheduler_must_not_serve_a_routed_batch_from_a_stale_recording():
    """THE B1 INVARIANT.

    A graph-accelerated scheduler must produce exactly the eager tokens for a
    LoRA workload -- either by making replay follow the live plan, or by
    refusing to replay a batch whose routing the recording does not match. The
    implemented fix takes the second route; this test is agnostic and passes
    under either.
    """
    model = _lora_model()
    eager = _serve(model, adapters=_LORA_ADAPTERS, runner_mode=None)
    graph_served = _serve(model, adapters=_LORA_ADAPTERS, runner_mode=True)
    assert eager == graph_served, (
        "graph-accelerated decode served different tokens than eager decode "
        "for a mixed-adapter workload"
    )


def test_routed_batch_is_refused_and_the_reason_is_recorded():
    """The guard REFUSES the replay -- not "the output happened to be right".

    Asserts on the mechanism rather than the symptom: replay() is never entered
    for a routed batch, and the refusal is attributable through the existing
    fallback accounting instead of being silent.
    """
    model = _lora_model()
    out: dict = {}
    _serve(model, adapters=_LORA_ADAPTERS, runner_mode=True, out=out)
    runner, sched = out["runner"], out["scheduler"]

    assert runner.replay_calls == 0, (
        f"a routed batch reached graph replay {runner.replay_calls} times; the "
        f"recording was captured unrouted, so those replays served the wrong "
        f"adapters"
    )
    diagnostics = sched.graph_diagnostics()
    assert diagnostics["runner_fallback_reasons"].get(
        "lora_routing_mismatch", 0) > 0, (
        f"the routing refusal was not recorded in the fallback accounting: "
        f"{diagnostics}"
    )
    assert diagnostics["reasons"].get("graph_hit", 0) == 0, (
        f"a graph hit was counted for a routed workload: {diagnostics}"
    )
    assert diagnostics["reasons"].get("runner_rejected", 0) > 0, (
        f"the scheduler did not record the runner as the refusing party: "
        f"{diagnostics}"
    )


def test_all_base_batch_still_takes_the_graph_fast_path():
    """The guard must not cost base traffic its graph acceleration.

    An all-base batch's plan is [None, ...] and an unrouted recording's is
    None; those canonicalise to the same thing, so the fast path stays open.
    If this fails, the fix has over-reached into a performance regression for
    every non-LoRA request.
    """
    model = _lora_model()
    out: dict = {}
    served = _serve(model, adapters=_BASE_ADAPTERS, runner_mode=True, out=out)
    runner, sched = out["runner"], out["scheduler"]
    eager = _serve(model, adapters=_BASE_ADAPTERS, runner_mode=None)

    diagnostics = sched.graph_diagnostics()
    assert runner.replay_calls > 0, "an all-base batch never reached replay"
    assert diagnostics["reasons"].get("graph_hit", 0) > 0, (
        f"an all-base batch lost the graph fast path: {diagnostics}"
    )
    assert diagnostics["runner_fallback_reasons"].get(
        "lora_routing_mismatch", 0) == 0, (
        f"an all-base batch was refused as a routing mismatch: {diagnostics}"
    )
    assert served == eager, "the graph fast path changed base-workload output"


def test_routed_graph_cannot_serve_a_different_routing_plan():
    """Plan equality is exact -- a routed recording is not a wildcard.

    Captures under ["b", "a"] and then serves a workload whose size-2 batches
    are ["a", "b"] -- same batch size, same bucket, same caches, different
    adapter order. Every decode step must be refused: plan equality is exact,
    and a recording taken under one adapter mix must not serve another.
    """
    model = _lora_model()
    out: dict = {}
    served = _serve(model, adapters=_LORA_ADAPTERS, runner_mode=True,
                    capture_plan=["b", "a"], out=out)
    runner, sched = out["runner"], out["scheduler"]
    eager = _serve(model, adapters=_LORA_ADAPTERS, runner_mode=None)

    assert runner.graph_plans[(2, runner.buckets[0])] == ("b", "a"), (
        "the runner did not record the plan it captured under"
    )
    assert runner.replay_calls == 0, (
        f"a recording captured under a routed plan served "
        f"{runner.replay_calls} batches routed differently"
    )
    assert sched.graph_diagnostics()["runner_fallback_reasons"].get(
        "lora_routing_mismatch", 0) > 0
    assert served == eager


def test_non_lora_model_is_unaffected_by_the_routing_guard():
    """A plain LlamaModel produces no plan, so the guard is inert for it."""
    plain = _tiny_model()
    assert not hasattr(plain, "get_batch_adapters")
    assert model_routing_plan(plain) is None

    sched = ContinuousBatchScheduler(
        plain, max_batch_size=4, num_blocks=256, block_size=BLOCK_SIZE,
        use_cuda_graphs=True,
    )
    runner = _StaticReplayRunner(
        plain, sched.pool, freeze_routing=True,
        max_batch_size=4, max_context_tokens=64,
    )
    runner.capture()
    sched._graph_runner = runner
    assert runner.graph_plans[(1, runner.buckets[0])] is None

    g = torch.Generator().manual_seed(7)
    for i in range(2):
        sched.add_request(f"p{i}", torch.randint(0, 256, (1, 6), generator=g),
                          max_new_tokens=5, eos_token_id=None)
    while sched.has_work():
        sched.step()

    diagnostics = sched.graph_diagnostics()
    assert runner.replay_calls > 0, "the non-LoRA path lost graph replay"
    assert diagnostics["runner_fallback_reasons"].get(
        "lora_routing_mismatch", 0) == 0, (
        f"the routing guard fired for a model that has no routing: {diagnostics}"
    )


# ---------------------------------------------------------------------------
# CUDA-gated. These are the only tests here that can say anything about real
# CUDA graph execution. They are unverified on a CPU-only host -- that is a
# reported gap, not a passing result.
# ---------------------------------------------------------------------------


@cuda_only
def test_cuda_capture_under_active_per_row_routing():
    """Does a real capture even SUCCEED while a per-row plan is active?

    `_apply_per_row` constructs its index tensor from a Python list once per
    wrapped projection (see the CPU test above). On CUDA that is a pageable
    H2D copy, which CUDA stream capture rejects. Records the outcome either
    way rather than asserting a guess: both answers are informative, and they
    imply different fixes.
    """
    from src.engine.live_graph import LiveDecodeGraphRunner

    model = _lora_model(device="cuda")
    sched = ContinuousBatchScheduler(
        model, max_batch_size=2, num_blocks=256, block_size=BLOCK_SIZE,
        use_cuda_graphs=True,
    )
    runner = LiveDecodeGraphRunner(
        model, sched.pool, max_batch_size=2, max_context_tokens=64,
    )
    model.set_batch_adapters(["a", "b"])
    outcome, detail = "captured", ""
    try:
        runner.capture_all(require_all=True)
    except Exception as exc:                                    # noqa: BLE001
        outcome, detail = "refused", f"{type(exc).__name__}: {exc}"
    print(f"\n[B1/cuda] capture under an active per-row plan: {outcome} {detail}")
    assert outcome in {"captured", "refused"}


@cuda_only
def test_cuda_replay_honours_live_lora_routing():
    """The real thing: capture unrouted, then serve a mixed-adapter batch.

    Asserts the same invariant as the CPU xfail above, against a genuine
    `torch.cuda.CUDAGraph`. Expected to FAIL until B1 is fixed; it is not
    xfail-marked because it has never been executed and its true outcome on
    hardware is exactly what this investigation could not determine.
    """
    from src.engine.live_graph import LiveDecodeGraphRunner

    model = _lora_model(device="cuda")

    def serve(attach_runner):
        model.clear_adapters()
        sched = ContinuousBatchScheduler(
            model, max_batch_size=4, num_blocks=256, block_size=BLOCK_SIZE,
            use_cuda_graphs=True,
        )
        if attach_runner:
            runner = LiveDecodeGraphRunner(
                model, sched.pool, max_batch_size=4, max_context_tokens=64,
            )
            runner.capture_all()          # unrouted, per documented usage
            sched._graph_runner = runner
        g = torch.Generator().manual_seed(7)
        tokens: dict[str, list[int]] = {}
        prompts = {rid: torch.randint(0, 256, (1, 6), generator=g).cuda()
                   for rid, _, _, _ in _WORKLOAD}
        for step in range(60):
            for rid, _a, max_new, admit_step in _WORKLOAD:
                if step == admit_step:
                    sched.add_request(rid, prompts[rid], max_new_tokens=max_new,
                                      eos_token_id=None,
                                      adapter_id=_LORA_ADAPTERS[rid])
            if not sched.has_work() and step > max(w[3] for w in _WORKLOAD):
                break
            for rid, token_id in sched.step():
                tokens.setdefault(rid, []).append(token_id)
        return tokens, sched

    eager, _ = serve(False)
    graphed, sched = serve(True)
    hits = sched.graph_diagnostics()["reasons"].get("graph_hit", 0)
    assert hits > 0, (
        f"no graph replay happened, so this test proved nothing: "
        f"{sched.graph_diagnostics()}"
    )
    assert eager == graphed, (
        f"CUDA graph replay served different tokens than eager decode for a "
        f"mixed-adapter workload ({hits} replays)"
    )

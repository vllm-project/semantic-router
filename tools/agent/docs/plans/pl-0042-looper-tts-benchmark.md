# PL-0042: Looper Fixed-Budget Benchmark

## Goal

Deliver the controlled quality/cost experiment matrix in [issue #2858](https://github.com/vllm-project/semantic-router/issues/2858).

## Scope

Versioned experiment contracts, real Looper execution with budget accounting,
cached candidate comparisons, native scoring and reproducible reports.

## Non-Goals

New production algorithms and claims based on synthetic fixture scores.

## Exit Criteria

Four algorithm families run under two matched budgets on frozen GPQA-Diamond
and HLE slices. Reports preserve raw evidence, paired uncertainty, token/cost
and latency accounting, with same-model and mixed-model results separated.

## Task List

- [x] `TTS-01` Validate offline contracts, matrix planning and deterministic fixtures.
- [ ] `TTS-02` Integrate real algorithm execution, budget enforcement and per-call
  accounting.
- [ ] `TTS-03` Add cached-panel replay, repeated sampling and voting controls.
- [ ] `TTS-04` Add native scoring, paired reports, CI fixture and reproducible live
  smoke.

## Next Action

Probe per-call evidence availability in the four existing algorithm
implementations for `TTS-02`.

## Operating Rules

Freeze inputs before live campaigns. Record failures and unknown usage explicitly.
Keep synthetic evidence separate from benchmark results and replay expenditure
separate from full algorithm cost. Use the canonical local image workflow for
runtime validation.

## Related Docs

- [Execution plan index](README.md)
- [Benchmark contract](../../../../bench/looper_tts/README.md)
- [Existing EvalScope runner](../../../../bench/router_flow/real_eval/README.md)
- [Cached Fusion comparisons](../../../../bench/grounded_fusion/README.md)

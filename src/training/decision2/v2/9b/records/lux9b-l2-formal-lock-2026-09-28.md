# 9B L2 formal post-key same-panel run: lock

Frozen 2026-09-28 before any formal prediction. Approved by the coordinator
after the [Milestone 2 result](lux9b-m2-continuation-result-2026-09-28.md).
The v3 labels were accessed earlier in the project, so this is a **post-key
same-panel** comparison; public 231 is a public-subset reproduction.

## Candidate and control

| | Candidate L2 | Same-limit control Lux1-8K |
| --- | --- | --- |
| Weights | Lux 1.0 `bd45a30a…` + L2 LoRA/head, `checkpoint-0000466` (preregistered primary seed 20260926): adapter `3ca05c63…`, head `febb93bd…`, decision config `68d08142…` | untouched Lux 1.0 package `bd45a30a…` (bundle `985ade73…`) |
| Calibration | CAL700 temperatures `6a003785…` (Choice .979, Noul .777, Score .102) | package temperature 2.00544 |
| Adapter | `v2/dec/adapter-spec-infer-dec.json` (`v2.dec.infer_dec`) | `v2/9b/lux9b/adapter-spec-lux1-infer1p0.json` (`v2.dec.infer_1p0`, the shared renderer L2 trains and reads with) |
| Input limit | 8,192 tokens, over-length inputs invalid, no truncation | 8,192 tokens, same rule |

Both use the pinned image `sha256:f83b1d10…` with the FLA overlay, node A
GPU2, `TRITON_CACHE_AUTOTUNING=1` and one persisted cache (the Milestone 2
cache, seeded from the eval track's frozen node-A Lux1 cache, tree `e215f8bd…`).
Each adapter first runs the runner's gold-free smoke (`--max-items 8`).

## Panels and reading

- Formal panels: typed FINAL 1,600 / 2,000 slots, CSS15 6,547, public 231;
  plus the eval track's `mlx-diag` multilingual diagnostic (2,275 prompts).
  Seal before scoring; report and compare with the eval runner.
- **Release gate (COORDINATION.md):** L2 is a 9B release candidate only if its
  paired post-key v3 95% interval against the same-limit Lux1-8K control has a
  lower bound > 0. The existing node-A Lux1 run at its native 16,384-token
  limit (65.808 / 183) is reported beside it descriptively.
- Reported with the comparison: T, H, Choice/Noul/Score, typed families,
  per-task CSS15 transfer, public 231 tiers, mlx-diag, typed and CSS
  calibration, invalid counts.
- No checkpoint, seed, calibration or limit changes after collection. A
  collection fault is recorded and not retried blindly.

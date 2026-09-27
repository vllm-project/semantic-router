# Official Qwen3.5 4B Base: first complete clean-v2 training arm

This independent arm is frozen after the source/one-update native preflight
and before its first optimizer update. It is a development candidate, not
the Decision 2.0 4B release. The prior smoke weights and optimizer are not
reused. The earlier own-Nox and third-party-start experiments keep their
recorded outcomes and are not rerun.

## Fixed initialization, data and budget

- Start from official `Qwen/Qwen3.5-4B-Base` revision
  `1001bb4d826a52d1f399e183466143f4da7b741b`, with the exact source
  hashes and successful preflight recorded in the linked
  [native admission](qwen35-4b-official-base-native-preflight-2026-09-27.md).
  Use a new 256-dimensional dynamic-option decision head, seed 20260926,
  LoRA rank 16, alpha 32, dropout .05 plus a trainable head. No third-party
  Decision weights initialize this candidate.
- TRAIN rights-clean v2 7,455 rows SHA-256
  `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`,
  SELECT 700 `32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6`,
  CAL 700 `3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a`.
  The 4B source tokenizer admits all TRAIN/SELECT/CAL inputs without
  truncation at max length 8,192; TRAIN has 4,194,465 encoded tokens.
  CAL is isolated and never used to choose an optimizer checkpoint.
- Train one complete epoch: **466 optimizer updates**, microbatch 1,
  accumulation 16 (partial last window), CE plus 0.5 Brier, AdamW weight
  decay .01, gradient clip 1.0, BF16 backbone compute and FP32
  parameters/head/loss. LoRA peak LR `1e-4`, head peak LR `2e-4`, warmup
  ratio .05, cosine tail, gradient checkpointing enabled. No replay,
  teacher, extra rows or smoke optimizer state.
- Save and evaluate SELECT after updates 64, 128, 192, 256, 320, 384, 448
  and final 466. Pick one BEST by SELECT family-macro accuracy descending,
  normalized Brier ascending, then earliest update. Typed DEV, CSS pilot,
  public231 and formal labels cannot alter the checkpoint choice. One
  reserved GPU, **3.0 GPU-hour** wall cap including SELECT inference and
  saves. Stop on OOM, nonfinite loss/gradient, source/data mismatch,
  missing type, failed save or cap; retain an incomplete receipt. Do not
  replace an underperforming BEST with another checkpoint after seeing DEV.

## Development screen before formal evaluation

First verify exact native reload of the SELECT-selected BEST on a fixed
32-row batch: zero category changes, p99 probability drift ≤.005 and
maximum ≤.02. Then evaluate the complete typed DEV 1,600 and CSS pilot
1,430 using the same native protocol as the archived own Nox 1.0 4B. If
the Nox control lacks these exact development predictions, run it once on
the same panels. Require candidate `100*sqrt(T_dev*H_pilot)` at least 2.0
points above that same-panel Nox development proxy and no invalid or
overbudget excess above 1 percentage point. Report every Choice/Noul/Score
and CSS task loss, probability quality and Score ordinal bias even if the
aggregate screen passes. A failed screen stops this arm without searching
another checkpoint. Passing it only authorizes separately locked, post-key
same-panel JevArena v3/public231 predictions and an independent
corroboration plan, not a release claim.

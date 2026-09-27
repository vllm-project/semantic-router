# Official Qwen3.5 9B: first complete clean-v2 training arm

This one-epoch candidate arm is frozen after the successful source and
one-update preflight, before the first optimizer update. The preflight
checkpoint is not reused. The earlier own-Lux-origin BEST224 and Lux 1.0
controls remain unchanged.

## Fixed recipe and stop criteria

- Initialize a new 256-dimensional dynamic-option Decision head on the
  official general posttrained `Qwen/Qwen3.5-9B` revision
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Its source hashes,
  complete native length audit and exact two-start/one-update numerical
  admission are recorded in the linked
  [preflight](qwen35-9b-official-posttrained-native-preflight-2026-09-27.md).
  No third-party Decision model initializes these weights.
- Use rights-clean v2 TRAIN 7,455 SHA-256
  `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`,
  SELECT 700 `32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6`,
  CAL 700 `3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a`.
  All inputs fit the 8,192 token ceiling; source-tokenized TRAIN has
  4,194,465 tokens. CAL is isolated from checkpoint selection.
- Train exactly one epoch, **466 optimizer updates**, seed 20260926,
  microbatch 1, accumulation 16 with partial final window, BF16 backbone
  computation and FP32 parameters/head/loss, gradient checkpointing.
  LoRA rank16/alpha32/dropout .05 at peak LR `1e-4`, head at `2e-4`,
  CE+0.5 Brier, AdamW weight decay .01, clip norm 1.0, warmup ratio .05,
  cosine tail. No replay, teacher, extra data or preflight optimizer state.
- Save/evaluate SELECT at updates 64,128,192,256,320,384,448 and 466.
  Select exactly one BEST by SELECT family-macro accuracy descending,
  normalized Brier ascending and earliest step. No DEV/CSS/public/formal
  checkpoint search. One reserved GPU and **4.0 GPU-hour** wall cap including
  loading, inference and saves. Stop on source/data mismatch, invalid task
  type, nonfinite loss/gradient, OOM, failed save or time cap; keep failure
  artifacts without changing this recipe.

## Development gate before formal prediction freeze

First reproduce a fixed 32-row native SELECT batch after BEST reload:
zero category changes, p99 probability drift ≤.005 and maximum ≤.02.
Then run complete typed DEV 1,600 and CSS pilot 1,430 under a documented
native adapter. Compare with own Lux 1.0 on the same development panels;
reuse existing complete same-condition control predictions if available.
Require candidate `100*sqrt(T_dev*H_pilot)` at least 2.0 points above the
matched Lux control and no invalid/overbudget excess above 1 percentage
point. Report all task/type regressions, probability quality, Score ordinal
bias and language coverage. A failed gate stops this arm without selecting
another checkpoint. A pass authorizes a separately locked post-key
JevArena v3/public231 evaluation versus Lux and a pinned near-size open
peer; it is not itself a first-release result.

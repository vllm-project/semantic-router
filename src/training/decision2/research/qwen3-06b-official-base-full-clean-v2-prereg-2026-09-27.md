# Official Qwen3 0.6B: first complete rights-clean v2 arm

This protocol is frozen after the native source/one-step preflight and before
any update in this independent full run. It is a candidate-development arm,
not a release or a fresh blinded formal evaluation. The prior one-step smoke
weights will not initialize this run.

## Inputs and budget

- Initialize a new dynamic-option Decision head on official
  `Qwen/Qwen3-0.6B-Base` revision
  `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`; see the linked
  [preflight](qwen3-06b-official-base-native-preflight-2026-09-27.md) for
  source-file fingerprints and proof of all three native task types.
- TRAIN rights-clean v2 7,455 rows,
  `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`;
  SELECT 700,
  `32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6`;
  CAL 700,
  `3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a`.
  CAL is isolated from training and checkpoint selection. Original TRAIN
  contains 4,094,489 encoded tokens under this source tokenizer, including
  English and Chinese, and all 7,455 inputs fit the declared 8,192 limit.
- Train the full model for **one complete epoch, 466 optimizer updates**,
  microbatch 1, accumulation 16 (last window partial), seed 20260926,
  CE plus 0.5 Brier, AdamW weight decay .01, gradient clip 1.0, BF16
  backbone compute, FP32 parameters/head/loss, gradient checkpointing on.
  Backbone peak LR `2e-5`, head peak LR `2e-4`, warmup ratio .05 and the
  trainer's cosine tail. No replay, teacher, source weight or added rows.
- Save and evaluate SELECT after updates 64, 128, 192, 256, 320, 384, 448
  and the final 466. Select a single BEST checkpoint by SELECT family-macro
  accuracy, then lower normalized Brier, then earliest update. Do not pick a
  different checkpoint after inspecting typed DEV, CSS pilot or public data.
  Do not transfer one-step smoke optimizer state.
- Use one reserved GPU. Budget cap **2.0 GPU-hours** including SELECT
  evaluations and saves. Stop on nonfinite loss/gradient, OOM, source/data
  mismatch, absent type, failed save, or this cap; preserve every receipt and
  mark the run incomplete. No recipe change inside this arm.

## Open-development gate and next comparisons

After the fixed BEST checkpoint passes exact native reload, run the same
typed DEV 1,600 and CSS pilot 1,430 adaptation as the archived Kai 1.0
control. Compare `100 * sqrt(T_dev * H_pilot)` against Kai 1.0 with the
four-family typed macro and median task macro-F1; require a gain of at least
2.0 points, typed DEV accuracy above Kai 1.0, and CSS pilot median task
macro-F1 no more than .01 below Kai 1.0 before spending the full formal v3
budget. Report all type/task losses, invalid answers, Score ordinal behavior
and calibration. This gate is a development screen, not a formal-release
claim. A clearly weaker result stops this arm without searching another
checkpoint. If the gate passes, lock the package and separately preregister
JevArena v3 (8,147 original questions) and public JevBench231 predictions,
including own Kai 1.0 and the selected near-size open peer on the same panel.

The previous 0.6B formal set has already been keyed in this project. A later
formal result will be labeled a post-key same-panel comparison, with an
independent corroboration requirement for any release claim. Historical
Decision Index values guide peer selection only and never enter these scores.

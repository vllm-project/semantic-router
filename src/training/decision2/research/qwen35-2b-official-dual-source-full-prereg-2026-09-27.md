# Official Qwen3.5 2B Base/Posttrained: matched complete arm

Conditional prospective protocol. Neither arm may start unless its own
independently frozen [native source preflight](qwen35-2b-official-dual-source-preflight-2026-09-27.md)
passes both zero-step starts, one-update finite-gradient smoke and the
same-batch reload gate. The preflight optimizer and checkpoint cannot seed
this training. The two arms differ only in their official source weights
and tokenizer files at the pinned revisions listed in the preflight; the
TRAIN/SELECT/CAL rows, architecture, optimizer, token count, native input
protocol and selection rule below are identical.

## Fixed budget and selection

- Each admitted arm independently initializes a new 256-dimensional
  dynamic-option head, seed 20260926, LoRA rank16/alpha32/dropout .05.
  TRAIN rights-clean v2 7,455 rows SHA-256
  `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`,
  SELECT 700 `32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6`,
  CAL 700 `3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a`.
  The two source tokenizers both yield 4,194,465 TRAIN tokens, zero
  overlength at 8,192, and the same 7,455 independent training rows.
- One epoch, **466 optimizer updates**, microbatch1/accumulation16 with
  partial final window, CE+0.5 Brier, AdamW decay .01, clip1.0, BF16
  backbone compute with FP32 parameters/head/loss, gradient checkpointing.
  LoRA peak LR `1e-4`, head peak LR `2e-4`, warmup .05 and cosine tail.
  No teacher, replay, extra rows or test-derived weights. Use one separately
  reserved GPU per arm. Budget cap **2.5 GPU-hours per arm**, including
  SELECT evaluations and saves.
- Save/evaluate SELECT after steps 64,128,192,256,320,384,448 and 466.
  The single BEST in each arm is selected by family-macro accuracy
  descending, normalized Brier ascending, earliest step. The two arms
  then both receive complete same-condition typed DEV1,600 and CSS
  pilot1,430, provided their selected checkpoint passes exact 32-row
  same-batch reload (zero categorical changes; p99/max probability drift
  ≤.005/.02). Neither zero/one-step SELECT nor typed DEV, CSS pilot,
  public231, formal v3 or CAL changes the intra-arm BEST choice.

## Development promotion and stop rule

Compare each complete arm's `100*sqrt(T_dev*H_pilot)` to pinned own Sol1
on the exact same development panel; reuse the existing complete Sol1
control. Require at least **+2.0 points** and no invalid/overbudget excess
above 1 percentage point for promotion to a separate gold-free post-key
formal prediction freeze. Also disclose per-type accuracy, Score level
bias and probability quality, all CSS task losses and language coverage.
If both arms pass, choose the higher development proxy; break a tie of
<0.2 point by lower typed DEV Brier, then the Base arm. This tie rule is
fixed before either complete arm starts. If neither passes, both remain
development HOLD and no formal evaluation or upload occurs. Do not rescue
an arm by selecting a different checkpoint after open development labels.

Stop on source/data mismatch, invalid native type, OOM, nonfinite loss or
gradient, save/reload failure or time cap. Preserve failed receipts and
never silently alter the recipe or task denominator. Already opened v3
labels make any future formal result a post-key same-panel comparison,
requiring independent corroboration before a broad release claim.

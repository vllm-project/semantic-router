# 0.6B official Qwen3 Base: prospective native preflight

The own-Kai continuation lost JevArena v3 composite score against Kai 1.0, and
its subsequent 16-step recovery failed independent development gates. This
preflight tests a different, permitted weight origin before committing a new
training budget. It is not a selected Decision 2.0 candidate or a release
score.

## Immutable inputs

- Official source: `Qwen/Qwen3-0.6B-Base` at
  `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, Apache-2.0. The HF CLI
  reported 596,049,920 source parameters; the direct loaded decision backbone
  and new head will be counted separately. The downloaded `config.json` SHA-256
  is `504a6b58c4271583724e66584b6b7698aea18450209df6b2f7582df0e89cee59`;
  the weight-file SHA-256 is
  `cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba`.
- TRAIN: rights-clean v2, 7,455 rows, SHA-256
  `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`.
  SELECT: 700 disjoint rows, SHA-256
  `32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6`.
  CAL: 700 disjoint rows, SHA-256
  `3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a`.
  CAL is not used for initialization or model selection.
- Source implementation: signed `6e7314d00`; `DecisionModel` uses the native
  Qwen3 text body with the existing dynamic-option head, without a third-party
  Decision initializer. Prompt and scoring contracts remain the existing
  Decision 2.0 segmented-option implementation.

## Admission sequence

1. Confirm all TRAIN/SELECT/CAL rows satisfy source/group separation, the
   pinned hashes, each task type, and complete nontruncated native inputs at
   the declared maximum length. Record the complete token-length distribution
   and any unfit rows; do not silently drop them.
2. On one exclusively reserved GPU, run two independent fixed-seed zero-step
   source initializations and complete SELECT700 native inference. Require all
   700 valid, identical prompt/token IDs, zero categorical changes, finite
   probability vectors, and maximum probability drift at most `1e-4`.
   Inspect one Choice, Noul and Score row in each pass. The combined load and
   inference budget is `0.15` GPU-hour.
3. Only after step 2 passes, run a single one-update smoke with the proposed
   full-training optimizer. Require finite loss/gradients and an exact
   checkpoint reload on a fixed 32-row selection slice. Cap this step at
   `0.05` GPU-hour. A failure is recorded and stops the arm.
4. Freeze the full 0.6B training budget, checkpoint selector and independent
   development gate in a separate run protocol before any further update.
   Initial proposed comparison is one epoch of the same rights-clean v2 rows,
   with the 0.8B control's CE+Brier, microbatch/accumulation and learning-rate
   schedule. The official Qwen3 token exposure and actual selected row count
   must be measured before calling it an equal-budget architecture ablation.
   Subsequent development uses typed DEV and CSS pilot together; a SELECT-only
   gain does not authorize formal v3 or publication.

JevArena typed FINAL/CSS15 and the public231 comparison are not read during
this preflight. Earlier own-Kai and Bosun v3/public231 measurements retain
their historical meanings. Any post-key formal run is reported as such.

## One-step reload comparator correction, before retry

The first saved-checkpoint probe used a sorted-by-ID 32-row slice. Its batches
paired different items and padding lengths from the trainer's complete
SELECT700 pass, so the measured max probability drift of `0.0051023` with no
category changes is **HOLD for that probe**, not evidence of a model-weight
serialization error. The planned same-condition comparison must preserve the
trainer's original SELECT order and adjacent batch-size-two pairs. Signed
comparator code is corrected before rerun; the checkpoint, weights, source,
data, threshold and frozen 32-row count remain unchanged. The old failed
receipt is retained. A passing same-batch comparison would establish only
same-condition reload parity; batch-shape sensitivity remains a separate
diagnostic and is not hidden by this correction.

## Completed preflight receipt

The source tokenizer admitted all 7,455 TRAIN, 700 SELECT and 700 CAL rows at
8,192 tokens without truncation. TRAIN has 4,094,489 encoded tokens, median
154, p95 2,726 and maximum 6,559. The read-only audit JSON SHA-256 is
`af07b66f734edf2069d40c1a98b32cdefea8d9c77147feeab6672a0466041a59`.

Two separate source starts returned 700/700 valid SELECT answers, including
Choice, Noul and Score, with zero category or probability changes. Their
prediction files have the identical SHA-256
`e62731df7a8b9f35c6ecda17aa763b1dd30700522b3cd8c082f8063f2829f5c3`.
The baseline was 232/700, family macro accuracy .278048. These are
source-contract checks, not evidence of 2.0 quality.

The one-update smoke completed from the same official source with finite loss
1.43413, gradient norm 27.68713, 16 examples and 4,977 tokens. Its SELECT
result was 244/700, family macro accuracy .326040. The checkpoint saved and
reloaded. The first, sorted-slice comparator remains **HOLD** as described
above; a subsequent read-only comparison using the original adjacent
batch-size-two pairs returned **PASS, 32/32, zero category changes and zero
probability drift**. Its receipt SHA-256 is
`ddb012c254f5d017060d1583f06fadb08678086f4dbd3ae6d2ebef5aaff01660`.
No extra optimizer step was used to resolve the comparator discrepancy. The
zero-step and one-step runs remained within their separate preregistered
GPU-time caps. This only clears the native training and same-condition reload
preflight; the one-step checkpoint is not a selected or releasable model.

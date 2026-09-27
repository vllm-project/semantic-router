# Joyfox 0.8B: one fixed-data soft-replay transfer experiment

**Status: prospective design, execution HOLD.** This record does not select a
checkpoint or claim a Decision 2.0 release. It separates a loss-function
hypothesis from the already completed Base/posttrained initialization and
context-limit ablations. The soft-target cache, exact runtime image and final
execution receipt must be frozen before an optimizer step. No sealed typed
FINAL, CSS 15-task labels or JevArena v3 score is involved.

## Why this experiment

The matched Qwen3.5-0.8B Base/posttrained clean-v2 runs used the same 7,455
TRAIN rows and 466 updates. Posttrained initialization improved typed DEV from
392 to 509/1,600, while CSS pilot fell from 481 to 439/1,430 and public231
fell from 129 to 118/231. This is a developmental transfer reversal, not a
causal result about posttraining in general. Eos 1.0 continuation gained only
3/1,600 typed DEV items and worsened Brier/ECE. The fixed Joyfox source is the
strongest measured native 0.8B typed comparator at 993/1,600, but CSS pilot
median task macro-F1 is .30119. Extending its local context cap from 1,024 to
4,096 admitted 21 additional CSS pilot rows yet reduced median task F1 from
.30119 to .29699; some already admitted answers changed. Length alone is not
the observed transfer repair.

The completed Joyfox hard-label continuation trained the source model's native
head/backbone LoRA for 64 updates on a fixed 512-row rights-clean v2 sample.
Its SELECT family-macro accuracy moved only .668426 to .670926, below the
frozen +.015 gate; SELECT Score remained 32/90. No DEV, CSS pilot or public
prediction was produced for that checkpoint. A new Score data mix and a new
loss must not be changed together if the goal is attribution. This experiment
changes **only** the training objective to add source-probability replay.

## Gold-free CPU source and overlap audit

The owner-private rights-clean v2 TRAIN and manifest bytes matched SHA-256
`61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`
and `61aa883052759830c4ecf897b36c1062ad816c935a12db824c80abd1f80e9ee8`.
The CSS pilot prompt file matched
`598319a429de16c659b59ede0eac3c269939356b0599f4e08d1983e44def3dda`.
The read-only audit used no GPU or CSS labels. It found 0/1,430 pilot states
matching a TRAIN state after Unicode NFKC, case folding and whitespace
normalization. The existing builder manifest also reports zero raw, normalized,
ID, group, input-hash and bounded approximate near-context matches against
CSS pilot; its SimHash/SequenceMatcher rule is approximate and cannot rule out
semantic paraphrases or source-model pretraining exposure.

| Pinned TRAIN view | Count | Why it matters |
| --- | ---: | --- |
| Choice / Noul / Score | 3,908 / 3,031 / 516 | Score is only 6.9% of TRAIN. |
| Score with 3 offered levels | 102/516 | Three-level ordinal evidence is sparse. |
| Score from stage4 ordinal / targeted median / dense table / stage3 replay | 279 / 150 / 55 / 32 | Most Score rows come from two programmatic mechanisms. |
| Previously selected Joyfox sample | 512 rows, 488 source groups | Sample SHA-256 `ecb50a755c351c903a72a73a285d54622b141fce3e09809bf79d609ae8d2e532`. |
| Selected sample Choice / Noul / Score | 128 / 192 / 192 | Fixed for the proposed control/treatment comparison. |
| Selected Score from ordinal / targeted median / stage3 / dense table | 118 / 61 / 10 / 3 | 179/192 Score rows come from two mechanisms. |

SELECT's 90 Score items all use the targeted median family, which also occurs
61 times in the selected TRAIN sample. Thus SELECT Score is group-disjoint,
but **not mechanism-disjoint**. CSS pilot's three human-label tasks are from
other source datasets and contain only Choice; it tests source transfer, not
Score transfer. Typed DEV Score is a separate generator and is a development
cross-generator check, not an untouched release panel. The present audit
therefore does not establish broad Score generalization even if SELECT improves.

## Fixed treatment and control

* Initialize from the unmodified public
  `joyfox/Qwen3.5-0.8B-JEV@ae7b7040aeff7802f6f2bcfdd27f08a72d5cd969`
  package and its native QK decision head, with native inference source
  `joyfoxai/jev-inference@2677b5a3714489847668175de793e2d92fe183f0`.
  Use its released 1,024-token, no-truncation request contract. The source
  checkpoint's documented teacher-distillation lineage remains disclosed;
  this experiment makes no Jev API calls and uses no teacher corpus.
* Training **data are byte-identical** to the completed hard-label control:
  the 512-row JSONL hash above, parent TRAIN/SELECT/CAL hashes respectively
  `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`,
  `32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6`,
  and `3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a`.
  The previous sample manifest SHA-256 is
  `6d7d22309b0271b8e430f38f75bb4af65c16899d0dabcd278910bae14e79a0d4`.
  Keep all 512 IDs, order after the same seed shuffle, option keys, and native
  encoded tokens unchanged; record the exact sum of encoded tokens before
  either comparison. Group/lineage isolation from SELECT/CAL must still pass.
* Reuse the source-preserving rank-8/alpha-16/dropout-.05 backbone LoRA and
  frozen source head, seed `20260927`, BF16, microbatch 1, accumulation 8,
  **exactly 64 optimizer updates and one pass over these 512 rows**, AdamW
  LR `1e-5`, floor `1e-6`, eight warmup updates, cosine decay, weight decay
  `.01`, epsilon `1e-8`, gradient norm cap 1. The prior hard-label control is
  retained, not retrained. If its exact native runtime/software/LoRA target
  identity cannot be reconstructed, do not call it a matched control.
* The existing hard-label loss is categorical CE plus `.25` times summed
  squared probability error. For the sole treatment, add
  `0.2 * 2^2 * KL(p_source(T=2) || p_student(T=2))`, where each probability
  vector is a softmax over the **same native offered candidates** using logits
  divided by temperature 2. The source distribution is detached, finite and
  normalized. Produce it with the source in evaluation mode.
  Cache one source logit vector per frozen training row in a private artifact;
  bind its SHA-256, model/source revisions, option order, row IDs and input
  hashes before training. Source-cache inference cost is reported separately;
  both optimizer arms process the exact same encoded training tokens.

## Numeric preflight and fixed evaluation

1. Before optimizer step 1, verify exact source/model/head/tokenizer hashes,
   sample and cache identities, and package rights. Choose 12 Choice, 10 Noul
   and 10 Score rows by SHA-256 rank of `joyfox08-replay-smoke-v1:<row-id>`
   within each type. Replace the last selected row of each type with that
   type's longest native-encoded sample if it is absent, breaking length ties
   by the same SHA-256 rank; freeze the 32 IDs and their hash. Run two
   independent source-native processes on one physical GPU.
   Require zero categorical changes and maximum absolute option-probability
   drift at most `1e-6`; otherwise STOP and retain both receipts. The treatment
   with zero-initialized LoRA must reproduce the source logits with max
   absolute drift at most `1e-6`, have finite loss and nonzero finite LoRA
   gradient. Record exact runtime image ID and dependency versions. A
   mismatched source runtime is not repaired by choosing a favorable process.
2. Save step 64 only as the **fixed** candidate. Log step 32 for instability
   diagnosis but do not choose it. Stop immediately for nonfinite loss or
   gradient, token/row mismatch or changed source output. Score step 0 and
   step 64 on the unchanged SELECT700 once; report by type/family, Brier and
   invalidity. SELECT is an internal monitor, not evidence of unseen task
   transfer and not a license to change the frozen step.
3. Evaluate that fixed treatment, the saved hard-label control step 64 and
   the source on the same typed DEV1,600 and CSS pilot1,430 once with their
   native 1,024-token path and **raw uncalibrated probabilities**. Invalid,
   missing and overlength answers are failures. Do not fit temperatures on
   DEV/CSS. Check Score on the 400-item typed slice and task-median macro-F1
   plus each task's F1 on CSS. Report paired group/task intervals and the
   input-length/validity strata. This open developmental comparison diagnoses
   the loss intervention; it does not produce v3 release evidence.
4. Advance this arm for later private CAL fitting and package work only if
   fixed-step treatment exceeds source by at least 32/1,600 DEV correct,
   gains at least 8/400 Score correct, loses at most 8/400 in each other typed
   family, improves CSS median task macro-F1 by at least `.02` over source,
   has no individual CSS task F1 drop exceeding `.01`, and has no higher
   invalidity or worse median CSS Brier than source. Additionally require
   treatment to beat the matched hard-label control on both DEV Score and CSS
   median F1; ties do not qualify. These thresholds are conjunctive and must
   not be relaxed after results. If any fail, record a negative arm and leave
   0.8B publication HOLD; no public231 or sealed v3 evaluation is triggered.

Only after the open developmental gate passes should the fixed step-64
checkpoint receive a single CAL700 temperature fit and public231 diagnostic.
The public subset must exceed Eos 1.0's 142/231 with the native 1,024-token
contract; it is not a selector or official closed-set rank. Full package
parity, rights review, matched 1.0/open controls and the separately frozen
JevArena v3 8,147-item protocol remain necessary for any release.

The causal conclusion is narrow. A positive result would support adding
source-probability replay **for this fixed sample and source**. A negative
result would not prove replay useless: the sample's two dominant Score
mechanisms, SELECT's same-family Score, English-heavy human training and
Joyfox's 1,024-token cap remain independent bottlenecks. The next data arm
would have to change Score mechanism coverage and real-label provenance under
its own frozen, token-matched control.

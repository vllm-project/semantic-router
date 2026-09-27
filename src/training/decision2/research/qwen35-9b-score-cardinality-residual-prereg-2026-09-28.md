# 9B Score-cardinality residual: same-start diagnostic arm

**Prospective status:** implementation and CPU audit only. No treatment
optimizer update, typed DEV/CSS pilot, formal v3, public-231 scoring, or new
release claim has occurred. Freeze this one arm before GPU preflight. The
finished official Posttrained control, failed official Base run, own Lux1
control, and failed Lux short-rule run remain immutable.

## Mechanism and negative controls

The completed official `Qwen/Qwen3.5-9B` Posttrained control on the filtered
rights-clean v2 TRAIN has typed DEV Score **174/400**, with zero predictions at
the middle of three levels, and Noul **193/400**. Its SELECT Score questions
all use five levels. A fresh CPU read of its exact TRAIN SHA-256
`fe9c419a3e751e4a4173b5c90c49e0a83bba7e1c35c683441bb036138acb971c`
found **7,324 rows / 5,261 groups**. Of 507 Score rows, **99 independent
groups use three levels**, with label counts **30/33/36** at levels 0/1/2.
Noul has 2,993 rows with false/true labels **1,498/1,495**. This rejects a
simple training-label-prior explanation of the Noul `true` bias and shows that
the target three-level cardinality has very little training exposure.

Test one narrowly specified native architecture change: add a zero-initialized
logit residual indexed by the *number of offered Score levels* and their
semantic numeric keys. It is a 9×10 FP32 table, only 90 parameters; Choice and
Noul have exactly the old shared readout. At step zero, every output must be
identical to the completed control under the same source, seed, prompt, batch,
and runtime. After training, improvement restricted to three-level Score would
support a cardinality-dependent readout contribution; no gain or only a
five-level gain would point back to missing rule/state semantics or optimizer
transfer. This residual does not add reasoning capacity or new source facts,
and a positive result cannot establish general Score transfer by itself.

This is not the 0.6B type-separated head, its failed candidate-interaction
head, its failed ranked-probability loss, its failed gradient projection, the
9B short-rule replay, the failed Base initialization contrast, or the 27B
English Score-v6 data treatment. No third-party Decision model initializes it.

## Frozen matched cell

| Field | Treatment and completed control |
| --- | --- |
| Weight source | Official general Posttrained `Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a`; fresh shared 256-dimensional head, seed `20260926`. Source config, four shards and tokenizer use the hashes in the original [admission](qwen35-9b-official-posttrained-native-preflight-2026-09-27.md). |
| Data | Same 7,324-row / 5,261-group filtered TRAIN SHA above; SELECT700 SHA `32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6`; CAL700 SHA `3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a`. Exactly 3,579,176 native TRAIN tokens, maximum 4,089; no new rows, replay, teacher, weighting, truncation or source remapping. CAL is lineage-checked only before model selection. |
| Training | LoRA rank16, alpha32, dropout .05; CE + .5 Brier; LoRA/head peak LR `1e-4`/`2e-4`; batch1 × accumulation16; one epoch, **458 updates**, length4096, AdamW weight decay .01, global gradient clip1, warmup .05, cosine tail, BF16 backbone/FP32 head, no gradient checkpointing. |
| Selector | Evaluate/save at steps 64,128,192,256,320,384,448,458; one BEST by SELECT family macro accuracy, then Brier, then earliest. Do not substitute an earlier checkpoint after typed DEV or public results. |
| Readout | `score-cardinality` versus the archived `shared` control; the added table starts at zero, and its indices use Score semantic level IDs, not candidate display positions. |

## Ordered preflight and stop rule

1. Validate exact source/data byte hashes, split isolation, 7,324-row count,
   3,579,176 native tokens, 99 three-level rows, and source architecture.
   CPU contract tests must show exact zero-bias shared-head logits, Score
   permutation invariance, and no Choice/Noul change. Fail closed on any
   mismatch. The CPU audit uses no protected answer key.
2. On one newly checked idle authorized GPU and the same pinned runtime image
   as the completed control, compare treatment zero-step native probabilities
   on the first 32 SELECT items with the archived baseline prediction file
   SHA `f2838de7611c8d3296825f81ff41d839d3a1573863801e02e5ce400b0f800133`.
   Require identical ordered IDs/input digests, 32/32 valid, zero category
   changes, and maximum option-probability drift **≤1e-6**. On the three
   TRAIN-only Score-three-level inputs with the lowest ordered input SHA-256,
   compare the treatment's zero-bias
   native output with its contained shared head under the same batch;
   require the same gate. No gold score enters parity. Cap at **0.20 GPU-hour**.
3. If zero step passes, a distinct fresh one-update technical process may
   check finite loss/gradient, output save and fresh native reload. Its
   one-step LR schedule is different from the full 458-step schedule, so it
   is **not a performance control** and contributes no initialization weight.
   Require 32/32 unchanged SELECT categories and max reload drift ≤.02;
   cap at **0.15 GPU-hour**. A technical fault, nonfinite value, OOM or cap
   breach stops the arm without a full optimizer run.
4. Only after these gates, launch one fresh 458-update treatment on a reserved
   GPU, not a rerun of the archived control. Cap source load, training, SELECT
   and saves at **2.0 GPU-hours**. Stop on source/code/data mismatch,
   overlength, missing checkpoint, nonfinite loss/gradient, OOM, native fault,
   or cap. Preserve all failed receipts and do not relax the gates or retry
   another seed/checkpoint under this registration.

Do not use formal v3, JevBench public231, or Decision Index to select or
modify this arm. If the SELECT BEST is below **620/700** correct or family
macro below **.863426** (the completed control is 625/700, macro .873426),
stop before CAL and complete typed DEV/CSS pilot. Also require five-level
SELECT Score at least **84/90** (control 86/90) and all 700 native answers
valid. These are development retention gates, not evidence of rule transfer.

After a SELECT pass, reload only the frozen BEST, fit CAL only for calibration,
then make one sealed gold-free typed DEV1,600 and CSS pilot1,430 native readout
before their development keys. The mechanism screen requires three-level Score
at least **245/400** (control 174), Noul at least **193/400** and proxy
`100 × sqrt(T_dev × H_pilot)` at least **65.0** (control 61.255). Report all
type/task regressions, Brier, ECE and invalids regardless. These are
development milestones only. Formal-candidate promotion additionally needs
the unchanged earlier Lux1-relative target of approximately **72.326** proxy
and independent confirmation. If the milestone fails, record HOLD and do not
try other residual initializations, bias scales or checkpoint searches on the
opened development panels. A later arm requires a new frozen protocol and
source-disjoint evidence data.

This experiment has no JevArena v3 score and cannot inherit one from the
completed control. Actual GPU-hours, code hashes, saved model identity and
any failure will be attached in a separate result receipt.

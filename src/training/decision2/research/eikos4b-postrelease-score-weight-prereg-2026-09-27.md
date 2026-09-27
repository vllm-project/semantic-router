# Decision 2.0 4B Score weighting: prospective postrelease screen

**Status: design and CPU feasibility only.** No new selector, optimizer update,
model prediction, calibration fit, Hugging Face mutation, or claim of improved
Score exists under this protocol. The first 4B JevArena v3 labels and scores
have been opened; they are excluded from model selection and from any later
independent improvement claim. This experiment must not delay or retrospectively
alter the first-release decision.

## Evidence and causal question

The rights-clean v2 Eikos-4B control used 7,455 TRAIN rows, of which 7,418
were admitted by its unchanged one-pass candidate policy. Only 516 admitted
rows were Score. A read-only CPU census of the original, hash-pinned partitions
found 102 three-level Score TRAIN rows, with labels 0/1/2 represented 33/33/36.
The remaining 414 Score rows have four to eight levels. SELECT and CAL each
have 90 Score rows, all five-level. Thus SELECT/CAL cannot independently
identify a gain on three-level Score. The private aggregate-only CPU receipt
is SHA-256 `a990791462399c6b136ca9e9f8a81ed8654d00a49aeda1c75f29f86b4a27c42b`;
it read only TRAIN/SELECT/CAL, produced no model output, and did not read
protected evaluation labels.

The earlier **4B** 1.5x *human-source* loss arm kept the same 7,418 rows,
4,402,743 native input tokens and 232 steps. It raised frozen SELECT Score
from 75 to 78/90 but human-source correct from 331 to only 334/400, below its
preregistered 337 gate. It stopped before DEV, CSS pilot or release testing;
its checkpoint is not a candidate here. A separate **27B** English Score v6
data arm gained 11/192 on its consumed r2 selector, below its frozen 12/192
gate. Level 1 gained 23/64 relative to control while level 2 lost 11/64.
That r2 key and those model outputs cannot be reused for selection. Earlier
Score v1-v3 generated data failed blind shortcut checks; v4/v5 failed frozen
shallow-feature gates. These findings motivate an inexpensive, single-variable
test of Score training mass before constructing a large new curriculum. They
do not predict a 4B gain.

**Hypothesis:** raising the effective weight of the existing, rights-audited
Score rows modestly improves new three-level Score decisions without degrading
Choice, Noul, human-task transfer or probability quality. This tests sample
weighting only; it cannot establish that the old three-level examples are
semantically diverse enough.

## Fixed paired arm, subject to an independent selector gate

- **Control B:** preserve the completed, unmodified clean-v2 Eikos-4B uniform
  weight run and its native selected checkpoint 0232. Do not rerun it unless
  the prospective common-runtime parity test proves historical comparison
  invalid; in that case freeze a new matched control before treatment.
- **Treatment A:** from the same pinned Eikos-4B source revision
  `582ffb13f19a4da3f455e3db198584190bd7755b`, same fresh rank-8 LoRA
  and head initialization, exact TRAIN row order and admission, multiply only
  the 516 Score rows' per-example CE plus Brier loss by **2.0**. All other
  rows, including human sources, retain weight 1.0. Normalize by the summed
  weight of each unchanged optimizer window. Score's nominal row-weight share
  rises from 516/7,418 (6.96%) to 1,032/7,934 (13.01%); actual token share
  must be measured, not inferred from this ratio.
- Freeze one epoch, seed `20260926`, 232 optimizer steps, original batch order,
  maximum native length 8,192, microbatch 2, accumulation 16, BF16 backbone,
  FP32 rank-8 LoRA/head, learning rate `2e-5`, AdamW, Brier coefficient .25,
  and the same stable PyTorch-reference native letter readout and calibration
  policy. **Use fixed final step 0232**; no best-of-checkpoints search, seed
  sweep, data substitution, changed prompt or changed Score projection.
- The exact TRAIN/SELECT/CAL SHA-256 values are, respectively,
  `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`,
  `32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6`,
  and `3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a`.
  The rights manifest remains
  `61aa883052759830c4ecf897b36c1062ad816c935a12db824c80abd1f80e9ee8`.
  No v6/v7 authored Score row or held-out answer enters training.

The arm is **not GPU-eligible yet**. Before code or training, independently
freeze a new Score selector and the exact baseline prediction, model, loader,
runtime, tokenizer, loss and scorer hashes. The prior original adapter's
initial tensors were not archived; as in the earlier matched 4B screen,
same-source/seed evidence must be supplemented by exact zero-step native
probability parity, and this limitation disclosed.

## Minimum new selector and contamination boundary

Create **at least 80 independent English source scenarios, each with a
coherent 0/1/2 triplet**: at least 240 Score answers and exactly 80 examples
at each level. Use at least four distinct semantic mechanisms and at least
20 independent scenarios per mechanism. Candidate mechanisms may include
evidence-state supersession, cross-document provenance, scoped exceptions
and unresolved-versus-verified prerequisites. They need new case structures,
renderers and documents; no r1/r2/v6 template or case may be paraphrased into
this selector. A separately reviewed Chinese extension is desirable, but no
Chinese improvement claim or Chinese selection gate is inferred from an
English-only minimum.

First freeze scenario IDs, source rights, casebook, generator/renderer,
structured oracle, a second independent answer derivation, target balance,
review protocol, scorer, seed and all hashes. Hold triplets together. Screen
the entire selector against rights-clean TRAIN/SELECT/CAL, all prior Score
curricula and selectors, typed DEV and the now-open v3 prompts, CSS pilot and
CSS15 prompts, JevBench public, and available authored/multilingual rosters.
Require zero exact raw or normalized context/full-prompt matches and zero
unresolved bounded near matches; quarantine a full group on an overlap or
uncertain semantic duplicate. Model authors must not see the key or any
per-item old v3 error. Two gold-blind reviewers independently solve complete
triplets and flag ambiguity, source necessity, realism, label leakage and
one-field, length, order or count shortcuts; adjudicate discrepancies before
opening the oracle key. All 240 must pass dual-oracle and editorial admission.
If fewer than 80 groups or 240 rows remain, **stop without GPU training**.
This is a development selector, never a release-panel component.

## Execution and stop gates

1. CPU preflight must reproduce the exact 7,418 admitted rows, 516 Score rows,
   4,402,743 native input tokens, 232 ordered windows and all source/rights
   hashes. Unit-test the loss profile on Score/non-Score rows and per-window
   normalization. Compare original control execution code semantically, pin
   the final source/image hash and retain any mismatch as a no-run.
2. Before optimizer step zero, require identical native predictions and option
   probabilities for all 700 SELECT inputs versus the source control: zero
   category differences, maximum probability drift `<=1e-6`, no truncation,
   and matching input identities. If the stable inference backend cannot
   reproduce the original control on a predeclared gold-free roster, freeze a
   new common-runtime B control before A; never compare unlike paths.
3. Stop for nonfinite loss, changed admitted rows/tokens/batch schedule,
   missing/invalid native answers, or incomplete 232-step execution. Do not
   select another checkpoint or relax thresholds afterward. Record allocated
   GPU-hours, source/checkpoint hashes and complete execution receipts.
4. Seal both final model packages and *all gold-free* selector predictions
   before opening selector labels. Compare A versus B by complete source
   group, invalid answers counted wrong. Advancement requires **at least
   15/240 more correct** (6.25 percentage points), a positive lower bound in
   a prespecified paired 80-group bootstrap, no decline greater than 2/80
   correct at either endpoint level 0 or 2, and no new invalid answer.
   Parent SELECT700 Choice and Noul may each fall by at most 2 percentage
   points, Score90 may not fall, and family-macro Brier may worsen by at most
   .02. These thresholds are conjunctive, fixed before selector prediction,
   and do not inherit the consumed v6/r2 pass.
5. If and only if step 4 passes, run **one** post-selection diagnostic on the
   already open typed DEV1,600 and CSS pilot1,430, with the same frozen native
   A/B packages. Require Score DEV at least 8/400 better than B; Choice and
   Noul each no more than 2 percentage points below B; CSS pilot median
   task macro-F1 at least B minus .01, with no task worse by more than .03.
   These are exposed development checks, not independent proof; they do not
   permit retrospective weighting, checkpoint, prompt or temperature changes.

## Fresh evidence for a later improvement claim

Old v3 typed FINAL and CSS15 labels are now open. Re-evaluation there may be
reported as an explicitly exposed regression check only. A postrelease 4B
Score improvement needs a **new, prospectively frozen v3.1 release panel**:
unseen authored groups (separate generator team and case universe from the
selector, with Choice/Noul/Score coverage), plus newly selected human-labeled
source tasks whose IDs and source groups are disjoint from clean-v2 TRAIN and
all opened CSS panels. Freeze model packages, scoring formula, slice/transfer
floors and gold-free predictions before those new labels are accessed. Rerun
the released 4B control, A and corresponding 1.0 comparator in one protocol;
report paired group/task confidence intervals and all regressions. Public
JevBench231 and Decision Index can be separately rerun as attributed stress
tests, but their old open labels cannot substitute for independent v3.1.

If this cheap weighting arm fails its new selector, stop it and move to a
separate, preregistered data intervention such as evidence-state Score v7.
Do not rescue it with the old v6 treatment, consumed r2 key, 1.5x human arm,
temperature-only changes, or a search over multiple weights.

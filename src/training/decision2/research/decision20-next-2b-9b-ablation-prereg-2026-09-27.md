# Decision 2.0: preregistered 2B/9B recovery ablations

Status: **pre-registered; no result from these arms yet**. This note fixes the
comparison panels, resource cap, stop rules, and success criteria before a new
checkpoint is inspected. It is a development plan, not a claim about sealed
FINAL/CSS15, official JevBench, or open SOTA.

## Evidence that determines the intervention

All values below come from the same 1,600-item typed DEV and 1,430-item CSS
three-task pilot; public JevBench is the same pinned 231-item public subset.
The independently fitted CAL set varies by lineage and is never used to pick
weights. Each score uses native type probabilities/expected Score rather than
generated answer text.

| 2B checkpoint | Typed DEV | CSS micro | CSS median macro-F1 | Public231 |
| --- | ---: | ---: | ---: | ---: |
| Decision 1.0 Sol | 58.9375% | 38.4615% | .31554 | 161 |
| Targeted3024 BEST160 | 59.1250% | 42.5175% | .34763 | 161 |
| Direct clean-v2 BEST448 | 58.5625% | 40.3497% | .32337 | 164 |
| Targeted then clean-v2 BEST320 | 58.3125% | 43.9860% | .36183 | 163 |
| Structured replay8360 BEST288 | 58.6250% | 40.4200% | .32781 | unmeasured |

Targeted then clean-v2 gained CSS transfer but lost typed DEV, especially
rule/transition. Adding 2,536 structured rows to the human 5,824 helped
neither axis versus targeted. Those rows were 30.33% of training examples but
65.14% of input tokens; this is a plausible exposure imbalance, **not** a
causal attribution. The Qwen3.5-2B-Base clean-v2 64-update control is already
complete: SELECT700 family macro .5515 and DEV transition 0/400. It is a
poor short-budget starting point, so this preregistration does not rerun that
arm or describe a base-model recovery without further evidence.

| 9B checkpoint | Typed DEV | Rule | Set | CSS micro | CSS F1 | Public231 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Decision 1.0 Lux | 86.7500% | 64.50% | 82.75% | 53.78% | .57011 | matched receipt required |
| Human5824 low BEST192 | 86.9375% | 65.25% | 83.25% | **56.7832%** | **.580324** | matched receipt required |
| Structured8360 low BEST224 | **87.3750%** | **69.00%** | 80.75% | 55.5245% | .573976 | 183 |
| Natural8522 low BEST256 | 87.2500% | 66.75% | 82.25% | 55.5944% | .567303 | 182 |

At 9B, replay improved typed rule decisions but reduced CSS transfer against
the human-only arm. The Jev 1.13 hosted baseline on the common development
panel reached 88.75% typed DEV and 59.16% CSS; the user goal remains above
these local gaps, but hosted results are not a same-runtime latency comparison.
The 9B replay recipes therefore are not worth repeating unchanged.

## Arm A: one bounded 2B joint-curriculum test

**Question.** Does mixing a small amount of the audited targeted3024 TRAIN
with the rights-clean v2 human/natural TRAIN improve typed rule/transition
without the CSS loss of full structured replay or the synthetic forgetting of
sequential targeted→clean-v2 training?

1. Freeze the Decision 1.0 Sol 2B source revision
   `0665a41108e8f0b33a9515c98311c45947b99399`, the clean-v2 TRAIN7455
   SHA `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`,
   and the independently audited targeted3024 TRAIN source. Build one
   deterministic **group-complete, deduplicated** joint TRAIN. Retain all
   7,455 clean-v2 rows; select at most 1,500 targeted rows while capping their
   Sol-tokenized input share at 20% of the mixed corpus. Stratify targeted
   selection across Choice/Noul/Score, rule and transition, with fixed seed.
   Record selected IDs, source SHA, type/family/token counts, rights, and all
   exact/near quarantine counts. Abort if a required group crosses
   SELECT700, CAL700, DEV1600, CSS pilot1430, gold-free CSS15, or pressure
   panels; never read sealed labels. Do not silently relax the token cap or
   truncate source contexts.
2. Use the same SELECT700 SHA
   `32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6`
   and CAL700 SHA
   `3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a`
   as both prior clean-v2 2B arms. Initialize directly from Sol 1.0, avoiding
   another unverified LoRA merge. Rank-16 LoRA, alpha32, dropout .05, CE +
   0.5 Brier, LoRA LR `1.5e-5`, head LR `7.5e-6`, microbatch1,
   accumulation16, max length8192, seed20260926; no teacher-probability KL.
   Run **at most 128 updates** on one available GPU; save/SELECT every32.
   This is a screen, not a full one-epoch result.
3. At step128, stop without CAL/DEV/CSS if SELECT700 has fewer than
   **570/700** correct (the completed direct-v2 control), or GoEmotions
   Choice <155/200, or GoEmotions Noul <175/200. These are absolute
   precommitted gates. If the screen passes and the assigned GPU remains
   available, extend the same exact optimizer run to at most one epoch; SELECT freezes
   BEST by family-macro accuracy, normalized Brier, then earliest checkpoint.
   CAL700 then fits native per-type temperatures. CAL is never a checkpoint
   selector.
4. For any arm that passes the early screen, report the complete
   DEV/CSS pilot/public231 and validity receipts even if the post-CAL scores
   disappoint. Promotion as a broad 2B release candidate requires
   DEV ≥59.125%, CSS micro ≥43.986%, CSS median macro-F1 ≥.36183, public231
   ≥164/231, 100% native-valid predictions, and no lower performance on any
   CSS pilot task than the targeted→v2 arm. Treat a one-item public change as
   descriptive until paired uncertainty is computed; do not claim SOTA from
   the public subset.

The Arm A control is the **same source + same SELECT/CAL** direct clean-v2
run, with targeted data the only intended intervention. The sequential
targeted→v2 arm is a second relevant comparator, but it uses a separately
audited, numerically distinct FP32 materialized warm start. Historical
TARGETED3024 and human5824 SELECT scores are not directly comparable because
their selection sets differ.

## Arm B: 9B token-balanced hard-rule replay, contingent on Arm A

Keep the strongest CSS-transfer 9B human5824 low BEST192 and the completed
structured/natural replay arms as the comparison set. Do **not** launch B
while the 27B candidate, review experiments, or existing services occupy its
assigned GPU. Build a source-group-complete 9B TRAIN variant from the audited
human5824 and at most 512 short rule/transition examples selected only from
eligible TRAIN splits. The replay additions must be ≤20% of 9B-tokenized
input mass and individually ≤1,024 tokens; if too few rights-audited,
nonoverlapping examples exist, abort instead of filling with long structured
contexts. Use the exact same SELECT600 and hard CAL900 as prior 9B replay;
start from the same Decision 1.0 Lux revision. Fix low-LR rank16 LoRA,
CE + 0.5 Brier, effective batch16, step32 SELECT, and a maximum **128
updates** (the same low-LR values as the human5824 run; freeze exact command
from its provenance before launch). This tests token-balanced hard-rule
rehearsal, rather than simply changing model size or repeating 8,360/8,522.

Early stop at step128 if SELECT600 family-macro is below the completed human
BEST192 or if either human-task SELECT slice drops by >2 points. If screened
in, fit hard CAL900 and compare native DEV1600/CSS1430/public231. Promotion
requires DEV ≥87.375%, rule ≥69%, set ≥83.25%, CSS micro ≥56.7832%, CSS
F1 ≥.580324, and no loss on any CSS task against the human5824 arm. These
thresholds intentionally require a Pareto improvement on the completed 9B
arms. A positive public231 delta must use identical prompts and paired
uncertainty, with a matching raw-prediction receipt.

## Architecture decision and interpretive limits

The instruct/post-trained Sol and Lux sources currently transfer better
than the observed short base initialization. A fresh base backbone remains a
future **separate** arm, with full-budget controls rather than claiming a
64-step base failure proves an architecture ceiling. The 2B and 9B source
families can diverge if independent matched evaluations justify it; the
published family name should be `dev-2.0-xxb`, with no internal code names.
No arm in this note uses the authored v3 panel as a release gate because its
independent editorial review blocked it. The redesigned v4 panel also remains
blocked until a new blind editorial pass and human quality review.

## Pre-training feasibility amendment: Arm A0 blocked, Arm A1 proposed

The original Arm A targeted3024 mix above **must not be launched**. A
gold-free source audit found that 1,553/3,024 targeted rows already occur in
clean-v2 with identical group, row ID, and input digest. The 1,471 remaining
rows are mostly attributed stance (500), dialogue function (500), quantized
median (250), and interval conjunction (150). They provide little of the
rule/transition replay that Arm A was meant to isolate. This is a data
feasibility failure discovered before any Arm A training or SELECT readout;
the pre-registered hypothesis cannot be tested with that mix.

Arm A1 is a **new** controlled screen, not an edit to A0 results. Starting
from the pinned legacy 6,000 TRAIN and removing exact clean-v2 groups/IDs/
inputs, a tokenizer audit found 296 complete groups of one row each in 13
structured reasoning families, all ≤1,024 Sol tokens and totaling 141,883
tokens: Choice149, Noul84, Score63. The major families are arithmetic69,
ordinal56, Boolean55, registers40, plus 76 mapping/policy/transition/logic/
evidence/rubric/authorization rows. This is a real but small intervention;
only seven rows explicitly name the stage3 transition-set replay. It cannot
honestly be described as 296 independent transition or rule-precedence cases.

Before training, a builder must quarantine exact and approximate near-context
neighbors of SELECT700, CAL700, DEV1600, CSS pilot1430, gold-free CSS15,
pressure, public benchmark, and authored prompts. Keep complete source groups;
abort if fewer than 256 rows remain or added input tokens exceed 20% of the
merged corpus. No protected gold enters that audit. Use the same Sol 1.0
source, LoRA objective, SELECT700/CAL700, 128-update GPU3 cap, step32
selection, early stop thresholds, native calibration, and promotion gates
listed for A0. The sole intended training intervention is the audited
≤296-row short structured replay; the matched direct clean-v2 control uses
the identical base/source/holdouts. Record this amendment and its data SHA
before step one. If the builder fails either gate, do not train another
mixture opportunistically under the A1 name.

## Frozen A1 data and prospective step-128 screen correction

The A1 builder retained 270 complete TRAIN groups after its protected-panel
quarantine: Choice125, Noul84, Score61. Its 7,725-row merged TRAIN has SHA-256
`f2390fe0c5540c39d7ad8886fe9fc35054f253398092a841fb25bd411e894baa`;
the data manifest has SHA-256
`02cdf7e644e8f5fc75f6a908ceb5b6dcc6a7d7e172c047b0bcfc684d4e73be13`.
The added rows are 135,143 of 4,329,608 Sol input tokens (3.1214%); all
exact and approximate near overlaps detected against the 56 protected
sources were removed. This limited exposure makes A1 a small sensitivity
test, not a strong new curriculum. The manifest retains all selected IDs,
input hashes, source terms, and source-specific audit counts.

The A0 step-128 threshold of 570/700 incorrectly compared a 128-update screen
to the completed direct clean-v2 **BEST448** (570/700). Before A1 training or
SELECT inspection, the A1 screen is corrected to the same-source clean-v2
**step128** reference: 538/700, family-macro .707407, GoEmotions Choice
143/200 and Noul 177/200. A1 continues past step128 only if it reaches
**at least 545/700**, family macro **at least .713**, Choice **at least
140/200**, and Noul **at least 174/200**. This asks for seven more correct
and a positive family-macro shift while allowing at most three errors per
human slice. Passing permits the original one-epoch cap; it is not evidence
of a material transfer improvement. Failure ends the arm before CAL/DEV/CSS.
The original promotion gates remain unchanged.

## Optimizer-schedule correction before A1 step one

The trainer fixes `planned_updates` and the learning-rate schedule in its
checkpoint contract. A run started with `--max-steps 128` cannot resume to a
longer horizon with the same optimizer contract. A1 therefore starts with
`--epochs 1` and **no** `--max-steps`, as did the direct clean-v2 control.
Its 7,725 examples imply 483 planned updates, compared with 466 for the
7,455-example direct control. Only the frozen step-128 SELECT result controls
the screen above. An external watcher pauses the task-owned container after
the complete step-128 checkpoint appears; if the screen fails, the run stops
there and any subsequently started partial optimizer window is discarded.
If it passes, the same process continues to at most step483. The additional
17 updates are disclosed in any full-run comparison; this arm does not claim
strictly equal compute with the direct control.

## A1 frozen step-128 result and stop decision

The A1 run used the pinned Sol 1.0 source, the frozen 7,725-row TRAIN and the
same 700-row SELECT/CAL as direct clean-v2. The container image digest was
`f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54`;
the trainer source SHA-256 was
`0315661042a3d78c77fd957d315fb6ee42c8b3ce99d25d3f39df133233b747e3`,
matching the direct clean-v2 control's recorded trainer. Source, selection,
and calibration SHA checks passed in the run provenance (SHA-256
`8191de42b0cd715416573f2101af052b3b775f95fceef1161992588fcd31c190`).
Both runs had the same pretraining SELECT baseline: 506/700 and family macro
.660833. The A1 run was paused after complete checkpoint128, screened on
that frozen checkpoint, and stopped. Three subsequent in-flight optimizer
windows reached training log step131 before the task-owned container stopped;
there is no later checkpoint or completion marker, and none of those windows
were used for selection. The complete checkpoint128 receipt SHA-256 is
`bb2ed4321657308cf697a4b40ad72c7ec50528d110`, with adapter SHA-256
`c49820d454b75ec817aba152fba2dc501b63780ccfc111b4e9083350f6b36046`
and head SHA-256
`db1ef1f005d4b6ec94f3b0cfc02597a10410d3276a1b84a24f8288be6c219d8c`.

| SELECT700 checkpoint | Correct | Family macro | Macro Brier | Human Choice | Human Noul | String composition | Quantized median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct clean-v2 step128 | 538 | .707407 | .169612 | 143/200 | 177/200 | 8/40 | 40/90 |
| A1 short replay step128 | 540 | .709352 | .170669 | 146/200 | 178/200 | 9/40 | 37/90 |

A1 missed the preregistered **545/700** and **.713** continuation gates;
the human-slice floors passed. It is eliminated, with **no CAL fit, no
DEV/CSS/public231 evaluation, and no 9B Arm B launch** from this evidence.
The 700 raw prediction IDs, prompt digests, and token IDs matched the direct
control exactly. Paired outcomes had nine wrong-to-right and seven
right-to-wrong flips, net +2; a 5,000-draw bootstrap over 455 SELECT source
groups (maximum group size two) gave micro-accuracy delta 95% interval
[-0.72, +1.30] percentage points and family-macro delta interval
[-1.82, +2.26] points. This is a diagnostic interval on SELECT, not proof of
generalization. Raw A1 metrics and predictions have SHA-256
`33f93dfbd413b32502ef61d24f67b332344e0835d9bd1d17e0ca24caa6e16972`
and `aeaa9a3266aa1455d3a3d48cd9d9196832eef8067d4348f133cb28fe0f3ed581`.

This result argues against repeating tiny, mostly arithmetic replay as the
main 2B/9B recovery intervention. The A1 data added only 3.12% input-token
mass and seven explicit stage3 transition-set rows; the observed +2 SELECT
items do not support a broad model-family claim. A larger intervention needs
new, genuinely diverse rights-audited rule/transition evidence and a frozen
equal-compute control before GPU training. No sealed FINAL/CSS15 gold was
opened for this arm.

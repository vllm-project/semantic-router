# AutoJev 27B teacher: full TRAIN Choice/Noul source audit

**Prospective freeze before GPU inference.** This is a private, descriptive
teacher-source audit on the original TRAIN partition. It is not student
training, an evaluation score, a release gate, or evidence of independent
transfer. The earlier 96-group pilot used a small sample; this run uses every
Choice and Noul TRAIN row in file order, with no answer-based filtering.

## Fixed inputs and CPU preflight

| Component | Frozen value |
| --- | --- |
| Teacher | `denis-pplx/autojev-27b@6f5b557e037f5edb25c7dc92dbc6553e5a19c015`; clean native source commit `ee63c1515980491a742f0bd0685c8dc5ca1f00c3`; package SHA-256 `d0b1e161c17d60889744b6ccb5fa588bf80f9856f8535e6a04e083ffcf667ca2` |
| TRAIN | Rights-clean v2 TRAIN SHA-256 `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`; rights manifest SHA-256 `61aa883052759830c4ecf897b36c1062ad816c935a12db824c80abd1f80e9ee8` |
| Exact roster | 3,908 Choice and 3,031 Noul rows, 6,939 distinct IDs and input hashes, 4,951 distinct groups; gold-free identity/group/type/source/family/language/option-count SHA-256 `21f7b73c20107d3aa00e525d0f670963f8e8347d3ef7ca9916854a8400268d75` |
| Language | English 5,792; Chinese 1,147; no other languages represented |
| Choice options | 2–128 options, with 2,612 four-option rows; no request exceeds the native 255-candidate limit |
| Gold class buckets | Choice first 917, interior 2,057, last 934; Noul true 1,517, false 1,514. These are audit strata and the gold never enters the teacher request. |
| Native input length | Pinned processor/rendering: minimum 84, median 158, p95 2,739, maximum 6,482 tokens; 0/6,939 exceed 8,192. Token-count roster SHA-256 `9184f4277c388b27ee792b7e5807ac7845f897551f78e38c6aeea7c87744d4cc`. |
| Code | [`autojev_choice_noul_train_audit.py`](autojev_choice_noul_train_audit.py), signed source commit `6bc9a4875c397405f40c76007c43ca878b1400e3`, script SHA-256 `4909ed107fa4dcffaaef0ba2110a73af7eed698a4f72d179c2ab855867109b3f` |
| Runtime | One freshly checked idle GPU on an authorized node, pinned image SHA-256 `f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54`. Exactly one run, **3,600-second wall cap including load and verification**, so at most 1 GPU-hour. |

The CPU-only signed code mirror matched the script SHA above. The independent
native release verifier returned 26,086,635,760 loaded parameters, model
config SHA-256 `bacbcbb281a53af5ef5cc6c9028601097d155bf981129f18a727219517921dcd`
and runtime source tree SHA-256
`550ccd857350c6771a1de03e4bcba9fb4412247b58bcdc9e8ac03de2ab9641a5`.
The full-roster native processor dry run passed. No GPU teacher request has
been issued in this audit yet. GPU and live container occupancy must be
checked again immediately before the single run.

Source counts: GoEmotions official TRAIN 2,800; stage4 composition 1,653;
stage3 replay 612; CosmosQA 448; targeted original generator 450; SQuAD2
answerability 334; SNLI 272; original generator 250; FLUTE 120. These are a
mixture of genuine source labels, source-specific mappings and related
programmatic families. The source and language mix is not representative of
JevArena transfer. The full manifest assigns different upstream conditions
including CC BY, CC BY-SA, AFL, and internally generated records. All
row-level distributions are private research material; no source text,
vectors or restricted records are placed in a public repository. A future
student arm must recheck per-source use, attribution, retention and weight
redistribution conditions. The manifest's exact/near-context checks against
SELECT, CAL, CSS and named typed panels found zero collisions; this is not a
guarantee against semantic overlap or overlap with the teacher's own training
data.

## Locked native request, scoring and stop rules

Each call uses the original TRAIN state and one typed Choice or Noul question
with live options; the native released `DecisionModel.answer` supplies the
whole probability distribution. No chat transformation, answer text, label,
truncation, prompt search, temperature fit or alternate checkpoint is sent.
The runner checks the exact data/manifest/roster, released package/source
revisions, loaded parameter count, native response type, option identity,
finite nonnegative probabilities and normalization. Native context or
candidate admission failure counts invalid; an unexpected error aborts the
sole run. Missing/invalid answers remain in the 6,939-row denominator as
failures. No retry or sampler change follows a cap/preflight failure.

The private mode-0600 aggregate records `n`, valid, invalid, context or
candidate limit, ties, unique-max agreement, summed gold probability and
half Brier by type, source, family, language, type×option count and gold
class bucket. Ties count wrong. All metrics are descriptive TRAIN-oracle
agreement, not independent accuracy. A separate private mode-0600
distribution artifact may contain only row/input identity, group, type,
source/family/language, option count and teacher probabilities, with no raw
text or gold. It is written only if all 6,939 native outputs are structurally
valid and source conditions permit keeping it private for a separately
authorized future distillation study. The artifact itself does not promote or
train any student.

Success requires exact hashes, all 6,939 native outputs structurally valid,
no context/candidate overflow, full aggregate facets, and completion within
the 3,600-second cap. Otherwise record HOLD with the one-run failure; never
drop difficult rows or change the rubric after seeing outcomes.

The planned matched-budget experiment, if this audit supports it, starts an
eligible official-Qwen or own-Decision-1.0 student from a separately frozen
weight revision and compares hard-label control with hard-label plus
low-weight teacher KL. Same groups, token/update budget, masks and native
inference apply to both arms. Source-disjoint SELECT and later held-out
transfer, calibration and retention—not this TRAIN screen—decide usefulness.

**No full Choice/Noul teacher result exists at this prospective freeze.**

## Sole frozen run: result and limits

The one native run completed in **1,105 wall seconds = 0.3069 GPU-hours**
within the 3,600-second cap. All **6,939/6,939** responses passed the native
type, option-key, finite-mass and normalization checks; no context/candidate
overflow or invalid response occurred. The task-owned container exited and
its GPU returned to zero allocated memory. The mode-0600 aggregate SHA-256 is
`5c5dc9aca484569cb4ec03f5ce4d3a2ce93256f209f95b67d9eac2c282ef8488`.
The separately retained 6,939-row, mode-0600 private probability artifact
has SHA-256
`8cf211e5af88920de556dc84aa0fcb8b16677fdfd5209abb04db0ffe8e8b195c`;
its exact roster, TRAIN, script and model hashes were rechecked and it has no
gold, state or option descriptions. It remains private and is **not** a
student-training authorization.

| TRAIN slice | Rows | Unique-max agreement | Ties | Mean gold probability | Half Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| All Choice/Noul | 6,939 | 5,381 (77.55%) | 124 | 0.6985 | 0.1471 |
| Choice | 3,908 | 2,852 (72.98%) | 90 | 0.6414 | 0.1741 |
| Noul | 3,031 | 2,529 (83.44%) | 34 | 0.7722 | 0.1123 |
| English | 5,792 | 4,607 (79.54%) | 77 | 0.7213 | 0.1373 |
| Chinese | 1,147 | 774 (67.48%) | 47 | 0.5835 | 0.1968 |

| TRAIN source | Rows | Unique-max agreement | Mean gold probability | Half Brier |
| --- | ---: | ---: | ---: | ---: |
| GoEmotions official TRAIN | 2,800 | 2,214 (79.07%) | 0.7079 | 0.1479 |
| stage4 composition | 1,653 | 989 (59.83%) | 0.5016 | 0.2387 |
| stage3 replay | 612 | 516 (84.31%) | 0.7829 | 0.1015 |
| CosmosQA | 448 | 363 (81.03%) | 0.7441 | 0.1282 |
| targeted original generator | 450 | 450 (100.00%) | 0.9509 | 0.0092 |
| SQuAD2 answerability | 334 | 303 (90.72%) | 0.8670 | 0.0692 |
| SNLI | 272 | 227 (83.46%) | 0.7699 | 0.1257 |
| original generator | 250 | 219 (87.60%) | 0.8506 | 0.0719 |
| FLUTE | 120 | 100 (83.33%) | 0.6982 | 0.1110 |

Choice difficulty is highly mechanism-dependent. The preregistered exact
option-count strata can be aggregated without changing predictions: 2–4
options 2,414/3,076 (78.5%); 5–8 options 181/362 (50.0%); 9–17 options
183/335 (54.6%); and 31–128 options 74/135 (54.8%). The wide bins hide
heterogeneous families: for example 127-option items were 0/15 whereas
105-option items were 20/21. Among families with at least 25 rows, stage4
arithmetic was **60/224 (26.8%)**, automaton **46/137 (33.6%)**, relations
**149/274 (54.4%)**, and boolean **161/277 (58.1%)**. These weak training
slices argue against unconditional KL on every teacher distribution. The
targeted-generator perfect score is same-family TRAIN agreement, not
out-of-source transfer.

Noul is almost globally balanced (true 1,517, false 1,514), yet the teacher
agreed with false labels **1,372/1,514 (90.6%)** and true labels
**1,157/1,517 (76.3%)**. There is also source-level imbalance: SQuAD2 has
191 true/143 false, stage3 replay 49 true/71 false. A student should not
inherit the teacher's false preference through indiscriminate soft replay.

The overall English/Chinese gap is confounded by source composition. A
post-result CPU join of the unchanged private artifact and exact TRAIN labels
found stage4 composition at 852/5,792 English rows but 801/1,147 Chinese
rows. Within that source, agreement was 515/852 (60.4%) English and 474/801
(59.2%) Chinese; in the original programmatic source it was 174/199 (87.4%)
and 45/51 (88.2%). The aggregate 79.5% versus 67.5% is therefore **not**
evidence of a language-only teacher deficit. These crossed slices were
post hoc, use no new GPU/model calls, and remain exploratory.

The earlier 96-row pilot gave AutoJev 26/32 Choice and 26/32 Noul, compared
with Eikos 21/32 and 26/32 on that same pilot. The complete AutoJev Choice
result is lower than its small pilot, and there is no corresponding full-TRAIN
Eikos run; the pilot is not a full-data teacher ranking. The separate Score
audit found 324/516 (62.8%) unique-max agreement, with its own weak high-level
and dense-table slices. None of these TRAIN numbers demonstrates student
gain, JevArena performance, or model-release quality.

**Recommendation, not an executed arm:** the next eligible
official-Qwen/own-1.0 student ablation should compare a matched hard-label
control against the same hard labels plus low-weight, source-aware KL,
predefining how wrong/tied teacher maxima and high-risk stage4 arithmetic,
automaton, high-option and Noul-true rows are masked or downweighted. Keep
identical row schedules and token/update budgets; give the control matching
per-row weighting. Score, Noul true/false, long-input, calibration and
source-disjoint transfer need explicit gates. Use TRAIN labels only to form
training masks, never SELECT/CAL or formal labels. No student run was started
by this audit.

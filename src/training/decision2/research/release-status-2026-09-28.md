# Decision 2.0 first-release status, 2026-09-28

This is an internal research status, not model-card copy. JevArena v3 is the
8,147-original-item typed FINAL plus human-transfer panel; its answer count is
larger because some typed items contain multiple named questions. The 231-item
JevBench result is a separate public-subset reproduction. Development SELECT,
typed DEV and CSS pilot scores are not release scores. The v3 keys have been
accessed by the project, so subsequent runs on the same panel are post-key
comparisons, not newly blinded tests. All model repositories remain private.

| Size | Direct eligible start and current evidence | Release state | Next discriminating action |
| --- | --- | --- | --- |
| 0.6B | Official Qwen3-0.6B-Base full-backbone/shared-head package is private. Same-panel v3 **38.520** vs own Kai1 **35.938** and Bosun **38.524**; paired 95% gain interval vs Kai1 **[-2.032,+7.846]**. Public231 **143** vs Kai1 **114**, Bosun **133**. Choice **109/800** vs Kai1 **277/800** and Score **80/400** vs **98/400** regress; Noul **458/800** vs **404/800** improves. Fixed gradient projection completed 466 updates but failed SELECT at **547/700** versus ordinary control **562/700** and frozen threshold **569/700**; Score remained **32/90**, at **0.311 GPU-hour**. LogiQA 2.0 Choice source is HOLD after duplicate/ID audit. QuALITY's pinned source audit leaves an upper bound of **144 TRAIN article groups / 2,424 questions** and zero eight-role lexical overlaps, but independent answer-blind evidence-necessity review passed only **7/24** pairs. RACE's fixed 128-article/477-question screen found no eight-role lexical hits, but passages are short and independent blind review passed only **6/24** against a frozen **18/24** evidence-necessity threshold. Zero QuALITY or RACE rows are admitted. | Private first package exists; stronger SOTA/large-gain claim HOLD. No new projected package or formal score. | Keep QuALITY and RACE source-wide arms HOLD. Prioritize truly evidence-dependent Choice and independent Score/transfer evidence; a prospective item filter requires new independent review. Do not repeat failed projection, head, teacher or shortcut-prone data arms. |
| 0.8B | Own Eos1 continuation: development proxy changed only **+0.163** and Score fell to **85/400**. A Score-only OCNLI replacement failed CPU admission: longest possible 192 rows carried **31,789** native tokens against a frozen **155,681** minimum. A separate VitaminC real-revision Score source had 820 sampled rows across 256 pages but median native length **180**; its protected-role overlap is unresolved. TREC DL 2023 remains a supplemental diagnostic HOLD: **8** conflicting near-duplicate grade classes and only **1,604/22,327** judged rows joined to source text. A frozen HelpSteer2 correctness pilot received independent answer-blind review: only **17/24** absolute Score grades were within one human grade, and the reviewer flagged construct/factual ambiguity throughout; it remains TRAIN HOLD. | HOLD; no qualified formal 2.0 result. | Find a genuinely long, source-disjoint, quality-checked Score corpus, or separately pre-register an equal-token short-source control. None of the screened external sources is admitted. |
| 2B | Own Sol1 continuation: same-panel v3 **43.960** vs Sol1 **45.580** and Decider2B **49.499**; public231 **162** vs **161** and **175**. Official Qwen3.5 Base/Posttrained arms failed development promotion. A proposed Sol soft replay stopped at its frozen BF16 zero-step probability-parity gate before any optimizer update. Evidence Inference 2.0 TRAIN provides a possible human-labeled three-class source, but its middle class is not a proven ordinal effect magnitude and no new row is admitted. | HOLD. | Resolve the proposed source rubric, article rights and source-disjoint overlap before a matched-token official-Posttrained arm; retain the parity failure and formal result as recorded. |
| 4B | Official Qwen3.5-4B-Base: same-panel v3 **53.218** vs own Nox1 **56.470** and Decider4B **61.882**; public231 **171** vs **173** and **192**. Own Nox continuation's SELECT gain did not survive typed DEV. Independent ConTRoL blind reviews agreed on **27/30** relations; each matched source labels on **21–22/30**, corroborating three likely label errors and leaving one further conflict for adjudication. | HOLD. A third-party-initialized private research package is ineligible for the formal lineage. | Resolve the disputed source labels, embedded-passage rights and unscanned long-input overlap before any source admission. A matched arm must separately retain Score. No GPU training yet. |
| 9B | Official Qwen3.5-9B Posttrained shared-head development proxy **61.255** vs Lux1 historical DEV **70.326**; three-level Score **174/400**. A matched 90-parameter Score-cardinality residual arm completed 458 updates and selected BEST448 on SELECT **644/700**, but its single sealed DEV/CSS pilot readout was typed **T .808125**, Score **237/400**, human-transfer **H .501403**, proxy **63.655**. This improves the shared-head control's T .683125/Score174/proxy61.255, yet CSS H falls from .54927 and the predeclared Score≥245 and proxy≥65 promotion gates both fail. The new arm used **0.941439 GPU-hour** including technical probes and readout. Current Lux1 package passed a separate gold-free cross-process repeatability gate; an older different-bundle example remains invalid for current-package parity. | HOLD; no qualified formal 2.0 score, public231 run, or new 9B package. | Check own-Lux soft-distribution coverage, source overlap and zero-step parity as a separate preflight before any retention arm; the Score-cardinality result does not authorize one. A future promising candidate also needs a current-package Lux1 same-panel comparator. |
| ~27B | Official Qwen3.8-27B selected development proxy **68.53** vs native AutoJev27 peer **79.15**; Score **162/400**. Official Gemma4-26B-A4B arm did not pass SELECT. An independent Score-source CPU gate admitted **0/160** sampled TRAIN/SELECT items. The fixed **2,560-row/160-step** teacher contrast stopped at CPU admission twice: exact whole-group source/type quotas were infeasible. A later gold-free CPU envelope found **7,324/7,455** context-admissible rows and a prospective pooled source-quota alternative with minimal L1 shift **2**, but **1,400 groups span types** and teacher mask, rights and protected overlap remain unverified. A new pinned input-only inventory now projects all eight core roles, including TRAIN/SELECT/CAL, with **20,263** role records; 27 optional roles remain excluded. A distinct pooled-quota schedule is now materialized with 2,560 rows/507 Score and a two-row L1 shift, but 600 cross-type groups are partially selected; no teacher-mask, rights or actual full-input overlap admission has passed. GPU use for this addition is zero. | HOLD; no qualified formal 27B score. | Audit the materialized pooled-quota schedule for cross-type group treatment, complete-input overlap, teacher-mask thresholds, source rights and semantics before any GPU run. Preserve both strict-quota failures; no formal panel yet. |

The architecture conclusion from the [project paper review](architecture-gap-paper-2026-09-28.md)
is narrow: 0.6B changes from Kai's bidirectional typed encoder to an official
Qwen causal model, while most larger arms retain the earlier decoder
candidate-head pattern with different legal initialization/data. The 27B tier
adds a backbone and size, not a demonstrated superior topology. No current
same-panel result establishes a Decision 2.0 Pareto frontier. The product
card should show matched ranks and task matrix, not this internal gate ledger
or a Pareto chart.

The separate [9B Score-cardinality readout](qwen35-9b-score-cardinality-residual-result-2026-09-28.md)
is an actual architectural change: a zero-initialized, level-count-dependent
Score bias over the shared native candidate head. Its fixed SELECT improvement
did not meet the sealed development promotion rule, and all three human pilot
tasks lost macro-F1. It therefore remains a negative transfer finding, not a
2.0 release architecture or a formal result.

The [0.6B token-level TRAIN audit](small06-source-exposure-diagnosis-2026-09-28.md)
found only **102 three-level Score rows / 72,973 native tokens (1.78% of
TRAIN tokens)**; one programmatic composition source accounts for **76.3%** of
tokens. This supports a prospective, source-reviewed Choice/Score data
contrast, not a causal explanation for the observed 0.6B regression. No
replacement source currently passes its answer-blind quality gate, so the
new 0.6B GPU arm remains HOLD.

A distinct [27B pooled-quota schedule candidate](qwen38-27b-pooled-schedule-candidate-2026-09-28.md)
was materialized on the authorized remote CPU from pinned TRAIN and tokenizer
bytes: 2,560 rows/160 updates, 507 Score rows and minimum source-quota L1
shift of two rows. **Six hundred** source groups are partially selected across
task types. No teacher-mask, complete-input overlap or rights admission has
passed; this creates no new trained weight or model score and does not reverse
the previous strict-quota failures. GPU-hours for the schedule screen: zero.

An [ANLI open-development Score diagnostic](anli-score-dev-feasibility-2026-09-28.md)
now passes a bounded input-only exact/near screen against the eight core
roles: 3,200 public development pairs, 2,842 independent normalized premise
groups and zero lexical row-pair hits in every role. It tests short
evidence relation, not long context or unseen release quality. Original
corpus reuse and semantic independence remain unproven; it contributes no
new model result and no JevArena release item.

The separate [ANLI TRAIN source screen](anli-score-train-source-audit-2026-09-28.md)
froze 12,000 rows across 930 whole premise groups before inspecting source
labels. One group (26 rows) exactly overlaps public ANLI DEV input, leaving
an unfilled technical upper bound of 11,974 rows/929 groups. Original-source
terms and repeated-pair quality remain unresolved. Its independent answer-blind
native Score rubric review passed only **9/24** sampled requests and **2/8**
complete groups; shortcuts and uncertain ordinal meanings prevent whole-source
admission. **Training admission is zero; no GPU arm ran.**

The [RACE Choice source screen](race06-choice-source-screen-2026-09-28.md)
measured official TRAIN passages and native 0.6B input lengths without GPU
inference. A separately preregistered, answer-blind 24-pair review passed
**6/24**, below the fixed **18/24** admission gate. The whole source stays
HOLD; it gives neither a new trained model nor long-context evidence.

Authenticated HF CLI readback on 2026-09-28 reconfirmed the private Decision
2.0 collection contains only `DEV2.0-0.6B`. That private repository has 30
standardized files at revision `7ac568e6ce99cdeb7cf423b4984a4f87dba8a204`;
its root `config.json`, README, manifest and referenced chart assets match the
recorded product-package hashes. The private training dataset remains at
revision `b81b78e5477ed712418efda4db26a631ebd2bd2d` with the existing
7,455/700/700 partition manifest. The ineligible 4B research repository is
still private and absent from the collection. This was a read-only inventory
check; it did not repeat model inference or change any quality conclusion.

The [ConTRoL CPU screen](qwen35-4b-context-nli-source-screen-2026-09-28.md),
[0.6B gradient preflight](small06-gradient-conflict-preflight-result-2026-09-28.md),
[0.6B one-update parity](small06-gradient-projection-parity-preflight-2026-09-28.md),
[0.8B Score-only CPU hold](eos08-human-evidence-score-cpu-hold-2026-09-28.md),
[VitaminC Score source screen](score-vitaminc-source-screen-2026-09-28.md),
[TREC passage Score diagnostic HOLD](trec23-passage-score-diagnostic-audit-2026-09-28.md),
[F1000RD human recommendation source HOLD](f1000rd-score-source-screen-2026-09-28.md),
[27B teacher contrast screen](qwen38-27b-train-teacher-residual-screen-2026-09-28.md),
[blind review](qwen35-4b-control-native-blind-review-2026-09-28.md),
[second independent review](qwen35-4b-control-second-independent-review-2026-09-28.md),
[2B clinical-evidence source screen](sol2b-evidence-inference-next-arm-screen-2026-09-28.md),
[Score human-source screen](score-human-ordinal-sources-2026-09-28.md),
[HelpSteer2 independent blind review](helpsteer2-independent-blind-review-result-2026-09-28.md),
[LogiQA 2.0 Choice source proposal](small06-logiqa2-choice-source-proposal-2026-09-28.md),
[LogiQA 2.0 source HOLD](small06-logiqa2-choice-source-audit-2026-09-28.md),
[QuALITY long Choice source screen](quality-long-choice-source-screen-2026-09-28.md),
[QuALITY long Choice admission HOLD](quality-long-choice-admission-2026-09-28.md),
[0.6B projection treatment HOLD](small06-gradient-projection-treatment-result-2026-09-28.md),
[27B whole-group capacity envelope](qwen38-27b-train-group-envelope-2026-09-28.md),
[eight-role input-only projection](goldfree-core-inventory-result-2026-09-28.md),
[ANLI open Score diagnostic](anli-score-dev-feasibility-2026-09-28.md),
[ANLI TRAIN source HOLD](anli-score-train-source-audit-2026-09-28.md),
[RACE Choice source HOLD](race06-choice-source-screen-2026-09-28.md),
[Lux process-isolation result](lux9b-release-example-process-isolation-result-2026-09-28.md), and
[Lux batch-shape result](lux9b-release-example-batch-shape-result-2026-09-28.md)
all remain negative or conditional. None authorizes a new model upload. Continue
recording each arm's source, prompt, adapter, data hashes, GPU-hours and exact
stop reason in the unified research gist.

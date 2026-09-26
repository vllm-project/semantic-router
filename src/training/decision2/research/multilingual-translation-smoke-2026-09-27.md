# Decision 2.0 multilingual TRAIN smoke and source audit

This is a **development-only data quality audit**, dated 2026-09-27. No translated row was used to train or select a model. Sealed FINAL prompts and labels were not accessed. Raw TRAIN, translations, and the manual-review packet remain in private storage; this report contains counts and cryptographic receipts only.

## Why this audit was needed

The clean-v2 TRAIN has 7,455 rows: 6,085 declared English and 1,370 declared Chinese. The Chinese rows are internal structured/programmatic examples; all 2,800 GoEmotions human rows are English. There are no declared Spanish, French, German, Japanese, or Arabic TRAIN rows. Existing public DEV/CSS/JevBench/DecisionBench panels are almost entirely English. The separate 100-source-ID XNLI/PAWS-X parallel development slice established non-ceiling multilingual diagnostic questions but supplied **validation data only**, never TRAIN. Its results and caveats are in `research/multilingual-2026-09-27.md`.

## Reproducible private smoke

`multilingual/translation_smoke.py` pins the clean-v2 TRAIN, SELECT, CAL, rights manifest, MIT [M2M100 418M](https://huggingface.co/facebook/m2m100_418M) translator revision, tokenizer and weight hashes. It deterministically selected 80 English TRAIN source groups from GoEmotions, FLUTE, internal original, and targeted programmatic Choice/Noul/Score buckets. An approximate state-context overlap screen quarantined **six entire groups** before translation: six near matches to clean SELECT/CAL, zero exact matches. It left 74 source groups and 118 parent rows. The same screen found zero exact or approximate near matches from the 707 emitted translations against SELECT/CAL and visible public DEV, CSS pilot, JevBench public231, Decision Bench v4 text1041, authored JevArena, and both multilingual development panels. Its candidate method is 8-band 64-bit SimHash followed by `SequenceMatcher >= 0.94`; it is approximate and cannot certify absence of semantic overlap.

The script translated six variants of each parent row to Chinese, Spanish, French, German, Japanese, and Arabic, with the same `group_id` across translations and paired task rows. It also back-translated the state as a **screening signal**. Option order, keys, label identity, numeric strings, length ratios, and output structure were checked. One Japanese generation had an empty segment and was explicitly omitted; there were 708 attempts and 707 emitted rows. A private 30-row stratified packet awaits bilingual semantic review. No gold-free sealed FINAL source-ID denylist was available, so FINAL source-ID exclusion has **not** been verified; sealed FINAL was not opened.

| Artifact | SHA-256 |
| --- | --- |
| Builder source | `2c26fcd556100950ad0b4ea4f30148c56cdfb9577ce5f4898b430019c99db87e` |
| Immutable smoke manifest | `bace2478d40b59942dbee39da8b90c6eab2adaabe0090935b2835a359e1eba68` |
| QA rows, 708 attempts | `0c8521e4e65564197ee697a2c336ce827caf9b94e687200d48248d62d63dc363` |
| Unapproved translated rows, 707 | `71bfbebaad1c6316baee4eddfcab2541021564d86cf01ee444ef772132b3790a` |
| Private source-overlap quarantine, six groups | `ce987a90167e46ca2e4dc47b3c547b0539a61862f79e979415a9fd0634e344bf` |
| Private manual-review packet, 30 rows | `36c8e36d21aee31b4e717e6f16871fb8d9b2d31614aca972a9c6c77cd6e4c122` |
| Posthoc `langid==1.1.6` count-only audit | `5041c264065f08ebba9f83ca7a970a90f62228382f7de3e730a3b93c13c180ed` |

The manifest binds clean-v2 TRAIN `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`, SELECT `32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6`, CAL `3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a`, and rights manifest `61aa883052759830c4ecf897b36c1062ad816c935a12db824c80abd1f80e9ee8`. The translator is `facebook/m2m100_418M@55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636`, weights SHA `d907ea45e4e4b9db163382a6674f6218b3c59566fe06d77f4055c208b4e87ed1`, tokenizer SHA `d8f7c76ed2a5e0822be39f0a4f95a55eb19c78f4593ce609e2edbc2aea4d380a`, greedy FP16 generation. The manifest states `training_approved=false` and `publication_eligible=false`.

## Automatic quality signals

“Pass” means **no automatic flag**, not human-confirmed semantic agreement. Numeric changes can include formatting changes, and round-trip character similarity can miss semantic errors or flag valid paraphrases. Flags can co-occur, so their counts do not sum to failed rows.

| Language | Attempts | No flag | Changed state numbers | Low round-trip similarity | Other notable flags |
| --- | ---: | ---: | ---: | ---: | --- |
| Arabic | 118 | 78 | 19 | 31 | 18 changed option numbers |
| German | 118 | 114 | 0 | 4 | None |
| Spanish | 118 | 101 | 13 | 4 | None |
| French | 118 | 93 | 19 | 6 | None |
| Japanese | 118 | 57 | 46 | 16 | 18 instruction-number changes, 18 option-number changes, seven extreme length ratios, one empty translation |
| Chinese | 118 | 55 | 48 | 10 | 18 option-number changes, 37 extreme length ratios |
| **All** | **708** | **498** | **145** | **71** | 54 option-number changes, 19 instruction-number changes, 44 extreme length ratios, one empty translation |

| Task type | Attempts | No flag | Changed state numbers | Changed option numbers | Low round-trip similarity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Choice | 300 | 255 | 7 | 0 | 26 |
| Noul | 300 | 220 | 55 | 0 | 27 |
| **Score** | **108** | **23** | **83** | **54** | **18** |

The source split sharpens the diagnosis: GoEmotions had 228/240 no-flag attempts, FLUTE 56/90, internal original programmatic 85/90, and targeted programmatic 129/288. Targeted programmatic states accounted for 137/145 state-number flags. Score no-flag counts by language were ar 0/18, de 18/18, es 5/18, fr 0/18, ja 0/18, zh 0/18. These are **automatic screening results**, not measured label retention.

The posthoc language census found the intended non-Latin script in all emitted Arabic, Japanese, and Chinese rows. Full visible-text `langid` top labels matched targets on ar 114/118, de 117/118, es 117/118, fr 118/118, ja 105/117, zh 76/118; 59/707 rows had fewer than 60 visible letters and were not classified confidently. Structured English instructions and short strings can distort language ID. This confirms script presence, not decision-label correctness.

**Decision:** no MT smoke row enters training. Numeric/ordinal meaning is especially fragile for Score, and semantic agreement is unmeasured for all types. The 30-row packet is an audit aid, not completed human review. Any future translation continuation requires stratified bilingual review of state, instruction, every option, gold label and score rubric; group-level quarantine; a complete gold-free FINAL source-ID exclusion check when such a denylist becomes available; and independent DEV/CSS/public/parallel diagnostics before a release claim.

## Better multilingual TRAIN source candidates

| Candidate | Source and rights | TRAIN-only use and isolation feasibility | Decision |
| --- | --- | --- | --- |
| [MASSIVE 1.1](https://github.com/alexa/massive) | The publisher's [NOTICE](https://github.com/alexa/massive/blob/main/NOTICE.md) states CC BY 4.0 for MASSIVE and its SLURP text ancestry; the [publisher dataset card](https://huggingface.co/datasets/AmazonScience/massive) describes human localization/judgments, 52 locales, 60 intents, and 11,514 TRAIN / 2,033 DEV / 2,974 TEST rows per locale. Target en/zh/es/fr/de/ja/ar locales are covered. Retain Amazon/SLURP attribution and versioned source hashes. | Most promising source for native Choice intent and a carefully balanced one-vs-rest Noul arm. Use only `partition=train`; keep all locales sharing original SLURP `id` as one group; filter by human `intent_score`, `grammar_score`, and target-language judgment after auditing their distribution. Pin official archive/revision and quarantine cross-partition IDs, normalized utterances, near matches, and all visible Decision panels. Keep dev/test out of TRAIN. | **Source/rights feasible; corpus not yet downloaded or deduplicated.** It tests localized short intent decisions, not hard transfer or our numerical Score rubric. |
| [PAWS-X official corpus](https://github.com/google-research-datasets/paws/blob/master/pawsx/README.md) | The [publisher license](https://github.com/google-research-datasets/paws/blob/master/LICENSE) permits any-purpose use with Google attribution appreciated. The publisher states its **296,406 multilingual TRAIN pairs are machine translated**; the 23,659 human-translated pairs are evaluation data. | TRAIN-only pairs could be an optional lower-confidence Noul auxiliary after pair-ID, source `sentence1`, sentence-level exact/near, and cross-language group quarantine. The official README warns that dev and test can share `sentence1`; our PAWS-X validation source IDs and translations are protected. | **Reject as the primary human/native TRAIN answer.** Do not promote our validation rows or describe machine TRAIN as human translated. |
| Multilingual product-review star ratings | Potentially useful native-language ordinal Score, but text redistribution and model-training rights require direct publisher terms and a pinned source audit; a repository/code license cannot establish corpus rights. | Would need review/product group dedup, language and rating balance, and exact/near comparison with all holdouts. Native star ratings are also a different scale and semantics from the Jev Score rubric. | **Deferred pending rights and task-fit audit; not ingested.** |

This is a source feasibility ranking, not an experiment. MASSIVE is the only ready-to-audit human-localized multilingual TRAIN source identified here; it cannot by itself cure Score weakness or prove broad transfer. The next gated action is source acquisition, attribution and group-level overlap audit, followed by a small typed conversion review. No new multilingual training was launched in this task.

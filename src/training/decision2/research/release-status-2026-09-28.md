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
| 0.6B | Official Qwen3-0.6B-Base full-backbone/shared-head package is private. Same-panel v3 **38.520** vs own Kai1 **35.938** and Bosun **38.524**; paired 95% gain interval vs Kai1 **[-2.032,+7.846]**. Public231 **143** vs Kai1 **114**, Bosun **133**. Choice **109/800** vs Kai1 **277/800** and Score **80/400** vs **98/400** regress; Noul **458/800** vs **404/800** improves. AutoJev teacher KL failed SELECT **510/700** vs hard-label **562/700**. A read-only TRAIN diagnostic found gradient conflict in **6/8** fixed windows; projection-disabled, same-schedule one-update parity then passed at **0** category and probability difference on SELECT32. Neither is a model gain. | Private first package exists; stronger SOTA/large-gain claim HOLD. | Implement and run exactly one frozen gradient-projection treatment after its projected one-step/reload safety gate; retain the archived control and stop at SELECT if thresholds fail. Seek source-disjoint Choice/Score supervision independently. |
| 0.8B | Own Eos1 continuation: development proxy changed only **+0.163** and Score fell to **85/400**. A Score-only OCNLI replacement failed CPU admission: longest possible 192 rows carried **31,789** native tokens against a frozen **155,681** minimum. A separate VitaminC real-revision Score source had 820 sampled rows across 256 pages but median native length **180**; its protected-role overlap is unresolved. | HOLD; no qualified formal 2.0 result. | Find a genuinely long, source-disjoint, quality-checked Score corpus, or separately pre-register an equal-token short-source control. Neither screened short source is admitted. |
| 2B | Own Sol1 continuation: same-panel v3 **43.960** vs Sol1 **45.580** and Decider2B **49.499**; public231 **162** vs **161** and **175**. Official Qwen3.5 Base/Posttrained arms failed development promotion. A proposed Sol soft replay stopped at its frozen BF16 zero-step probability-parity gate before any optimizer update. Evidence Inference 2.0 TRAIN provides a possible human-labeled three-class source, but its middle class is not a proven ordinal effect magnitude and no new row is admitted. | HOLD. | Resolve the proposed source rubric, article rights and source-disjoint overlap before a matched-token official-Posttrained arm; retain the parity failure and formal result as recorded. |
| 4B | Official Qwen3.5-4B-Base: same-panel v3 **53.218** vs own Nox1 **56.470** and Decider4B **61.882**; public231 **171** vs **173** and **192**. Own Nox continuation's SELECT gain did not survive typed DEV. Independent ConTRoL blind reviews agreed on **27/30** relations; each matched source labels on **21–22/30**, corroborating three likely label errors and leaving one further conflict for adjudication. | HOLD. A third-party-initialized private research package is ineligible for the formal lineage. | Resolve the disputed source labels, embedded-passage rights and unscanned long-input overlap before any source admission. A matched arm must separately retain Score. No GPU training yet. |
| 9B | Official Qwen3.5-9B Posttrained development proxy **61.255** vs Lux1 historical DEV **70.326**; Score **174/400**. Base arm stopped at step106 runtime crash; exact saved-state replay reproduced steps65–106 and completed step107, ruling out a deterministic checkpoint-state crash. The Lux1 package parity gate still fails on Score probability drift **0.021284** against frozen **0.02**. Process isolation changed nothing; one exact-image Score-only request shifted the prior three-question output by only **0.000953**, still **0.020655** from the published example. | HOLD; neither candidate has a qualified formal 2.0 score. | Obtain the original release per-op/kernel selection trace or byte-identical original build environment to locate Lux numerical drift; retain the 0.02 gate. Improve the official 9B candidate on development before formal testing. |
| ~27B | Official Qwen3.8-27B selected development proxy **68.53** vs native AutoJev27 peer **79.15**; Score **162/400**. Official Gemma4-26B-A4B arm did not pass SELECT. An independent Score-source CPU gate admitted **0/160** sampled TRAIN/SELECT items. A new matched-budget, source-aware teacher contrast is a pre-optimizer proposal only; TRAIN Score teacher agrees with gold on **324/516** rows. | HOLD; no qualified formal 27B score. | Implement and pass its source, overlap, zero-step and gradient gates before any bounded 160-step comparison. Stop on SELECT safety failure; no formal panel yet. |

The architecture conclusion from the [project paper review](architecture-gap-paper-2026-09-28.md)
is narrow: 0.6B changes from Kai's bidirectional typed encoder to an official
Qwen causal model, while most larger arms retain the earlier decoder
candidate-head pattern with different legal initialization/data. The 27B tier
adds a backbone and size, not a demonstrated superior topology. No current
same-panel result establishes a Decision 2.0 Pareto frontier. The product
card should show matched ranks and task matrix, not this internal gate ledger
or a Pareto chart.

The [ConTRoL CPU screen](qwen35-4b-context-nli-source-screen-2026-09-28.md),
[0.6B gradient preflight](small06-gradient-conflict-preflight-result-2026-09-28.md),
[0.6B one-update parity](small06-gradient-projection-parity-preflight-2026-09-28.md),
[0.8B Score-only CPU hold](eos08-human-evidence-score-cpu-hold-2026-09-28.md),
[VitaminC Score source screen](score-vitaminc-source-screen-2026-09-28.md),
[27B teacher contrast screen](qwen38-27b-train-teacher-residual-screen-2026-09-28.md),
[blind review](qwen35-4b-control-native-blind-review-2026-09-28.md),
[second independent review](qwen35-4b-control-second-independent-review-2026-09-28.md),
[2B clinical-evidence source screen](sol2b-evidence-inference-next-arm-screen-2026-09-28.md),
[Score human-source screen](score-human-ordinal-sources-2026-09-28.md),
[Lux process-isolation result](lux9b-release-example-process-isolation-result-2026-09-28.md), and
[Lux batch-shape result](lux9b-release-example-batch-shape-result-2026-09-28.md)
all remain negative or conditional. None authorizes a new model upload. Continue
recording each arm's source, prompt, adapter, data hashes, GPU-hours and exact
stop reason in the unified research gist.

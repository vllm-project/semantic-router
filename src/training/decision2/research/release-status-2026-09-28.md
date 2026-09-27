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
| 0.6B | Official Qwen3-0.6B-Base full-backbone/shared-head package is private. Same-panel v3 **38.520** vs own Kai1 **35.938** and Bosun **38.524**; paired 95% gain interval vs Kai1 **[-2.032,+7.846]**. Public231 **143** vs Kai1 **114**, Bosun **133**. Choice **109/800** vs Kai1 **277/800** and Score **80/400** vs **98/400** regress; Noul **458/800** vs **404/800** improves. | Private first package exists; stronger SOTA/large-gain claim HOLD. | The same-source AutoJev teacher-KL arm failed SELECT **510/700**, macro **.59454** vs hard-label control **562/700**, **.77259**; do not advance it. Audit source-specific loss/gradient conflicts and seek source-disjoint Choice/Score supervision. Do not repeat failed type-separated, candidate-interaction, Posttrained-source, or Score-RPS runs. |
| 0.8B | Own Eos1 continuation: development proxy changed only **+0.163** and Score fell to **85/400**. | HOLD; no qualified formal 2.0 result. | Audit same-source replay/Score retention and test one source-disjoint data contrast with a fixed control. |
| 2B | Own Sol1 continuation: same-panel v3 **43.960** vs Sol1 **45.580** and Decider2B **49.499**; public231 **162** vs **161** and **175**. Official Qwen3.5 Base/Posttrained arms failed development promotion. A proposed Sol soft replay stopped at its frozen BF16 zero-step probability-parity gate before any optimizer update. | HOLD. | Admit a genuinely new labeled source and matched-budget retention contrast; retain the parity failure and formal result as recorded. |
| 4B | Official Qwen3.5-4B-Base: same-panel v3 **53.218** vs own Nox1 **56.470** and Decider4B **61.882**; public231 **171** vs **173** and **192**. Own Nox continuation's SELECT gain did not survive typed DEV. | HOLD. A third-party-initialized private research package is ineligible for the formal lineage. | Independently review the 6,618 retained ConTRoL native Choice/Noul source pairs for rubric correctness and long-text overlap, then design a matched Score-retaining arm. No GPU training until source admission. |
| 9B | Official Qwen3.5-9B Posttrained development proxy **61.255** vs Lux1 historical DEV **70.326**; Score **174/400**. Base arm stopped at step106 runtime crash; exact saved-state replay reproduced steps65–106 and completed step107, ruling out a deterministic checkpoint-state crash. The Lux1 package parity gate still fails on Score probability drift **0.021284** against frozen **0.02**, including isolated process rerun. | HOLD; neither candidate has a qualified formal 2.0 score. | Diagnose BF16/runtime drift and admit a candidate through the unchanged native/package gates before any formal panel. |
| ~27B | Official Qwen3.8-27B selected development proxy **68.53** vs native AutoJev27 peer **79.15**; Score **162/400**. Official Gemma4-26B-A4B arm did not pass SELECT. An independent Score-source CPU gate admitted **0/160** sampled TRAIN/SELECT items. | HOLD; no qualified formal 27B score. | Replace the unsupported ordinal source with a source-disjoint, oracle-checked native Score curriculum before another large optimizer. |

The architecture conclusion from the [project paper review](architecture-gap-paper-2026-09-28.md)
is narrow: 0.6B changes from Kai's bidirectional typed encoder to an official
Qwen causal model, while most larger arms retain the earlier decoder
candidate-head pattern with different legal initialization/data. The 27B tier
adds a backbone and size, not a demonstrated superior topology. No current
same-panel result establishes a Decision 2.0 Pareto frontier. The product
card should show matched ranks and task matrix, not this internal gate ledger
or a Pareto chart.

The [ConTRoL CPU screen](qwen35-4b-context-nli-source-screen-2026-09-28.md),
[Score human-source screen](score-human-ordinal-sources-2026-09-28.md), and
[Lux process-isolation result](lux9b-release-example-process-isolation-result-2026-09-28.md)
all remain negative or conditional. None authorizes a new model upload. Continue
recording each arm's source, prompt, adapter, data hashes, GPU-hours and exact
stop reason in the unified research gist.

# Decision 2.0 first-release status — 2026-09-27 UTC

This is a research status snapshot, not a Hugging Face model card. A first
release requires an eligible starting weight, an exact downloadable native
package, JevArena v3 and public JevBench same-panel measurements, and a
material overall gain over the own 1.0 size where one exists. Decision Index
history selects strong peers only; their scores are never mixed into those
panels. v3 has 1,600 typed items / 2,000 answer slots plus 6,547 CSS items;
the score is `100 × sqrt(T × H)`. Public JevBench has 231 questions and is a
separate independent public-only rerun. Already-open project formal labels
make new v3 comparisons post-key, so their limitations must be stated.

| Size | Direct official/own starting point and current status | Verified result | Main obstacle | Next discriminative action |
| --- | --- | --- | --- | --- |
| 0.6B | Official Qwen3-0.6B-Base, **first model published** as `DEV2.0-0.6B` | v3 **38.5200** vs own Kai1 **35.9383** and same-panel Bosun **38.5243**; public231 **143** vs **114/133**. Full Hub download/native parity passed. | Overall point gain over Kai1 has paired interval crossing zero; typed Choice/Score regressions. | New independent Score/Choice recovery data and unchanged first-release package as control; do not revise the published model's evidence in place. |
| 0.8B | Own Eos1 hard/soft replay and official Qwen3.5 pilots, **HOLD** | Same-budget hard/soft development difference only **+0.1627** proxy, below +1.0 gate; both Score **85/400**, all predicted level 0. No formal v3/public result for these arms. | Score collapse and poor transfer; further open-panel checkpoint choice would overfit. | Admit an independently reviewed, source-disjoint three-level Score curriculum, then one matched-source/budget development screen before any protected panel. |
| 2B | Own Sol1 continuation and official Qwen3.5 Base/Posttrained arms, **HOLD** | Own BEST160 formal v3 **43.9596** vs own Sol1 **45.5804** and Decider2B **49.4992**; public231 **162/161/175**. Official Base/Posttrained full arms failed their DEV promotion gate. | SELECT/DEV gains did not transfer to 15 real tasks; official Posttrained Score **95/400** on DEV. | Audit the transfer/error clusters, add a source-disjoint real-label/ordinal arm under matched budgets and frozen SELECT rule; require independent confirmation before formal promotion. |
| 4B | Official Qwen3.5-4B-Base BEST466, **HOLD**; older private third-party-origin 4B is research only | Formal v3 **53.2179** vs own Nox1 **56.4702**, Kev4B **59.2263**, native Decider4B **61.8816**. Public231 **171** vs **173/175/192**. | Large DEV-to-FINAL reversal: Noul −9 pp, exception −16 pp, Score −4.25 pp vs Nox; CSS H and Score calibration also regress. | Do not prioritize a Choice-only reweight: Choice already slightly exceeds own1. Pre-register a mechanism-targeted Noul/exception/Score and real-task transfer intervention with an external or fresh independent check. |
| 9B | Own Lux1 continuation and official Qwen3.5-9B, **HOLD** | Short-rule own arm SELECT **0.5633** below **0.5767** gate; official Qwen ROCm backward SIGSEGV at 6,144 tokens on two nodes and with/without gradient checkpointing. No eligible full candidate/formal score. | Faulting autograd operator unresolved; repeated full optimizer retries would waste GPU-hours. | A bounded, no-optimizer operator isolation at the first failing length, then exact native zero-step/reload before any resumed training. |
| ~27B | Official Qwen3.8-27B BEST368 **HOLD**; official Gemma4-26B-A4B-it feasibility **PASS only** | Qwen development proxy **68.53** vs same-size AutoJev **79.15**, Score **162/400** vs **400/400**; no formal v3. Gemma official source zero-step/LoRA identity and one TRAIN-only optimizer update/reload passed, with no quality score. | Large Score/peer gap; earlier synthetic Score sets failed realism/overlap; Gemma still lacks a multi-type full-arm result. | Audit complete TRAIN under both tokenizers and design a pinned Gemma multi-type matched-budget development arm; first release must reach near-peer first tier without a 1.0 baseline. |

All public model-card graphics remain rank and model-by-task matrix views.
Parameter Pareto plots are excluded from this first-release product scope.
Detailed hashes, GPU-hours, rejected arms and source evidence are maintained
in the unified research gist and size-specific signed experiment notes.

# Decision 2.0 size gate audit: development evidence only

**No size currently qualifies for publication.** The numbers below are
existing typed DEV1600, human CSS pilot1430 or exposed public231 diagnostics
under their recorded native adapters. They are neither JevArena FINAL nor an
official JevBench rank. Source/checkpoint hashes, panel versions and paired
intervals are retained in the [unified research ledger](https://gist.github.com/Xunzhuo/cd90fce0fa548616d8a4f1b2d2398dea).

| Requested size | Strongest relevant completed diagnostic | Gate and next discriminating experiment |
| --- | --- | --- |
| 0.6B | Kai1 425/1600 typed DEV; Kai continuation 587 but CSS pilot 410/1430 versus Kai1 418. GLiNER English source 652/1600, 64-step continuation 584; Qwen3 reranker continuation 431. | No candidate improves reasoning and independent transfer together. Pin/audit two additional open Qwen3-0.6B decision heads and run the same native panels before selecting a new initialization. Their internal card scores cannot substitute. |
| 0.8B | Eos1 49.50% typed DEV; Eos continuation 49.6875% and CSS pilot 33.78% versus 29.30%, but Brier/ECE worsen. Matched Qwen3.5 Base/posttrained ablation favors posttrained on typed DEV (509 vs 392/1600) and Base on CSS/public (481 vs 439/1430; 129 vs 118/231). | A few extra correct typed items are insufficient when calibration worsens. Test calibration/transfer with same checkpoint and frozen selector; no architecture chosen yet. |
| 2B | Sol1 58.9375% typed DEV and 38.46% CSS pilot. Targeted-to-rights-clean continuation gives 933/1600 and 629/1430; public231 163 versus Sol1 161. | The human pilot improves while typed reasoning regresses. An architecture or data replay arm must recover synthetic reasoning under a source-disjoint selector; CSS pilot alone is not unseen-task evidence when its task families occur in TRAIN. |
| 4B | Eikos clean-v2 1442/1600 typed DEV; CSS pilot 787/1430 is flat against clean-v1 788. The earlier public231 pilot trails open source Eikos at 84.42% versus 85.71%. | Improve task-level transfer and public robustness, then verify deterministic package parity under a pinned runtime. Small 1–3 item differences are below previously observed non-deterministic repeat variance. |
| 9B | Lux1 86.75% typed DEV and 53.78% CSS pilot. Low-rate continuation gives 86.94% and 56.78%; high-rate arm gives 86.75% and 57.48%, but the task pattern differs. | Select on a source-disjoint diagnostic with explicit task/calibration floors. The small typed gain and pilot tradeoff do not establish broad superiority or a size frontier. |
| 27B | Qwen3.8-27B posttrained-source clean-v2 BEST368: typed DEV1213/1600, CSS pilot842/1430, public231 199; Score only161/400 with 0-level bias. | Main target is three-level Score evidence joining, not temperature-only repair. Score TRAIN v1–v5 are blocked; prospective v6 and separate three-level SELECT require independent quality gates before a matched continuation/control experiment. |

Across all sizes, the dominant release blocker is the missing high-quality
1,200–1,500-original sealed authored panel and same-panel final evaluation.
The existing private TRAIN/SELECT/CAL dataset and collection are preserved;
the collection has no published model item. No development score is a release
claim. We should publish each `dev-2.0-xxb` repository only after its own
package parity, rights, rank/Pareto, full human transfer, calibrated native
inference and predeclared non-regression gates pass.

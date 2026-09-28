# Milestone 1 same-panel baseline matrix — post-key same-panel

Eval & peers track, 2026-09-28. Every number is **post-key same-panel**: JevArena
v3 typed FINAL (1,600 items / 2,000 slots) plus CSS15 human transfer (6,547 items),
`v3 = 100 * sqrt(T * H)`; JevBench public 231 is a separate public-subset rerun
(not the official sealed rank). Missing, invalid and native over-budget answers
count as failures. Frozen panels, scorers, runtime and rules are in
[`m1-baseline-runs-prereg-2026-09-28.md`](m1-baseline-runs-prereg-2026-09-28.md)
(amendments A1, A2); per-model reports are in [`m1-reports/`](m1-reports/).
Decision Index numbers were used only to choose peers.

## Matrix

| Model @ pinned revision | Tier | Loaded params | v3 | T | H | Choice / Noul / Score (of 800/800/400) | Public 231 (easy/standard/hard) | Invalid CSS / public | Provenance |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| GLiNER2.5-Decide `7ee5da4c` | 0.6B | 486,444,053 | 42.524 | .4100 | .4410 | 313 / 502 / 88 | 116 (48/46/22) | 1,072 / 56 | run P1 |
| Bosun v3.1 0.6B `1d8b6f96` | 0.6B | 606,131,200 | 38.524 | .4338 | .3422 | 365 / 541 / 83 | 133 (47/50/36) | 0 / 0 | reused |
| DEV2.0-0.6B (checkpoint `5380e01e`) | 0.6B | 597,103,104 | 38.520 | .3091 | .4801 | 109 / 458 / 80 | 143 (48/51/44) | 15 / 0 | reused |
| Decision 1.0 Kai (main `9d6872cd`) | 0.6B | 571,909,635 | 35.938 | .3619 | .3569 | 277 / 404 / 98 | 114 (45/39/30) | 404 / 44 | reused; rerun bit-identical |
| Decision 1.0 Lex (main `6c5e3d48`) | 0.6B | 571,909,635 | 31.022 | .3619 | .2659 | 235 / 402 / 107 | 113 (42/42/29) | 404 / 44 | run R3 |
| Decision 1.0 Eos (main `363c4a5e`) | 0.8B | 753,446,208 | 42.547 | .3925 | .4612 | 315 / 410 / 120 | 142 (48/55/39) | 4 / 0 | run R4 |
| JPT-0.8B `1431c050` | 0.8B | 852,985,920 | 40.085 | .4297 | .3739 | 399 / 462 / 101 | 171 (47/60/64) | 0 / 0 | run P2 |
| Decider 2B `533964da` | 2B | 1,881,825,088 | 49.499 | .5831 | .4202 | 545 / 497 / 133 | 175 (48/64/63) | 0 / 0 | reused |
| Decision 1.0 Sol (main `ce0c018a`) | 2B | 1,900,750,144 | 45.580 | .4219 | .4925 | 366 / 441 / 154 | 161 (48/66/47) | 4 / 0 | reused |
| Bosun v3.1 1.7B `1d8dc82a` | 2B | 1,737,985,024 | 42.117 | .4669 | .3799 | 415 / 493 / 99 | 151 (46/62/43) | 0 / 0 | run P3 |
| Decider 4B `eb5fbdfc` | 4B | 4,205,751,296 | 61.882 | .6894 | .5555 | 750 / 623 / 114 | 192 (48/71/73) | 0 / 0 | reused |
| Decision 1.0 Nox (main `cde2a68d`) | 4B | 4,208,383,488 | 56.470 | .6144 | .5190 | 552 / 653 / 178 | 173 (48/66/59) | 4 / 0 | reused |
| JPT-4B `78312f85` | 4B | 4,539,265,536 | 54.562 | .6256 | .4758 | 659 / 577 / 160 | 203 (48/68/87) | 0 / 0 | run P4 |
| Decision 1.0 Lux (main `cdf4d3ef`), node A frozen runtime | 9B | 7,940,895,744 | 65.808 | .7762 | .5579 | 711 / 704 / 227 | 183 (48/67/68) | 4 / 0 | runs R1 = D1 = D2 |
| JPT-9B `7114b0c3` | 9B | 9,409,813,744 | 60.994 | .7488 | .4969 | 715 / 656 / 227 | 197 (48/68/81) | 0 / 0 | reused |
| AutoJev 27B `6f5b557e` | 27B | 26,086,635,760 | 72.310 | .8869 | .5896 | 800 / 737 / 282 | 200 (48/71/81) | 13 / 0 | reused |

Decision 1.0 rows use the current `main` weights with the native runtime of the
last runtime-bearing revision (weights byte-identical; see the bootstrap record).
The node B Lux1 r4 run (66.268) is kept as a cross-node reference only (see
[`lux1-frozen-runtime-result-2026-09-28.md`](lux1-frozen-runtime-result-2026-09-28.md)).
JPT and GLiNER parameter counts include tensors their native loaders instantiate
(JPT's vision tower is loaded by the HF VLM backend but unused for text decisions).

## Paired v3 intervals versus the tier's own 1.0 model

Joint bootstrap (`compare_v3`, 5,000 draws, seed 20260927; typed groups within
family, CSS tasks then items; invalid answers stay in every draw):

| Pair | Δ v3 | 95% interval |
| --- | ---: | --- |
| GLiNER2.5-Decide − Kai1 | +6.586 | [+0.148, +8.749] |
| Bosun v3.1 0.6B − Kai1 | +2.586 | [+0.374, +9.089] |
| DEV2.0-0.6B − Kai1 | +2.582 | [−2.032, +7.846] |
| Lex − Kai1 | −4.916 | [−10.480, +3.932] |
| JPT-0.8B − Eos1 | −2.462 | [−5.434, +4.735] |
| Decider 2B − Sol1 | +3.919 | [+1.304, +10.914] |
| Bosun v3.1 1.7B − Sol1 | −3.463 | [−5.076, +2.105] |
| Decider 4B − Nox1 | +5.411 | [+0.505, +10.538] |
| JPT-4B − Nox1 | −1.908 | [−6.687, +3.546] |
| JPT-9B − Lux1 (node A) | −4.814 | [−7.822, +1.502] |

## Slices and side views (details in `m1-reports/`)

- **Per-task transfer:** macro-F1 for all 15 CSS tasks per model (`panels.css15.tasks`).
  Tropes and TalkLife remain the weakest tasks for every model.
- **Long input:** 509 CSS and 37 public items are at least 4,000 characters. Kai and
  Lex answer almost none (1,024-token limit: 402 of 509 invalid); GLiNER2.5 answers
  none (512-token encoder window); the 4B–27B models reach 0.46–0.58 accuracy on the
  long CSS slice.
- **Multilingual:** a frozen language screen finds 8,375 of 8,378 prompts English, so
  these panels cannot measure multilingual ability; a separate panel is required.
- **Calibration:** typed Brier / ECE-10, CSS median-task Brier and ECE (pmax, 15 bins),
  public Brier and ECE per model.
- **Robustness:** typed order-, label- and counterfactual-pair consistency (order
  consistency ranges 0.55 for Bosun 0.6B to 0.93 for AutoJev).
- **Latency / throughput:** per-prompt native latency (model load excluded) with the
  batch policy recorded per adapter, on one MI325X per run.

## GPU-hours

Node A GPU6–7 only: R1 0.128, R2 0.080, R3 0.077, R4 0.085, D1 0.113, D2 0.109,
P1 0.046, P2 0.082, P3 0.085, P4 0.113 → **0.919 GPU-hour**. Two launcher attempts
stopped before Docker (script mode bit, then a `pipefail` idle-check bug) and used
no GPU time; both were fixed in commits before the counted runs. Reused rows cost
nothing new.

# 9B Milestone 3 result: v2 data, AutoJev targets and E8F-style full fine-tuning

Protocol: [preregistration](lux9b-m3-prereg-2026-09-28.md) with amendments
[1](lux9b-m3-prereg-amendment-1-2026-09-28.md) (arm B inputs, same-renderer control),
[2](lux9b-m3-prereg-amendment-2-2026-09-28.md) (arm D) and [3](lux9b-m3-prereg-amendment-3-2026-09-28.md)
(arm DL). Development readouts (typed DEV 1,600 + CSS pilot 1,430, CAL698 temperatures) are never
release scores; JevArena v3 numbers are **post-key same-panel** comparisons on node A at 16,384
tokens.

**Disposition: RESULT_PENDING**

## Arms

| Arm | Start / update | TRAIN | Teacher term | Seeds |
| --- | --- | --- | --- | --- |
| A | Lux 1.0, LoRA r16 | mx-v2-full-M (pk1 A0s) + A7 natural24k replay: 53,398 rows / 24.7M tokens (`55abf2ad…`) | own Lux KL 0.5 on recipe rows | 20260926, 1 |
| B | Lux 1.0, LoRA r16 | same as A (byte-identical) | AutoJev-27B KL 0.5 on recipe rows (`0d1a71f1…`) | 20260926, 1 |
| D | Lux 1.0, full fine-tuning (E8F settings) | full-M minus 752 shortcut-family A0s rows + A7 v3 core + A7g 45M + v1 A1–A6 + A7k/A7s/A7r + A7q 5M: 205,790 rows / 116.1M tokens (`34c18d7a…`) | own Lux KL 0.5 on recipe rows | 20260926, 1, 2 + soup |
| DL | Lux 1.0, LoRA r16 (arm A recipe) | same as D | same as D | 20260926, 1 |

## Development readout (paired against the same-runtime Lux 1.0 readout)

| Checkpoint | T | H (median) | H mean | Proxy P | Choice / Noul / Score | CSS discourse / implicit hate / SemEval stance |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Lux 1.0 | .8763 | .5724 | .5283 | 70.82 | 799 / 272 / 331 | .572 / .423 / .590 |
| A seed 20260926 (BEST 3,131) | .8731 | .5843 | .5508 | 71.42 | 799 / 279 / 319 | .584 / .402 / .666 |
| A seed 1 (BEST 2,734) | .8681 | .5807 | .5563 | 71.00 | 800 / 273 / 316 | .581 / .410 / .678 |
| B seed 20260926 (BEST 3,131) | .8694 | .5743 | .5619 | 70.66 | 798 / 261 / 332 | .574 / .406 / .706 |
| B seed 1 (BEST 2,734) | .8688 | .5688 | .5489 | 70.30 | 796 / 275 / 319 | .569 / .398 / .680 |
| D_DL_ROWS | | | | | | |

- **A:** two-seed mean P 71.21 (tie); mean-H intervals above zero in both seeds; typed-DEV Score
  −13.5 items (−3.4 points, fewer middle-level answers) breaches the 3.0-point floor → not a
  finalist.
- **B:** two-seed mean P 70.48 (tie), all floors held → finalist. Matched contrast B − A: P −0.76
  [−2.18, +0.67] and −0.70 [−2.36, +0.91] (seed-paired).

## Formal post-key runs (node A, 16,384 tokens)

| Post-key same-panel | B-s1 | FORMAL_D_COL | Lux1 same-renderer | Lux1 native (gate) |
| --- | ---: | ---: | ---: | ---: |
| JevArena v3 | 64.600 | | 65.231 | **65.808** |
| T / H (H w/o FLUTE) | .7588 / .5500 (.5477) | | .7656 / .5558 (.5540) | .7762 / .5579 (.5568) |
| Choice / Noul / Score | 657 / 697 / 260 | | 710 / 688 / 227 | 711 / 704 / 227 |
| Constraint / evidence / exception / ledger | .642 / 1.000 / .743 / .650 | | .775 / 1.000 / .720 / .568 | .777 / 1.000 / .760 / .568 |
| Public 231 (E/S/H) | 181 (48/66/67) | | 183 (48/67/68) | 183 (48/67/68) |
| Typed Brier / ECE-10 | .154 / .101 | | .125 / .025 | .119 / .017 |
| mlx-diag type macro (en / non-en) | .817 (.872 / .808) | | .832 (.885 / .823) | .828 |
| Paired v3 vs native Lux1 [95%] | −1.208 [−3.280, +0.941] | | −0.577 [−1.057, +0.093] | — |

Details for B: [formal result](lux9b-m3-formal-result-B-2026-09-28.md).

## Findings

- The shared 2.0 renderer alone costs Lux 1.0 0.58 v3 at 16K (65.231 vs 65.808), exactly the
  Milestone 2 8K control: the earlier "16K vs 8K" gap was the renderer, not the input limit.
  Every candidate read through the shared renderer starts 0.58 behind the gating comparator.
- LoRA continuation on v2-M with soft replay moves the development proxy by less than one point
  in either direction (A +0.39, B −0.34); the development readout (typed DEV has no constraint
  competition family) did not foresee B's post-key constraint-competition loss (−.133).
- FINDINGS_D

## Resources and failures

GPU_HOURS_AND_FAILURES

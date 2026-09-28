# 9B Milestone 3 result: v2 data, AutoJev targets and E8F-style full fine-tuning

Protocol: [preregistration](lux9b-m3-prereg-2026-09-28.md) with amendments
[1](lux9b-m3-prereg-amendment-1-2026-09-28.md) (arm B inputs, same-renderer control),
[2](lux9b-m3-prereg-amendment-2-2026-09-28.md) (arm D), [3](lux9b-m3-prereg-amendment-3-2026-09-28.md)
(arm DL) and [4](lux9b-m3-prereg-amendment-4-2026-09-28.md) (arm DW). Development readouts
(typed DEV 1,600 + CSS pilot 1,430, CAL698 temperatures) are never release scores; JevArena v3
numbers are **post-key same-panel** comparisons on node A at 16,384 tokens.

**Disposition: no 9B candidate meets the release gate; 9B stays on Lux 1.0.** The best result,
arm DW (½·arm D seed soup + ½·Lux 1.0), reaches post-key v3 **68.571** vs the native Lux1 16K
**65.808** (+2.763 [−2.085, +6.202]) — the highest 9B v3 measured — with a significant typed
gain (T +.076 [+.054, +.098]) but flat, high-variance human transfer, so the lower bound stays
below zero. DW is kept in private staging (`llm-semantic-router/dev2-9b-staging@8a98dc82`,
`m3/DW/`) for the coordinator and the next milestone.

## Arms

| Arm | Start / update | TRAIN | Teacher term | Seeds |
| --- | --- | --- | --- | --- |
| A | Lux 1.0, LoRA r16 | mx-v2-full-M (pk1 A0s) + A7 natural24k replay: 53,398 rows / 24.7M tokens (`55abf2ad…`) | own Lux KL 0.5 on recipe rows | 20260926, 1 |
| B | Lux 1.0, LoRA r16 | same as A (byte-identical) | AutoJev-27B KL 0.5 on recipe rows (`0d1a71f1…`) | 20260926, 1 |
| D | Lux 1.0, full fine-tuning (E8F settings) | full-M minus 752 shortcut-family A0s rows + A7 v3 core + A7g 45M + v1 A1–A6 + A7k/A7s/A7r + A7q 5M: 205,790 rows / 116.1M tokens (`34c18d7a…`) | own Lux KL 0.5 on recipe rows | 20260926, 1, 2 + soup |
| DL | Lux 1.0, LoRA r16 (arm A recipe) | same as D | same as D | 20260926, 1 |
| DW | ½·D soup + ½·Lux 1.0 (weight interpolation, α fixed in amendment 4) | — | — | one artifact |

## Development readout (paired against the same-runtime Lux 1.0 readout)

| Checkpoint | T | H (median) | H mean | Proxy P | Choice / Noul / Score | CSS discourse / implicit hate / SemEval stance |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Lux 1.0 | .8763 | .5724 | .5283 | 70.82 | 799 / 272 / 331 | .572 / .423 / .590 |
| A seed 20260926 (BEST 3,131) | .8731 | .5843 | .5508 | 71.42 | 799 / 279 / 319 | .584 / .402 / .666 |
| A seed 1 (BEST 2,734) | .8681 | .5807 | .5563 | 71.00 | 800 / 273 / 316 | .581 / .410 / .678 |
| B seed 20260926 (BEST 3,131) | .8694 | .5743 | .5619 | 70.66 | 798 / 261 / 332 | .574 / .406 / .706 |
| B seed 1 (BEST 2,734) | .8688 | .5688 | .5489 | 70.30 | 796 / 275 / 319 | .569 / .398 / .680 |
| D seed 20260926 (BEST 2,683) | .8713 | .5105 | .5421 | 66.69 | 788 / 268 / 338 | .510 / .439 / .677 |
| D seed 1 (BEST 2,677) | .7644 | .5531 | .5560 | 65.02 | 673 / 247 / 303 | .553 / .422 / .693 |
| D seed 2 (BEST 1,672) | .7819 | .4455 | .4988 | 59.02 | 610 / 296 / 345 | .445 / .387 / .664 |
| D soup (`9a1d7db8…`) | .8719 | .5516 | .5560 | 69.35 | 759 / 272 / 364 | .552 / .422 / .694 |
| **DW** (`de0dcebd…`) | **.9581** | **.6038** | **.5817** | **76.06** | 800 / 367 / 366 | .604 / .426 / .716 |
| DL seed 20260926 (BEST 11,646) | .7956 | .5716 | .5753 | 67.44 | 627 / 305 / 341 | .572 / .463 / .691 |
| DL seed 1 (BEST 10,201) | .8981 | .5773 | .5618 | 72.00 | 800 / 288 / 349 | .577 / .413 / .695 |

- **A:** two-seed mean P 71.21 (tie); mean-H intervals above zero in both seeds; typed-DEV Score
  −13.5 items (−3.4 points, fewer middle-level answers) breaches the 3.0-point floor → not a
  finalist.
- **B:** two-seed mean P 70.48 (tie), all floors held → finalist. Matched contrast B − A: P −0.76
  [−2.18, +0.67] and −0.70 [−2.36, +0.91] (seed-paired).
- **D:** single full-fine-tuning seeds are unstable (typed-DEV transition table 394 / 280 / 210 of
  400); the soup (≥ the seed mean, so D's artifact) ties Lux (ΔP −1.47 [−3.96, +1.91]) with typed
  Score +33 but breaches the Choice (−5.0 points) and CSS-H (−0.021) floors → not a finalist.
- **DW:** ΔP **+5.24 [+3.14, +8.37]** vs Lux 1.0 (+6.71 vs the D soup); rule precedence 367 vs 272
  (neither endpoint solved it); every floor held → finalist.
- **DL:** two-seed mean P 69.72 (tie); one seed loses the transition table (271) and attribute gate
  (356), so mean Choice falls 85.5 items (−10.7 points) → not a finalist. DL − A (the added A7 +
  v1 data under LoRA): seed-paired P −4.0 and +1.0, T −.078 and +.030.

## Formal post-key runs (node A, 16,384 tokens)

| Post-key same-panel | B-s1 | **DW** | Lux1 same-renderer | Lux1 native (gate) |
| --- | ---: | ---: | ---: | ---: |
| JevArena v3 | 64.600 | **68.571** | 65.231 | **65.808** |
| T / H (H w/o FLUTE) | .7588 / .5500 (.5477) | .8519 / .5520 (.5442) | .7656 / .5558 (.5540) | .7762 / .5579 (.5568) |
| Choice / Noul / Score | 657 / 697 / 260 | 745 / 753 / 265 | 710 / 688 / 227 | 711 / 704 / 227 |
| Constraint / evidence / exception / ledger | .642 / 1.000 / .743 / .650 | .863 / 1.000 / .882 / .662 | .775 / 1.000 / .720 / .568 | .777 / 1.000 / .760 / .568 |
| Public 231 (E/S/H) | 181 (48/66/67) | 179 (48/67/64) | 183 (48/67/68) | 183 (48/67/68) |
| Typed Brier / ECE-10 | .154 / .101 | .099 / .057 | .125 / .025 | .119 / .017 |
| mlx-diag type macro (en / non-en) | .817 (.872 / .808) | .813 (.865 / .804) | .832 (.885 / .823) | .828 |
| Paired v3 vs native Lux1 [95%] | −1.208 [−3.280, +0.941] | +2.763 [−2.085, +6.202] | −0.577 [−1.057, +0.093] | — |

Details: [B formal result](lux9b-m3-formal-result-B-2026-09-28.md), [DW formal result](lux9b-m3-formal-result-DW-2026-09-29.md)
(per-task CSS15, axis intervals, staging). DW vs the same-renderer control: +3.341 [−1.577, +6.733].

## Findings

- The shared 2.0 renderer alone costs Lux 1.0 0.58 v3 at 16K (65.231 vs 65.808), exactly the
  Milestone 2 8K control: the earlier "16K vs 8K" gap was the renderer, not the input limit.
  Every candidate read through the shared renderer starts 0.58 behind the gating comparator.
- LoRA continuation on v2-M with soft replay moves the development proxy by less than one point
  in either direction (A +0.39, B −0.34); the development readout (typed DEV has no constraint
  competition family) did not foresee B's post-key constraint-competition loss (−.133).
- **A7's Stage curricula are the typed-reasoning lever for Lux**, as the coordinator's 21:00 note
  predicted: DW lifts every held-out typed FINAL family (constraint competition +.086, exception
  stack +.122, resource ledger +.094 vs native Lux1) with better typed Brier (.099 vs .119).
- **Full fine-tuning of a strong 9B start is seed-unstable** on this mixture (and so is LoRA on the
  same data: DL seed 20260926): typed-DEV transition table collapses in 3 of 5 runs. Uniform seed
  soups repair much of it, and **interpolating the soup half-way back to Lux 1.0 beats both
  endpoints** on every development axis (WiSE-FT behavior); the own-Lux KL on recipe rows alone
  did not hold the full fine-tune near Lux.
- **Human transfer is the binding axis.** No arm raised post-key H: DW is −.006 with task-level
  swings (reddit humor −.167, persuasion +.070), and the three-task CSS pilot cannot see those
  tasks, so development selection cannot target them. The v3 interval is dominated by the H
  axis (median over 15 resampled tasks): DW's T gain alone is significant.
- AutoJev-27B targets (B) did not transfer AutoJev's typed strength through v2-M prompts (its
  constraint-competition skill was lost, not gained).

## Resources and failures

**GPU-hours (Milestone 3): 30.56** on node A (A 2.45, B 2.44, D 14.11 incl. soup CAL/readouts,
DL 10.39, DW 0.06, formal runs 0.62 = B 0.26 + DW 0.18 + same-renderer Lux1 control 0.18, Lux 1.0
readouts 0.04, preflights 0.45); CPU-only data builds and soup builds excluded. Cap 38.

- Preflights: all four arm preflights passed (A, B, D, DL; DW needs none). No nonfinite loss, no
  OOM (full fine-tuning peaked at 151 GiB), no ROCm fault in 6 LoRA and 3 full-fine-tuning runs.
- Process failures (no GPU time lost): a first data build could not resolve the new two-level HF
  blob links (the job now mounts the whole HF cache); the first wave launch exited silently in
  the GPU wrapper's render lookup under `pipefail` (fixed in `bb9b01d41`); a host-side import
  check wrote bytecode into one mirror (removed; `--verify` passes again; host scoring is now
  bytecode-free); one commit was refused by the repository's shellcheck hook and fixed.
- Throughput: LoRA ≈ 6,400 tokens/s (1.2–1.7 s per 16-row update, 46 GiB); full fine-tuning
  ≈ 6,800–7,100 tokens/s (≈ 6 s per ≥ 64-row update).

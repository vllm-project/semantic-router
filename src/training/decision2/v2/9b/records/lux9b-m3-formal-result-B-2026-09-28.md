# 9B Milestone 3 arm B formal post-key result: release gate NOT met

Run per the [lock](lux9b-m3-formal-lock-B-2026-09-28.md). **Post-key same-panel** comparison (the
v3 labels were accessed earlier in the project); public 231 is a public-subset reproduction.

- Collection: node A GPU2, eval runner, pinned image `f83b1d10…` + FLA, frozen Milestone 3
  autotune cache copy, 16,384 tokens, over-length inputs invalid; mirror `e9b1231be`. Seal
  `ad67af55…`. Smoke, formal panels and `mlx-diag` completed without faults.
- Candidate: B-s1 = Lux 1.0 + LoRA r16 trained on mx-v2-full-M + A7 natural24k replay with KL 0.5
  to **AutoJev-27B** targets (provenance caveat: AutoJev's public pipeline includes SFT rows
  generated with a closed model), `checkpoint-0003131`, CAL698 temperatures.

| Post-key same-panel, 16K | B-s1 | Lux1 same-renderer | Lux1 native (gate) |
| --- | ---: | ---: | ---: |
| JevArena v3 | **64.600** | 65.231 | 65.808 |
| T / H | .7588 / .5500 | .7656 / .5558 | .7762 / .5579 |
| H without FLUTE | .5477 | .5540 | .5568 |
| Choice / Noul / Score | 657 / 697 / 260 | 710 / 688 / 227 | 711 / 704 / 227 |
| Constraint / evidence / exception / ledger | .642 / 1.000 / .743 / .650 | .775 / 1.000 / .720 / .568 | .777 / 1.000 / .760 / .568 |
| Public 231 (E/S/H) | 181 (48/66/67) | 183 (48/67/68) | 183 (48/67/68) |
| Typed Brier / ECE-10 | .154 / .101 | .125 / .025 | .119 / .017 |
| CSS15 median Brier / ECE-pmax | .587 / .152 | .566 / .139 | .565 / .136 |
| Invalid typed / CSS / public | 0 / 4 / 0 | 0 / 4 / 0 | 0 / 4 / 0 |
| mlx-diag type macro (en / non-en) | .817 (.872 / .808) | .832 (.885 / .823) | .828 (eval) |

**Paired v3 (eval runner):** B-s1 − Lux1 native **−1.208 [−3.280, +0.941]**; B-s1 − same-renderer
Lux1 −0.631 [−2.784, +1.389]. The lower bound is below zero: **arm B does not meet the 9B
release gate**; no staging upload.

Per-task CSS15 macro-F1, B-s1 minus the same-renderer control: tempowic +.106, flute +.075,
emotion +.051 (GoEmotions, a same-source task present in A0s), indian_english +.017, raop +.013,
media_ideology +.007, tropes +.006, persuasion −.002, wiki_politeness −.007, ibc −.018, talklife
−.026, conv_go_awry −.041, wiki_corpus −.046, reddit_humor −.046, mrf −.090.

Reading: AutoJev distillation on v2-M moves typed Score (+33, resource ledger +.082) and exception
stack (+.023) in AutoJev's direction but costs constraint competition (−.133, the family AutoJev
itself solves), several CSS tasks and typed calibration; the development readout (proxy tie, no
floor breach) did not flag the constraint-competition loss, which typed DEV does not contain.
Development readouts of the LoRA arms A and B are recorded in the Milestone 3 result.

GPU-hours: B-s1 formal 0.264 (smoke 164 s, panels 621 s, mlx-diag 167 s); same-renderer Lux1 16K
control 0.178 (640 s).

# 9B Milestone 3 arm DW formal post-key result: best 9B v3 so far, release gate NOT met

Run per the [lock](lux9b-m3-formal-lock-DW-2026-09-29.md). **Post-key same-panel** comparison (the
v3 labels were accessed earlier in the project); public 231 is a public-subset reproduction.

- Collection: node A GPU4, eval runner, pinned image `f83b1d10…` + FLA, frozen Milestone 3
  autotune cache copy, 16,384 tokens, over-length inputs invalid; mirror `e78a79a9c`. Seal
  `c40ff4f2…`. Smoke, formal panels and `mlx-diag` completed without faults.
- Candidate DW (`model_sha256` `de0dcebd…`) = ½·(arm D seed soup) + ½·Lux 1.0; arm D = Lux 1.0
  full fine-tuning on 116.1M tokens (v2-M with own-Lux KL, A7, v1, new A7 human Score arms); own
  Lux targets only. CAL698 temperatures.

| Post-key same-panel, 16K | DW | Lux1 same-renderer | Lux1 native (gate) |
| --- | ---: | ---: | ---: |
| JevArena v3 | **68.571** | 65.231 | 65.808 |
| T / H | **.8519** / .5520 | .7656 / .5558 | .7762 / .5579 |
| H without FLUTE | .5442 | .5540 | .5568 |
| Choice / Noul / Score | 745 / 753 / 265 | 710 / 688 / 227 | 711 / 704 / 227 |
| Constraint / evidence / exception / ledger | .863 / 1.000 / .882 / .662 | .775 / 1.000 / .720 / .568 | .777 / 1.000 / .760 / .568 |
| Public 231 (E/S/H) | 179 (48/67/64) | 183 (48/67/68) | 183 (48/67/68) |
| Typed Brier / ECE-10 | .099 / .057 | .125 / .025 | .119 / .017 |
| CSS15 median Brier / ECE-pmax | .569 / .094 | .566 / .139 | .565 / .136 |
| Invalid typed / CSS / public | 0 / 4 / 0 | 0 / 4 / 0 | 0 / 4 / 0 |
| mlx-diag type macro (en / non-en) | .813 (.865 / .804) | .832 (.885 / .823) | .828 (eval) |

**Paired v3 (eval runner, 5,000 replicates; typed groups within family, CSS tasks then items):**
DW − Lux1 native **+2.763 [−2.085, +6.202]**; DW − same-renderer Lux1 +3.341 [−1.577, +6.733].
Axis intervals vs native: **T +.0756 [+.0538, +.0975]**, H −.0059 [−.0803, +.0488]. The lower bound
of v3 is below zero, so **DW does not meet the 9B release gate**. It is the highest post-key v3
of any 9B model measured on node A (Lux1 65.808, Nimble v2 62.056, JPT-9B 60.994).

Per-task CSS15 macro-F1, DW minus the same-renderer control: persuasion +.070, emotion +.066
(GoEmotions, a same-source task in A0s), tempowic +.053, flute +.052 (same-task source removed
from A0s; still +.052 here), raop +.049, tropes −.007, indian_english −.009, wiki_politeness
−.011, talklife −.017, media_ideology −.019, wiki_corpus −.036, conv_go_awry −.040, mrf −.041,
ibc −.062, **reddit_humor −.167**.

Reading: A7's Stage curricula (new for Lux) plus weight interpolation toward Lux deliver a large,
significant typed-reasoning gain on held-out typed families (constraint competition +.086,
exception stack +.122, resource ledger +.094 vs native Lux1; typed Brier better) and better CSS
calibration, while human transfer is flat on the median with high task-level variance (one task,
reddit humor, −.167) and multilingual diagnostic accuracy drops .019. The v3 interval is wide
because H is a median over 15 resampled tasks. Disclosed: typed families share mechanisms (no
text) with A7 Stage4 generators (A7 track record).

**Kept for the next step (internal staging, never public):** private
`llm-semantic-router/dev2-9b-staging` commit `8a98dc8261472999af7557da6117efd56bf742e4`, folder
`m3/DW/` (`checkpoint/` full FP32 Decision 2.0 checkpoint, `cal698/calibration.json`,
`STAGING.json`); privacy verified; 19 files / 31.8 GB, byte counts equal to the node copy.
Scored run directory: node A `/data/dev2/runs/9b/formal-m3/DW-16k` (+ `-mlx`, `-smoke`).

GPU-hours: DW CAL + readouts 0.06 (the soup build ran on CPU); formal 0.18 (smoke 105 s,
panels 425 s, mlx-diag 118 s).

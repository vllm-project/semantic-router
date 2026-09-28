# 9B L2 formal post-key same-panel result: release gate NOT met

Run per the [lock](lux9b-l2-formal-lock-2026-09-28.md). The v3 labels were
accessed earlier in the project: **post-key same-panel** comparison, not a new
blind test. Public 231 is a public-subset reproduction, not the official rank.

- Collection: node A GPU2, `v2/eval/run_same_panel.sh`, pinned image
  `sha256:f83b1d10…` with FLA, `TRITON_CACHE_AUTOTUNING=1` and the persisted
  Milestone 2 cache for both runs. L2 from mirror `d67513f34`; the Lux1-8K
  control and both mlx-diag collections from `e25b2142e` (adds only a smoke
  wrapper for the control's adapter). Both at the 8,192-token limit, over-length
  inputs invalid. Seals: L2 `af9de66f…`, Lux1-8K `f815ccd7…`.
- Two launcher errors before any model load, recorded: the first L2 smoke passed
  an unsupported `--reason` flag (0.2 s), and the first control smoke failed
  because `infer_1p0` has no `--max-items` (7 s). Fresh run directories were used.

| Post-key same-panel | L2 (8K) | Lux1-8K control | Lux1 16K (eval, node A) |
| --- | ---: | ---: | ---: |
| JevArena v3 | **65.361** | 65.231 | 65.808 |
| T / H | .7509 / .5689 | .7656 / .5558 | .7762 / .5579 |
| Choice / Noul / Score | 691 / 679 / 231 | 710 / 688 / 227 | 711 / 704 / 227 |
| Families: constraint / evidence / exception / ledger | .728 / .999 / .700 / .578 | .775 / 1.000 / .720 / .568 | — |
| Public 231 (E/S/H) | 184 (48/66/70) | 183 (48/67/68) | 183 (48/67/68) |
| Typed Brier / ECE-10 | .165 / .129 | .125 / .025 | .119 / .017 |
| CSS15 median Brier / ECE-pmax | .580 / .180 | .566 / .139 | — |
| Invalid typed / CSS / public | 0 / 18 / 0 | 0 / 18 / 0 | 0 / 4 / 0 |
| mlx-diag type macro (en / non-en) | .824 (.872 / .815) | .832 (.885 / .823) | .828 (eval `x-lux1`) |

**Paired v3 (eval runner):** L2 − Lux1-8K **+0.130, 95% CI [−2.357, +1.480]**;
L2 − Lux1-16K −0.447 [−2.838, +1.014]. The lower bound is below zero, so **L2
does not meet the 9B release gate** and no staging upload follows.

Per-task CSS15 macro-F1, L2 minus Lux1-8K: flute +.189, emotion +.065,
media_ideology +.021, raop +.017, reddit_humor +.014, persuasion +.009, ibc,
talklife, conv_go_awry ≈ 0; tropes −.011, wiki_corpus −.022, indian_english
−.029, mrf −.033, tempowic −.048, wiki_politeness −.048. FLUTE is a disclosed
same-task source in A0 TRAIN (removed in A0s); without it the median is ≈ .568
versus .554. The development Score gain did not carry to the formal typed
panel (231 vs 227, resource ledger .578 vs .568), while typed Choice and Noul
fell (−19, −9) and typed calibration worsened (ECE .129 vs .025).

GPU-hours: 0.401 (L2 smoke 139 s, L2 558 s, control smoke 85 s, control 394 s,
mlx-diag 155 s and 107 s, failed launches 7 s). GPU2 lease released to research
& data at 09:49 UTC.

**Disposition:** 9B stays on Lux 1.0. A continuation on current A0 does not
beat Lux1 post-key; the next 9B attempt needs A0s-based data Lux does not
already fit (Score and held-out typed reasoning), with soft replay retained to
protect human transfer, and CAL on the clean CAL revision.

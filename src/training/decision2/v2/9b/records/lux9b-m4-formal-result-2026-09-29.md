# 9B Milestone 4 formal post-key result: K-a13 and KN-a12 meet the 9B gate, U-a13 does not

Run per the [lock](lux9b-m4-formal-lock-2026-09-29.md) under the [preregistration](lux9b-m4-prereg-2026-09-29.md) and amendments
[1](lux9b-m4-prereg-amendment-1-2026-09-29.md)–[4](lux9b-m4-prereg-amendment-4-2026-09-29.md). Every v3 number is a **post-key same-panel** comparison (the v3 labels
were accessed earlier in the project); public 231 is a public-subset reproduction; development readouts are never release scores.

**Verdict: two of the three finalists meet the preregistered gate**, the first 9B formal runs to do so (L2, B-s1 and DW did not). Post-key same-panel v3 against native
Lux1 16K 65.808 (paired joint bootstrap, 5,000 draws, seed 20260927): K-a13 67.737, +1.929 [+0.607, +4.144] and KN-a12 67.982, +2.174 [+0.358, +4.962] pass; U-a13
67.866, +2.058 [−0.215, +4.040] fails on its lower bound. Both passing finalists stay on node A and go to the coordinator; nothing is uploaded (see "Outcome").

- Rule stage (development only; independent re-run byte-identical): α\* K ⅓, U ⅓, D ½ (= DW, formally run in Milestone 3 at 68.571, so its slot passes on), KN ½,
  P ½ (fills no slot); no proxy drop (best 76.06). Finalists in priority order: K-a13, U-a13, KN-a12.
- Candidates (FP32, CAL698 temperatures): K-a13 = ⅓·K soup + ⅔·Lux 1.0 (`b9d973b3…`); U-a13 = ⅙·D soup + ⅙·K soup + ⅔·Lux (`8c1ab371…`); KN-a12 = ½·KN soup +
  ½·Lux (`e0dffa0e…`). K / KN: Lux 1.0 full fine-tuning on 60M XL r2 tokens with 1.0·KL(own Lux) on every row; KN without the A7q / H1 / H8 human arms.
- Collection: node A GPU6 (K-a13, then KN-a12) and GPU7 (U-a13), eval runner, image `f83b1d10…` + FLA 0.5.2, one copy of the frozen `formal-m3` autotune cache,
  16,384 tokens, over-length inputs invalid, mirror `3277dec9d`. Smoke, panels, mlx-diag and the type gate ran without faults (exit 0, ended 04:58–05:09 UTC).

| Post-key same-panel, 16K | K-a13 | U-a13 | KN-a12 | DW (M3) | Lux1 same-renderer | Lux1 native (gate) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| JevArena v3 | **67.737** | 67.866 | **67.982** | 68.571 | 65.231 | 65.808 |
| Paired v3 vs native Lux1 [95%] | **+1.929 [+0.607, +4.144]** | +2.058 [−0.215, +4.040] | **+2.174 [+0.358, +4.962]** | +2.763 [−2.085, +6.202] | −0.577 [−1.057, +0.093] | — |
| Paired v3 vs same-renderer [95%] | +2.507 [+1.037, +4.599] | +2.636 [+0.265, +4.496] | +2.751 [+0.733, +5.511] | +3.341 [−1.577, +6.733] | — | — |
| T / H | .8106 / .5660 | .8106 / .5682 | .8200 / .5636 | .8519 / .5520 | .7656 / .5558 | .7762 / .5579 |
| T axis vs native [95%] | +.0344 [+.015, +.054] | +.0344 [+.014, +.054] | +.0438 [+.023, +.064] | +.0756 [+.054, +.098] | — | — |
| H axis vs native [95%] | +.0081 [−.011, +.042] | +.0103 [−.024, +.041] | +.0057 [−.022, +.049] | −.0059 [−.080, +.049] | — | — |
| H without FLUTE | .5648 | .5681 | .5615 | .5442 | .5540 | .5568 |
| Choice / Noul / Score (of 800 / 800 / 400) | 736 / 715 / 246 | 738 / 719 / 240 | 746 / 721 / 245 | 745 / 753 / 265 | 710 / 688 / 227 | 711 / 704 / 227 |
| Constraint / evidence / exception / ledger | .840 / 1.000 / .787 / .615 | .845 / 1.000 / .797 / .600 | .865 / 1.000 / .802 / .613 | .863 / 1.000 / .882 / .662 | .775 / 1.000 / .720 / .568 | .777 / 1.000 / .760 / .568 |
| Public 231 (E / S / H) | 178 (48 / 66 / 64) | 180 (48 / 67 / 65) | 181 (48 / 67 / 66) | 179 (48 / 67 / 64) | 183 (48 / 67 / 68) | 183 (48 / 67 / 68) |
| Typed Brier / ECE-10 | .113 / .048 | .120 / .057 | .125 / .087 | .099 / .057 | .125 / .025 | .119 / .017 |
| CSS15 median Brier / ECE-pmax | .527 / .079 | .533 / .071 | .541 / .069 | .569 / .094 | .566 / .139 | .565 / .136 |
| mlx-diag type macro (en / non-en; cross-language) | .822 (.862 / .816; .683) | .821 (.862 / .814; .668) | .809 (.856 / .801; .695) | .813 (.865 / .804; .686) | .832 (.885 / .823; .652) | .828 (eval) |
| Gate | **PASS** | FAIL | **PASS** | FAIL | — | — |

## Gate reading

- Lock rule: `ci95.low` > 0 and `axis_ci95.H.delta.high` ≥ 0 in `PAIRED-vs-Lux1-16K.json`, and every `types.json` verdict `OK`. K-a13 +0.607, +.042, OK ×3 → **PASS**;
  KN-a12 +0.358, +.049, OK ×3 → **PASS**; U-a13 −0.215 → **FAIL** (H upper +.041, OK ×3). The largest single-answer share is ≤ .54 in every type (collapse: ≥ .90).
- Why DW (post-key same-panel v3 68.571) was not a release candidate although it is higher: its paired lower bound was below zero, +2.763 [−2.085, +6.202], because
  the H axis (a median over 15 resampled CSS tasks) was wide, −.0059 [−.080, +.049] (reddit humor −.167). The gate reads the lower bound, not the point estimate; the
  M4 finalists' H intervals are 40–55% of DW's width and centred above zero, so a lower v3 clears it.
- Per-task CSS15 macro-F1 minus the same-renderer control: up in 11 / 10 / 9 of 15 tasks (K-a13 / U-a13 / KN-a12; DW 5); largest gain tempowic (+.068 / +.076 / +.074).
  Regressions: K-a13 wiki_corpus −.027, talklife −.022, tropes −.011, mrf −.002; U-a13 reddit_humor −.053, wiki_corpus −.022, conv_go_awry −.017, talklife −.017,
  mrf −.004; KN-a12 wiki_corpus −.045, mrf −.024, conv_go_awry −.018, talklife −.017, tropes −.014, ibc −.005. Reddit humor: +.012 / −.053 / +.024 (DW −.167).

Reading: the own-Lux-KL soups interpolated back toward Lux 1.0 (α ⅓–½) keep a significant typed gain over native Lux1 (Choice +25 to +35, Score +13 to +19;
constraint competition +.06 to +.09, resource ledger +.03 to +.05) with human transfer at Lux1 level and smaller task swings than DW. K-a13 and KN-a12 differ in data
(human arms) and α (⅓ vs ½), so their gap is not the preregistered K − KN contrast. Disclosed: public 231 (public-subset reproduction) 178–181 vs 183, hard tier 64–66
vs 68; mlx-diag (a development diagnostic) .809–.822 vs .832 same-renderer, largest drop Korean (−.070 / −.050 / −.090; KN-a12 also Chinese −.041, Japanese −.033);
KN-a12's Score temperature sits on the calibrator floor (.05), so its typed Score Brier is .373 (K-a13 .279, U-a13 .323); argmax metrics and the gate are unaffected.

## Outcome

The fixed priority K, U, D, KN, P ("hypothesis order, not development rank") only fills finalist slots; the preregistration and its amendments do not say which of
several passing finalists becomes the 9B candidate. Amendment 1: a passing finalist "is reported to the coordinator immediately with its scored run directory. Only a
coordinator-approved release candidate is uploaded, directly to its final release repo (release engineering, after the steward's `v2/common/hf_headroom.sh` check)."
Applied: **K-a13 and KN-a12 are gate-passing finalists reported to the coordinator, not yet release candidates**; the choice between them is the coordinator's. For it:
K-a13 is first in priority, with the higher lower bound, typed calibration and mlx-diag; KN-a12 has the higher post-key v3 and T. U-a13 fails and keeps the same node copy.

## Identity, storage and resources

- Identity: every `output/*.manifest.json` (typed FINAL, CSS15, public 231, mlx-diag) carries the lock's `model_sha256` and calibration `file_sha256`; all nine
  `GPU-TIME.json` show image `f83b1d10f14d…`, exit 0; `triton-cache.copy.json` source tree `af623300d71a…`. Loaded parameters 7,940,895,744; torch 2.12.0, Triton 3.7.1,
  FLA 0.5.2 and HIP 7.2 as for DW and the same-renderer control; the paired compares use the same panel (`201f01fb3e54…`) and comparator predictions as DW's.
- Invalid: typed 0 / 2,000 slots, public 0 / 231, CSS15 4 / 6,547 per finalist (all over the 16,384-token limit, none truncated; as for Lux1), mlx-diag 0 / 2,275.
- Cache: tree `af623300d71a` → `d10b24d792c8` (after the K-a13 / U-a13 smokes) → `5604ffdc5f19` (after their panels; unchanged through KN-a12), 4,785 files throughout.
  The only change: 13 Triton `__grp__*.json` group files now list `formal-m4` absolute paths (FLA gated-delta-rule kernels, recompiled to identical bytes because
  `formal-m3` is not mounted); every `.hsaco`, kernel metadata file and the 56 autotune JSONs are byte-identical to the frozen `formal-m3` cache. M3's same-renderer
  control rewrote the same 13 files on 09-28; B-s1 and DW were pure cache hits. GPU6 and GPU7 shared the copy, so the change is not attributed to one run.
- Storage (no Hugging Face upload, amendment 1): each finalist stays on node A as `m4/NAME-build/soup` + `m4/NAME-cal/` with `m4/NAME-build/SHA256SUMS` (23 files,
  31.78 GB; re-hashed once, 23 / 23 OK); manifest sha256 K-a13 `6913eb61836a…`, U-a13 `d426c1a4f18f…`, KN-a12 `1159daf23803…`.
- GPU-hours: formal 0.5519 (1,987 GPU-s): K-a13 0.1873 (smoke 123.6 s, panels 430.9 s, mlx-diag 119.9 s), U-a13 0.1839 (122.2 / 422.2 / 117.5 s), KN-a12 0.1808
  (115.6 / 419.3 / 115.9 s). Milestone 4 total **19.86** = 19.31 training + readouts (node-B copies in, CPU soups out) + 0.55 formal; cap 22. `gpu_hours.py` lists
  20.15 for `m4` because it also counts the 16 CPU soup / line builds (0.83 h).
- Run directories (node A `/data/dev2/runs/9b/`): scored `formal-m4/K-a13-16k` (seal `11abc1cc7818…`), `formal-m4/KN-a12-16k` (`92084d7ed1e0…`), `formal-m4/U-a13-16k`
  (`3b802a0aa180…`), each with `-smoke`, `-16k-mlx`, `-16k.gates/types.json`, `NAME-cache.jsonl`; scratch readout `formal-m4/readout-w3/`; logs `m4/logs/formal-gpu{6,7}.*`.
- Commits: rule code `3277dec9d` (`lux9b/m4_rules.py`); rule outputs node A `m4/rules/` (alpha `258835e7c7cc…`, seed-K `51c559bce1fd…`, seed-P `0e0d6d53af51…`,
  seed-KN `0e9237c608a4…`, finalists `d6dd34d9ebfe…`; re-run in `m4/rules/verify-w3/`), recorded in the lock `d9c04cbda`; state `9f64f9284`; this record.

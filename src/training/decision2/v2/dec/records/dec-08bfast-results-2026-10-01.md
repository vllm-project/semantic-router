# Decoder 0.8B fast track — results: M12's `08b-RA` fails successor items 1–7 (2026-10-01)

Lock [`dec-08bfast-formal-lock-2026-10-01.md`](dec-08bfast-formal-lock-2026-10-01.md) (`a3ab9eda2`, before any GPU
job); state [`dec-08b-fast-state.md`](dec-08b-fast-state.md); aggregates in
[`dec-08bfast-results-2026-10-01/`](dec-08bfast-results-2026-10-01/). The coordinator's one formal attempt for `08b-RA`
(COORDINATION 2026-10-01 22:35). Post-key same-panel evidence. Nothing was uploaded; C1 was not opened; no Index run.

## Bottom line

- **`08b-RA` is not a successor. Items 1–7 fail on items 1, 4 and 6(b)**, against both bars, so there is no C1
  recheck, item 8, release or Index run. **DEV2.0-0.8B (`4afea305`) stays the 0.8B release.** Per the coordinator's
  exception this was the only attempt; nothing more is run on `08b-RA`.
- The development screen was right. The `attribute_gate` family-floor miss shows up formally as a typed loss: typed
  FINAL T −.024 [−.045, −.004], choice 483 vs 529. mlx-diag card-eligible is also significantly lower
  (−.0136 [−.0256, −.0024]).
- What 08b-RA gains is human transfer, and the gain is large: H .514 vs .440, +.073 [−.031, +.098]. CSS15 has
  15 tasks, though, so the CI is wide and the v3 composite is +2.87 [−2.87, +4.58]. Public 231 is 166 vs 156
  (p .11, not a regression).

## Formal verdict (node E GPU6, image `dbe5f32b`, `--isolate`, copies of node B's frozen 0.8B masters; scored on node A)

Bars: **`bar-t1`** = the stored formal run of DEV2.0-0.8B (`runs/release/dev2-0p8b-t1-derived`, v3 50.236);
**`bar-e`** = `f08-08b-C0`, node E's collection of the same weights on this path. `f08-08b-C0` reproduces node B's
`m8s-ref-08b-I` exactly: 0 changes on typed FINAL, CSS15 and public 231; v3 50.263; mlx-diag Δ 0.0 vs node A's E8F
run. It is **not** answer-identical to `bar-t1` (9 / 20 / 1 category changes; +0.03 [−0.34, +0.47]), the node A vs
node B difference M8s found. So items 1, 2, 4, 6(b) and 7 were evaluated against both bars, as the lock requires.

| Item | Result vs `bar-t1` (stored run, 50.236) | Result vs `bar-e` (node-E C0, 50.263) | Pass |
| --- | --- | --- | --- |
| Package / calibration | revision `366794a6…` (10 files), 753,446,208 parameters; CAL698 16K rejected by the 23:15 rule → T = 1 | | |
| v3 / T / H | **53.102** / .5491 / .5136 (bar .5734 / .4401) | (bar .5741 / .4401) | |
| 1. v3 paired 95% lower bound > 0 | +2.866 **[−2.866, +4.579]** | +2.839 **[−2.920, +4.487]** | **FAIL** |
| 2. H not significantly below | +.073 [−.031, +.098] | +.073 [−.030, +.097] | PASS |
| 3. No type collapsed | choice / Noul / Score OK; typed FINAL C / N / S **483 / 594 / 117** (bar 529 / 611 / 107; C0-E 531 / 610 / 107) | | PASS |
| 4. mlx-diag card-eligible (Choice + Noul) upper bound ≥ 0 | **−.0136 [−.0256, −.0024]** vs node A's E8F run (full −.0019 [−.0129, +.0089]) | the same (C0-E = node A) | **FAIL** |
| 5. Tier gate vs adopted Eos 1.0 (42.547) | +10.56 [+3.14, +14.75]; no collapse | | PASS |
| 6(a). New TRAIN file exposure | `08b-RA` TRAIN `12bd63d8…` (309,225 rows): 0 groups vs the r2 payload `2194716a…` | | PASS |
| 6(b). Rules 1 and 5 without the 84 flagged items | rule 1 [−2.95, +4.68]; rule 5 [+3.14, +14.62]; 6 / 6 stored files reproduced | rule 1 [−2.96, +4.58] | **FAIL** |
| 7. Public 231 not REGRESSION | 166 vs 156, +10 [−1, +21], p .11 (hard 54 vs 45) | 166 vs 155, p .07 | PASS |
| 8. C1 post-key | not reached (items 1–7 fail) | | — |

Report only: vs Intern-0.8B +9.57 [+1.89, +11.41], vs Kev-0.8B +9.89 [+1.99, +12.21], vs Eos 1.0 at 16K +10.74
[+3.16, +14.74]. CSS15 macro-F1 vs `bar-t1`: up on mrf +.117, conv_go_awry +.103, wiki_politeness +.084, ibc +.082,
media_ideology +.079, persuasion +.019, emotion +.016; down on flute −.078, raop −.023, tempowic −.021, talklife
−.019, reddit_humor −.014, tropes −.011, indian_english_dialect −.008, wiki_corpus −.001. mlx-diag non-English by type:
choice .657 vs .687, Noul .538 vs .542, Score .731 vs .719.

## Operations

- Frozen inputs: the soup tree `83926edf…` (equal to M12's), C0 tree `26baab01…`, select file `2f1123a2…`. Chain:
  C0 smoke and collection (14:48–14:53Z), then RA (14:53–14:58Z); mlx-diag C0 on GPU7 and RA on GPU6. Every step
  passed and nothing was rerun. The seals are RA `50b65064…` and C0 `756b9108…`.
- Node E GPU6–7 only, under lease `track=dec-08bfast`, co-tenant entries `owner.dec-formal` /
  `owner.m6-formal-smoke`. **0.21 GPU-h** in total: CAL fits 0.01, smokes 0.03, collections 0.13, mlx-diag 0.04.
  M13's GPUs (node E GPU0–3, node F) and files were not touched. Node A is CPU only: scoring, the exposure receipt,
  overlap_effects.
- Code: `ops/f08/f08-formal.sh`, `f08-score.sh`, `f08_successor.py` (m6_successor per bar, combined), tests
  `v2/dec/tests/test_f08.py`.

## What this suggests (not preregistered)

The 0.8B breadth recipe moves human transfer up, but it costs typed choice and multilingual choice. This is the M12
mechanism (losses move between heads), and here it reaches the formal panels. M13's `RA-SD` (typed-row
self-distillation) and `RA-AG` (`attribute_gate` upweighted) target exactly this head. They should be judged on
typed FINAL choice and mlx-diag card-eligible as well as v3.

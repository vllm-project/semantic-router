# Rescreen overlap vs released scores (eval & peers, 2026-09-29)

**Question.** The research & data M3b rescreen excluded 305 training groups (727 rows) from r2 because they share
topical phrases with evaluation items; there are no exact duplicates. Models trained on earlier data may have seen
them. Does removing those items change any reported score or release conclusion for DEV2.0-0.8B, DEV2.0-0.6B or the
approved 2B candidate (S2T)? Post-key same-panel, stored sealed predictions only.

**Flagged items** (receipts under the rescreen's node-A directory; the id list `flagged.json` `0ed60dd7…` stays on
node A). The set of 305 groups matches `rescreen.private.json` exactly. The groups flag these items:

- **CSS15, 82 items:** `media_ideology` 73, `wiki_corpus` 4, `conv_go_awry` 2, `ibc` 1, `persuasion` 1, `tropes` 1.
  The evidence is rare n-grams for 80 and long-window near matches for 2.
- **public 231:** 1 hard item.
- **mlx-diag:** 1 item (English PAWS-X, Noul).
- **Typed FINAL / DEV:** none, so T is unchanged everywhere.
- **Decision Bench v4 (33) and ml-parallel-dev (3):** in no reported panel, and no card number uses them.

**Own exposure.** Every training row was matched to the excluded groups by group id, row id and input hash; all
three methods agree and every file's hash equals its run manifest. All three training sets were built before the
rescreen.

| Model | Training set | Excluded groups in it | Scored items it could have seen |
| --- | --- | --- | --- |
| DEV2.0-0.6B | `m4-mix-t` `98e4e859…` | none | none |
| DEV2.0-0.8B | `m2-full-a7-v1` `d1dc33fc…` | 33 (V1-A3 29, A7m 4) | 11: `media_ideology` 9, `wiki_corpus` 1, mlx-diag 1 |
| 2B S2T | `m3-v2m-ret` `13804ac6…` | 24 (H3 15, H6 4, E11 3, H1 2) | 19: `media_ideology` 18, `tropes` 1 |

The flagged public-231 item comes from a G6 group that is in none of the three training sets.

**How it ran.** `python3 -m v2.eval.overlap_effects` ran on CPU only, with no GPU and no C1 access. Scoring took 29 s
on node A; the 0.8B and 2B training files were matched on node B's CPU, where they are stored, against a payload of
training-side ids only.

- **Code:** preregistration [`m5-overlap-effects-prereg-2026-09-29.md`](m5-overlap-effects-prereg-2026-09-29.md)
  at `e4976c5d2`; the own-exposure analysis is post hoc,
  [amendment 1](m5-overlap-effects-prereg-amendment-1-2026-09-29.md) at `96641c0db`.
- **Spec:** [`m5-overlap-effects/spec.json`](m5-overlap-effects/spec.json). Each candidate is compared with its own
  1.0 model(s) and its card peers, plus JPT-0.8B (internal only).
- **Runs:** run 3 is final. It adds Bosun 1.7B, the fifth model on the released 2B card; every other model and pair
  is identical to run 2.
- **Validation:**
  - Every full-panel rescore equals that model's `REPORT.json` or stored mlx-diag score exactly.
  - All 17 stored paired files are reproduced bit for bit (gate files and release `PAIRED-vs-*` files).
  - The card-bound runs (0.8B, 2B) give outcomes identical to their gate runs.
- **Artifacts:** private eval-artifacts dataset `m5/overlap-effects/`: runs 1–2 at commit `9d43d47e`, run 3 at
  `879adff5`.

## (a) Scores with and without the flagged items (same items removed for every model)

| Model | v3 | H | `media_ideology` F1 | public 231 (/230) | mlx-diag |
| --- | --- | --- | --- | --- | --- |
| **DEV2.0-0.8B** | 50.236 → 50.267 | .4401 → .4406 | .302 → .298 | 156 → 156 | .6521 → .6519 |
| Eos 1.0 | 42.547 → 42.547 | .4612 → .4612 | .366 → .367 | 142 → 142 | .6648 → .6646 |
| Intern-Decision | 43.535 → 43.546 | .3824 → .3826 | .372 → .362 | 164 → 163 | .5831 → .5830 |
| Kev | 43.217 → 43.217 | .3896 → .3896 | .350 → .339 | 147 → 147 | .6475 → .6473 |
| **DEV2.0-0.6B** | 43.541 → 43.541 | .4796 → .4796 | .260 → .249 | 142 → 142 | .6147 → .6145 |
| Kai 1.0 | 35.938 → 35.969 | .3569 → .3575 | .195 → .190 | 114 → 113 | .4536 → .4534 |
| Lex 1.0 | 31.022 → 31.022 | .2659 → .2659 | .172 → .158 | 113 → 112 | .3692 → .3690 |
| Bosun | 38.524 → 38.476 | .3422 → .3413 | .312 → .309 | 133 → 133 | .6008 → .6006 |
| GLiNER2.5-Decide | 42.524 → 42.524 | .4410 → .4410 | .044 → .046 | 116 → 116 | .5196 → .5198 |
| **2B S2T** | 53.437 → 53.437 | .5255 → .5255 | .351 → .348 | 171 → 171 | .7085 → .7083 |
| Sol 1.0 16K | 45.781 → 45.781 | .4928 → .4928 | .305 → .295 | 160 → 160 | — |
| Sol 1.0 | 45.580 → 45.610 | .4925 → .4931 | .300 → .292 | 161 → 161 | .7105 → .7103 |
| Decider 2B | 49.499 → 49.499 | .4202 → .4202 | .310 → .302 | 175 → 174 | .7416 → .7414 |
| This-That 1.2 | 46.112 → 46.112 | .4050 → .4050 | .256 → .251 | 147 → 146 | .7418 → .7417 |
| Bosun 1.7B | 42.117 → 42.117 | .3799 → .3799 | .283 → .293 | 151 → 151 | .7209 → .7208 |

- **v3 and H.** v3 moves by at most 0.05 for every model (JPT-0.8B 40.085 is unchanged). H is the median of the 15
  task F1 values, and `media_ideology` is no model's median task. H moves only where the median task is itself an
  affected task: `wiki_corpus` for DEV2.0-0.8B, Bosun and Sol 1.0, and `conv_go_awry` or `persuasion` for Intern and
  Kai.
- **Public 231.** All three candidates missed the flagged item. Intern, JPT-0.8B, Kai, Lex, Decider 2B and This-That
  answered it, so only those comparators lose an item.
- **mlx-diag.** Only the English Noul cell changes: every model except GLiNER2.5 answered the item. Every
  non-English language is unchanged, so the 2B's Korean .59 vs Sol 1.0's .63 stands.

## (b) Paired candidate − comparator (5,000 draws; full → without)

| Pair | Δ v3 | Δ H | Δ `media_ideology` F1 |
| --- | --- | --- | --- |
| 0.8B − Eos 1.0 | +7.69 [+3.65, +13.32] → +7.72 [+3.53, +13.16] | −0.021 [−0.085, +0.088] → −0.021 [−0.086, +0.086] | −0.064 [−0.107, −0.021] → −0.069 [−0.117, −0.021] |
| 0.8B − Intern | +6.70 [+0.63, +10.35] → +6.72 [+0.84, +10.37] | +0.058 [−0.043, +0.119] → +0.058 [−0.037, +0.121] | −0.070 [−0.124, −0.016] → −0.064 [−0.124, −0.002] |
| 0.8B − Kev | +7.02 [+1.33, +11.25] → +7.05 [+1.39, +11.22] | +0.050 [−0.041, +0.122] → +0.051 [−0.039, +0.124] | −0.048 [−0.101, +0.006] → −0.041 [−0.099, +0.019] |
| 0.6B − Kai 1.0 | +7.60 [+4.70, +10.76] → +7.57 [+4.75, +10.77] | +0.123 [+0.068, +0.182] → +0.122 [+0.067, +0.185] | +0.064 [+0.017, +0.111] → +0.059 [+0.008, +0.110] |
| 0.6B − Lex 1.0 | +12.52 [+3.47, +18.12] → +12.52 [+4.11, +18.46] | +0.214 [+0.033, +0.305] → +0.214 [+0.054, +0.311] | +0.088 [+0.043, +0.132] → +0.091 [+0.045, +0.139] |
| 0.6B − GLiNER2.5 | +1.02 [−1.77, +7.80] → +1.02 [−1.64, +7.87] | +0.039 [−0.013, +0.167] → +0.039 [−0.010, +0.171] | +0.216 [+0.172, +0.260] → +0.203 [+0.158, +0.249] |
| 0.6B − Bosun | +5.02 [−1.35, +7.80] → +5.07 [−1.32, +7.98] | +0.137 [+0.010, +0.192] → +0.138 [+0.011, +0.195] | −0.052 [−0.107, +0.005] → −0.059 [−0.120, +0.001] |
| 2B − Sol 1.0 16K | +7.66 [+3.26, +10.81] → +7.66 [+3.23, +10.74] | +0.033 [−0.044, +0.092] → +0.033 [−0.045, +0.091] | +0.046 [−0.000, +0.090] → +0.053 [+0.005, +0.100] |
| 2B − Sol 1.0 | +7.86 [+3.36, +10.80] → +7.83 [+3.30, +10.82] | +0.033 [−0.044, +0.090] → +0.032 [−0.046, +0.088] | +0.051 [+0.006, +0.094] → +0.056 [+0.009, +0.102] |
| 2B − Decider 2B | +3.94 [−2.48, +5.67] → +3.94 [−2.79, +5.65] | +0.105 [−0.008, +0.132] → +0.105 [−0.016, +0.132] | +0.041 [+0.000, +0.081] → +0.045 [+0.001, +0.089] |
| 2B − This-That 1.2 | +7.33 [−2.00, +13.60] → +7.33 [−1.90, +13.39] | +0.120 [−0.050, +0.223] → +0.120 [−0.051, +0.221] | +0.095 [+0.054, +0.135] → +0.097 [+0.053, +0.138] |
| 2B − Bosun 1.7B | +11.32 [+3.37, +13.04] → +11.32 [+3.43, +12.95] | +0.146 [+0.003, +0.171] → +0.146 [+0.005, +0.171] | +0.068 [+0.015, +0.122] → +0.055 [−0.002, +0.114] |

- **Resampling noise.** Rerunning the full panels with a second seed moves the interval bounds by up to 0.8 v3
  (Lex's lower bound goes from 3.47 to 4.28) and 0.02 H. The with/without shifts above are the same size or smaller;
  the largest are Lex's v3 lower bound (3.47 → 4.11) and H vs Decider 2B (−0.008 → −0.016).
- **Public 231 and mlx-diag deltas.** These change by at most one item and by 0.0004 respectively; the full tables
  are in the artifact. T is unchanged.

**Preregistered rules: no release conclusion changes.**

- **Own-1.0 gate.** Every lower bound stays above 0.
- **First-release threshold.** The best peers stay the same, and every candidate still clears the threshold:

  | Candidate | v3 | Threshold | Best peer |
  | --- | --- | --- | --- |
  | 0.8B | 50.27 | 39.19 | Intern |
  | 0.6B | 43.54 | 38.27 | GLiNER2.5 |
  | 2B | 53.44 | 44.55 | Decider 2B |

- **Human transfer vs the best peer.** It is still not significantly below that peer at any tier.
- **Three interval-status changes, all at a boundary, none on a number any card states.**
  - For 2B − Sol 1.0 16K, the `media_ideology` interval moves from including 0 to above 0 (lower bound −0.0001 →
    +0.005). The change is in the candidate's favour.
  - For 2B − Bosun 1.7B, the `media_ideology` interval moves from above 0 to including 0 (lower bound +0.015 →
    −0.002). The change is against the candidate.
  - For 0.8B − JPT-0.8B, the public 231 interval's upper bound moves from −1 to 0. JPT is not on the card.
- **Three rank changes, all among comparators; no candidate changes place on any metric.**
  - On `media_ideology` at 0.8B, Intern drops from first to third. The 0.8B is last either way, and on the card's
    chart Eos and Intern swap places in that column.
  - On `media_ideology` at 2B, Sol 1.0 and Bosun 1.7B swap fourth and fifth place. The 2B stays first.
  - On H at 2B, Sol 1.0 and Sol 1.0 16K swap. They are 0.0003 apart and are the same weights at two token limits.
- **Card-stated per-task disclosures.** They keep their sign and size:
  - 2B `wiki_corpus` vs Sol 1.0 16K: −0.045 → −0.046;
  - 0.8B `media_ideology` vs Eos: still significantly below.

## (c) Contamination signature: none

**Preregistered check** (all 82 flagged CSS15 items). For each model, accuracy on the flagged items minus accuracy on
the unflagged items of the same tasks; the table compares the candidate's difference with the comparator's.

| Pair | Candidate | Comparator | Difference [95% CI] |
| --- | --- | --- | --- |
| 0.8B − Eos / Intern / Kev | +0.045 | +0.006 / +0.029 / +0.061 | +0.040 [−0.072, +0.157] / +0.017 [−0.133, +0.170] / −0.015 [−0.156, +0.127] |
| 0.6B − Kai / Lex / GLiNER2.5 / Bosun | +0.065 | +0.006 / +0.074 / −0.043 / −0.018 | +0.059 [−0.065, +0.184] / −0.009 [−0.121, +0.105] / +0.108 [−0.006, +0.226] / +0.083 [−0.075, +0.243] |
| 2B − Sol 16K / Sol / Decider / This-That / Bosun 1.7B | +0.049 | +0.060 / +0.050 / +0.087 / +0.042 / −0.083 | −0.011 [−0.133, +0.108] / −0.001 [−0.120, +0.116] / −0.038 [−0.148, +0.065] / +0.006 [−0.118, +0.125] / +0.132 [−0.013, +0.271] |

- **No interval lies above 0,** for all 82 items or for the 73 `media_ideology` items alone.
- **The flagged items are simply a little easier.** Most models, including the 0.6B, which never trained on them,
  score higher on them. GLiNER2.5's number is a floor: only 5 of its 82 answers are valid.
- **Candidates are not more confident on flagged items.** For each model, compare the mean gold-label probability on
  flagged items with the same-task unflagged items (task-weighted, valid answers only). The candidates' differences
  (+0.010, −0.003, +0.020) sit inside the comparators' range (−0.069 to +0.083, where −0.069 is GLiNER2.5's floor;
  Decider 2B +0.046, This-That +0.083).

**Own exposure (post hoc).** This repeats the check on only the items each candidate's training rows could have
touched.

- **0.8B, 10 exposed CSS items.** It answered 4, against 5 for Eos, 8 for Intern and 8 for Kev. The differences are
  −0.072 [−0.293, +0.060], −0.391 [−0.681, −0.079] and −0.428 [−0.734, −0.137]. The 0.8B is, if anything, weaker on
  the items it could have seen.
- **2B, 19 exposed CSS items.** It answered 9 (.474, against .376 on the same tasks' unflagged items, a gap of
  +0.097).
  - The unexposed comparators' gaps on the same items are similar: Decider +0.152 (10 answered), Sol 16K +0.070 and
    Sol +0.072 (7 each), This-That +0.079 (7). The differences are −0.054, +0.028 [−0.302, +0.343], +0.025 and
    +0.019.
  - Only Bosun 1.7B differs: it answered 2 (gap −0.204), so the difference is +0.301 [+0.086, +0.528].
  - This is the one interval above 0 among the nine post hoc exposure tests, two of which lie below 0. It reflects a
    Bosun 1.7B weakness on these items rather than a 2B gain: the 2B's gap is in the middle of the unexposed
    comparators' range.
- **Worst case.** The candidate misses every exposed item it answered correctly; comparators are unchanged.
  - 0.8B v3 goes from 50.236 to 50.176, and vs Eos it is +7.63 [+3.59, +13.31].
  - 2B v3 is unchanged at 53.437, because neither of its exposed tasks is its median task; vs Sol 16K it is +7.66
    [+3.24, +10.81], and vs Bosun 1.7B +11.32 [+3.26, +13.04].
  - No rule, interval status or candidate rank changes.
  - The 2B's H margin over Bosun 1.7B is borderline in every variant, independent of the overlap: its lower bound is
    +0.003 on the full panel, +0.002 with the second seed and +0.0001 in the worst case.
- **Power.** With only 10 and 19 items the intervals are about ±0.3 wide, so this check can only detect large
  effects. The worst case, however, bounds every reported number regardless of power.

## Verdict and card wording

Nothing material changes, and no card number needs a correction. Keep the full-panel numbers as the reported ones
and add one line per card:

- **DEV2.0-0.8B** (next card-only revision): "A later training-data screen found topical-phrase overlap (no exact
  duplicates) between 33 of this model's training groups and 11 evaluation items (9 media_ideology, 1 wiki_corpus,
  1 multilingual diagnostic). Rescored without those items, or counting them all as errors, post-key v3 changes by
  at most 0.06 and none of this model's comparisons change."
- **DEV2.0-2B** (its `card.text.limitations` already has a draft "Evaluation familiarity" item, which is accurate):
  keep it and append "Those rows touch 19 of the panel's items (18 media_ideology, 1 tropes); rescored without them,
  or counting all 19 as errors, post-key v3 stays 53.44 and none of this card's comparisons change."
- **DEV2.0-0.6B**: no change is needed, because none of its training rows are involved. If every card should carry
  the line: "A later screen of the program's training pools found topical-phrase overlap with 84 items of the
  reported evaluation panels; none of them involve this model's training data, and removing them leaves its
  post-key v3 unchanged."

Output hashes (sha256 prefixes):

| Output | Hash |
| --- | --- |
| Final `overlap-effects.json` (run 3) | `55358069` |
| Run 2 / run 1 | `0b99c9fe` / `1e69a257` |
| Exposure 0.6B / 0.8B / 2B | `63549469` / `5849d5a1` / `7cfa59bd` |
| Training-side payload | `2194716a` |

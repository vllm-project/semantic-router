# `m3/mixtures/xl-r2/` — XL release recipes, revision r2 (private)

Revision r2 of the Milestone 3b XL recipes. **Use r2 for every new run.** r1 (`m3/mixtures/xl/`,
revision `ba848147…`) stays published but is superseded; `rescreen.public.json` gives its flagged counts.

**This is the final r2 (preregistration amendment 4).** The first r2 build of this folder, at revision
`9f83ce9b…` ("r2-strict"), is **superseded**. It removed every group flagged by the rescreen, including the 43,944
groups whose only hits are on held-out slices (v1 AHO, A7 AHO) or on SELECT700 / CAL700. Those slices are not
evaluation panels. Removing them cut 81–85% of the v1 generated pools, about 27% of A7g / A7o and 22% of the
anchor. The final r2 excludes only the groups with a hit on an evaluation or benchmark role. It keeps the
other flagged groups and discloses them below. The file names are unchanged.

## What changed against r1

1. **Rescreen of every r1 row** (amendment 3 §2).
   - The 481,432 rows of the six r1 recipes were scanned with `v2.data.overlap`, as rendered in their pool files.
   - The scan used the four methods and thresholds of every arm's audit. It ran against the 58 quarantining roles of
     PI-v4 (manifest `48fd2537…`): evaluation and development panels, SELECT / CAL, and the v1 and A7 held-out slices.
   - Each pool was scanned on its own, and the whole union once.
   - Flagged by either pass: 44,249 group ids, 74,404 rows, 49.9M native tokens.
2. **Exclusion rule** (amendment 4). A flagged group id is removed from every recipe when any of its hits, in any
   pool, is on a role other than `v1_aho_*`, `a7_aho_*`, `rights_clean_select` (SELECT700) and `rights_clean_cal`
   (CAL700). The roles that excluded something are:
   - CSS15 (gold-free and native);
   - Decision Bench v4;
   - JevBench-231 (and native);
   - `mlx-diag`;
   - `ml-parallel-dev`.

   The rule removed 305 group ids, 727 rows and 1.15M tokens:

   | Pool | Groups | Rows | Tokens |
   | --- | ---: | ---: | ---: |
   | H3 | 179 | 360 | 0.80M |
   | H6 | 54 | 175 | 0.16M |
   | V1:A3 | 29 | 58 | 0.14M |
   | E11 | 23 | 96 | 0.04M |
   | H1 | 12 | 19 | 3.4K |
   | A7m | 4 | 4 | 0.9K |
   | G6 | 2 | 6 | 2.0K |
   | A7q | 1 | 7 | 5.5K |
   | H5 | 1 | 2 | 0.5K |

   The other 43,944 flagged group ids (73,677 rows, 48.8M tokens) stay in the recipes.
3. **Gap arms H7 and H8** (amendment 2 §4).
   - Whole TRAIN groups of `m3/arms/H7` and `m3/arms/H8` (revision `09f73967…`), in
     `sha256("mx-xl-r2:<variant>:<pool>:" + group)` order.
   - A group is skipped if it would put a human source above 8%, or English above 60%, of the r2 total
     (the base plus every candidate group). No group was skipped.
   - The short recipes take only groups whose rows are all ≤ 1,024 native tokens. H7 has none.
   - **H7 / H8 rows have no teacher targets. Train them on gold labels.**

## Recipes

"Long" is the share of tokens in rows of ≥ 2,000 native tokens, over all pools / without generated pools / human
pools only.

| Recipe | Definition | Rows | Native tokens | Choice / Noul / Score % | English % | Long % |
| --- | --- | ---: | ---: | --- | ---: | --- |
| `mx-xl-full-r2` | `mx-xl-full` − excluded + H7 (all 1,713 groups) + H8 (all 17,062 groups) | 365,970 | 179,200,447 | 31.9 / 45.6 / 22.6 | 58.7 | 29.0 / 29.1 / 28.4 |
| `mx-xl-short-r2` | `mx-xl-short` − excluded + the 15,360 short H8 groups | 373,577 | 127,353,971 | 29.6 / 41.2 / 29.1 | 56.7 | 0 |
| `cx-xl-r2-nogap-full` | `mx-xl-full-r2` minus H7 / H8 | 340,306 | 150,701,655 | 37.2 / 38.0 / 24.8 | 58.8 | 19.4 / 11.9 / 10.1 |
| `cx-xl-r2-nogap-short` | `mx-xl-short-r2` minus H8 | 355,632 | 123,708,072 | 29.7 / 41.0 / 29.3 | 58.3 | 0 |
| `cx-xl-r2-a7v1-full` | `cx-xl-a7v1-full` − excluded | 191,536 | 118,209,822 | 58.4 / 22.7 / 18.9 | 56.1 | 32.0 / 20.2 / 16.6 |
| `cx-xl-r2-a7v1-short` | `cx-xl-a7v1-short` − excluded | 167,067 | 58,216,274 | 50.3 / 18.1 / 31.6 | 56.2 | 0 |
| `cx-xl-r2-v2v1-full` | `cx-xl-v2v1-full` − excluded | 295,998 | 118,091,015 | 15.7 / 57.5 / 26.8 | 61.5 | 10.5 / 13.7 / 12.1 |
| `cx-xl-r2-v2v1-short` | `cx-xl-v2v1-short` − excluded | 282,419 | 98,815,193 | 16.6 / 56.5 / 26.9 | 58.8 | 0 |

**Gap-arm control.** `cx-xl-r2-nogap-*` is the matched control for the gap arms. Train it to the r2
recipe's token budget by repetition. The `a7v1` / `v2v1` controls are published at their own totals, as in
r1. They are not English-capped; `cx-xl-r2-v2v1-full` is 61.5% English, and r1's was 61.8%.

## Files

- `<recipe>.ids.jsonl`: one line per row, `{id, pool, source, task_type, language, native}`, sorted by
  `id`. `native` is the Qwen3.5-0.8B-Base native token count.
- `<recipe>.lux1.missing.jsonl` and `<recipe>.autojev27.missing.jsonl`: the ids without a published
  target from that teacher.
- `mx-xl-r2.manifest.json` (schema `decision2-mx-xl/2`):
  - **Rule:** under `rescreen`, the rule, `disclosed_roles`, the hit roles split into excluding and disclosed, and
    the excluded and disclosed counts, by pool and by role.
  - **Per recipe:** the r1 fields (tokens by type, language, Score level count, pool and source; English share;
    teacher coverage), long-evidence shares with and without generated pools, the shift against r1, the excluded
    and the disclosed rows by pool, and the gap-arm additions.
  - Also the selection report (caps, skipped groups) and the hash of every input.
- `rescreen.public.json`, `rescreen/<pool>.overlap.public.json` and `rescreen/union.overlap.public.json`:
  counts only; unchanged from r2-strict.
- `check.json`: 67 checks re-derived from the written files:
  - the exclusion rule re-derived from the private receipt equals the manifest;
  - r2 = r1 − excluded + whole H7 / H8 groups, and the controls;
  - no excluded group in any recipe;
  - unique ids and `input_sha256`;
  - caps, English share and long share.

## Joining

Rows join by `id` to their pool's file. These files are byte-identical to the ones the recipes were
built from:

| `pool` | Rows file |
| --- | --- |
| `A0s-strict` | `m3/pk1/A0s-strict/train.jsonl` |
| `H1`, `H3`, `H5`, `H6`, `E11`, `G2`, `G6`, `G4h` | `m2/arms/<pool>/train.jsonl` |
| `V1:A1`, `V1:A2`, `V1:A3`, `V1:A4v2h`, `V1:A5`, `V1:A6g`, `V1:A6h` | `v2/arms/<name after "V1:">/train.jsonl` |
| `A7g`, `A7h`, `A7i`, `A7k`, `A7m`, `A7o`, `A7p`, `A7q`, `A7r`, `A7s` | `v2/a7/arms/<pool>/train.jsonl` |
| `H7`, `H8` | `m3/arms/<pool>/train.jsonl` |

Teacher targets also join by `id`:

- **Own Lux1:**
  - `m3/pk1/A0s-strict/lux1.targets.jsonl`;
  - `m2/teachers/lux1/rp-v2/wave{1,2,3,4}.targets.jsonl`;
  - `m3/teachers/lux1/xl/w{1,2,3,4,5}.targets.jsonl`;
  - the control wave `m3/teachers/lux1/xl/c-w1.targets.jsonl`, at revision `0736a4ce…`.

  Every non-H7 / H8 row of every recipe has a Lux1 target, except 34,216 rows of `cx-xl-r2-v2v1-full` and 24,856
  of `cx-xl-r2-v2v1-short`. **That coverage is projected:** all of those rows are in the control wave
  `lux-xl-c-w2`, which was still pending at publication. It will be published as
  `m3/teachers/lux1/xl/c-w2.targets.jsonl`.
- **AutoJev:** `m3/teachers/autojev27/pk1/A0s-strict-train.targets.jsonl` and
  `m3/teachers/autojev27/rp-v2/aj-{m,sl}.targets.jsonl`.
  - These cover 127,985 rows of `mx-xl-full-r2` and 123,794 of `mx-xl-short-r2`.

## Disclosure: familiar readouts

These readouts are familiar for models trained on r2, on r1 or on the underlying arms. The counts are r2 groups
with an overlap hit.

- **SELECT700 / CAL700:** A0s-strict (SELECT 198 groups, CAL 220), and so every recipe. This includes checkpoint
  selection and calibration.
- **v1 AHO:**
  - a1: V1:A1 1,226, H1 48, A7q 8, H3 2, H6 2.
  - a2: G2 2,559, V1:A2 1,451, G6 1,352, G4h 1,241, V1:A4v2h 944, V1:A6g 716.
  - a3: H3 1,426, V1:A3 560, H6 464, E11 65, H1 49, H5 4, V1:A1 2.
  - a4v2h and a4v2r: G2 2,258, V1:A2 1,282, G4h 1,161, G6 1,040, V1:A4v2h 907 / 912, V1:A6g 587.
  - a6g: G6 1,100, G2 695, V1:A6g 654, V1:A2 576, G4h 489, V1:A4v2h 430.
  - a6h: V1:A6h 339, V1:A5 5.
  - a5: V1:A6h 3.
- **A7 AHO:**
  - A7g: A7g 12,097, A0s-strict 952.
  - A7o: A7o 8,646, A7p 1,247, A0s-strict 123, A7r 69, H1 2.
  - A7p: A7p 1,010, A7o 839, A7r 207, A0s-strict 132, H1 2.
  - A7r: A7r 1,475, A7p 516, A7o 203, A0s-strict 39.
  - A7q: A7q 1,324, H3 47, H6 27, H1 25, V1:A3 16, E11 14, H5 11, V1:A1 9, A7i 2, A7m 1.
  - A7m: A7m 97, H3 16, E11 8, H6 3, V1:A3 3, H1 2, A7q 2, H5 1, V1:A6h 1.
  - A7h: H3 12, E11 8, H6 8, V1:A3 5, A0s-strict 2, A7q 2, A7h 1, A7m 1.
  - A7i: A7i 303, A0s-strict 1, H1 1, V1:A1 1.
  - A7s: A7s 223. A7k: H6 1, V1:A5 1. A7x: A7i 1.
- **Evidence strength.** Of the 43,944 disclosed groups, 939 carry exact (E: 333) or long-window near (L: 610)
  evidence; the rest are short-near or n-gram hits.
- **Evaluation panels:** no r2 row has a hit. Models trained on r1 or on the underlying arms saw these hits; see
  `rescreen.public.json` `flagged.by_role`:
  - CSS15 and Decision Bench v4: H3, H6, H1, E11, H5, V1:A3, A7m, A7q.
  - `mlx-diag` and `ml-parallel-dev`: H3, V1:A3.
  - JevBench-231: G6.
- As for H7 / H8, the v2 AHO slices are report-only. For models trained on r2, the H1 / H3 / H5 / H6 / E11
  AHO readouts are passage-familiar.

Rules and results: branch `xunzhuo/decision-2-training-data`, `src/training/decision2/v2/data/records/`
(`m3b-prereg-amendment-2-2026-09-28.md` §4, `m3b-prereg-amendment-3-2026-09-28.md`,
`m3b-prereg-amendment-4-2026-09-28.md`, `m3b-xl-r2-2026-09-28.md`). Private access does not waive upstream terms.
Keep this dataset private.

# `m3/mixtures/xl-r2/` — XL release recipes, revision r2 (private)

Revision r2 of the Milestone 3b XL recipes. **Use r2 for every new run.** r1 (`m3/mixtures/xl/`,
revision `ba848147…`) stays published but is superseded; `rescreen.public.json` gives its flagged counts.

## What changed against r1

1. **Rescreen of every r1 row** (preregistration amendment 3 §2). The 481,432 rows of the six r1 recipes,
   as rendered in their pool files, were scanned with `v2.data.overlap`, using the four methods and
   thresholds of every arm's audit. The scan used only the 58 quarantining roles of PI-v4 (manifest
   `48fd2537…`): evaluation and development panels, SELECT / CAL, and the v1 and A7 held-out slices.
   Each pool was scanned on its own and the whole union once. A group id flagged by either pass is
   removed from every recipe.
   - Flagged: 44,249 groups, 74,404 rows, 49.9M native tokens.
   - 97.6% of the flagged groups are n-gram hits, mostly against the pools' own sibling held-out slices.
2. **Gap arms H7 and H8** (amendment 2 §4).
   - Whole TRAIN groups of `m3/arms/H7` and `m3/arms/H8` (revision `09f73967…`), in
     `sha256("mx-xl-r2:<variant>:<pool>:" + group)` order.
   - A group is skipped if it would put a human source above 8%, or English above 60%, of the r2 total
     (the base plus every candidate group).
   - The short recipes take only groups whose rows are all ≤ 1,024 native tokens. H7 has none.
   - **H7 / H8 rows have no teacher targets. Train them on gold labels.**

## Recipes

"Long" is the share of tokens in rows of ≥ 2,000 native tokens.

| Recipe | Definition | Rows | Native tokens | Choice / Noul / Score % | English % | Long % |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| `mx-xl-full-r2` | `mx-xl-full` − flagged + H7 (all 1,713 groups) + H8 (16,968 of 17,062 groups; 94 TyDi QA groups hit the 8% cap) | 313,407 | 146,509,459 | 31.5 / 47.0 / 21.5 | 55.2 | 29.9 |
| `mx-xl-short-r2` | `mx-xl-short` − flagged + the 15,360 short H8 groups | 320,063 | 103,584,906 | 29.0 / 43.4 / 27.6 | 52.6 | 0 |
| `cx-xl-r2-nogap-full` | `mx-xl-full-r2` minus H7 / H8 | 287,931 | 118,650,623 | 38.0 / 38.0 / 23.9 | 54.3 | 18.2 |
| `cx-xl-r2-nogap-short` | `mx-xl-short-r2` minus H8 | 302,118 | 99,939,007 | 29.1 / 43.2 / 27.7 | 54.5 | 0 |
| `cx-xl-r2-a7v1-full` | `cx-xl-a7v1-full` − flagged | 147,949 | 88,372,366 | 63.2 / 18.4 / 18.4 | 54.1 | 33.7 |
| `cx-xl-r2-a7v1-short` | `cx-xl-a7v1-short` − flagged | 128,937 | 41,861,918 | 55.0 / 13.1 / 31.8 | 53.6 | 0 |
| `cx-xl-r2-v2v1-full` | `cx-xl-v2v1-full` − flagged | 254,522 | 92,760,896 | 14.0 / 60.6 / 25.4 | 53.6 | 6.4 |
| `cx-xl-r2-v2v1-short` | `cx-xl-v2v1-short` − flagged | 245,570 | 81,869,782 | 14.3 / 62.0 / 23.6 | 52.3 | 0 |

**Gap-arm control.** `cx-xl-r2-nogap-*` is the matched control for the gap arms. Train it to the r2
recipe's token budget by repetition. The `a7v1` / `v2v1` controls are published at their own totals, as in
r1.

## Files

- `<recipe>.ids.jsonl`: one line per row, `{id, pool, source, task_type, language, native}`, sorted by
  `id`. `native` is the Qwen3.5-0.8B-Base native token count.
- `<recipe>.lux1.missing.jsonl` and `<recipe>.autojev27.missing.jsonl`: the ids without a published
  target from that teacher.
- `mx-xl-r2.manifest.json` (schema `decision2-mx-xl/2`) holds, per recipe:
  - the r1 fields: tokens by type, language, Score level count, pool and source; English share; teacher
    coverage;
  - long-evidence shares with and without generated pools;
  - the shift against r1;
  - flagged exclusions by pool and the gap-arm additions.

  It also holds the selection report (caps, skipped groups) and the hash of every input.
- `rescreen.public.json`, `rescreen/<pool>.overlap.public.json` and `rescreen/union.overlap.public.json`:
  counts only.
- `check.json`: checks re-derived from the written files:
  - r2 = r1 − flagged + whole H7 / H8 groups, and the controls;
  - no flagged group;
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
  - `m3/teachers/lux1/xl/w{1,2,3,4,5}.targets.jsonl`.

  Every non-H7 / H8 row of `mx-xl-*-r2` and `cx-xl-r2-nogap-*` has a Lux1 target. Every `a7v1` / `v2v1`
  control row that lacks one is in the control waves `lux-xl-c-w1` / `lux-xl-c-w2`. Those waves were pending
  at publication; they will be published under `m3/teachers/lux1/xl/`.
- **AutoJev:** `m3/teachers/autojev27/pk1/A0s-strict-train.targets.jsonl` and
  `m3/teachers/autojev27/rp-v2/aj-{m,sl}.targets.jsonl`.

## Disclosure

- Models trained on r1, or on the underlying arms, saw the flagged rows.
  - `rescreen.public.json` (`flagged.by_role`) lists the roles those rows touch: the A7 and v1 held-out
    slices, CSS15, Decision Bench v4, SELECT / CAL, `mlx-diag`, JevBench-231 and `ml-parallel-dev`.
  - The affected readouts must be disclosed.
- As for H7 / H8, the v2 AHO slices are report-only. For models trained on r2, the H1 / H3 / H5 / H6 / E11
  AHO readouts are passage-familiar.

Rules and results: branch `xunzhuo/decision-2-training-data`, `src/training/decision2/v2/data/records/`
(`m3b-prereg-amendment-2-2026-09-28.md` §4, `m3b-prereg-amendment-3-2026-09-28.md`,
`m3b-xl-r2-2026-09-28.md`). Private access does not waive upstream terms. Keep this dataset private.

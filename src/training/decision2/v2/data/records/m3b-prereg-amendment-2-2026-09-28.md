# M3b preregistration — amendment 2: gap arms H7 / H8, PI-v4, XL recipe revision r2 (2026-09-28)

Committed at 2026-09-28 ~16:10 UTC by the restarted research & data worker. Commit order:

- **Before this amendment:** the draft H7 / H8 builds b1–b4 and their lexical, shortcut and length audits
  (node A, CPU, uncommitted working trees; `m3b-gap-sources-2026-09-28.md`).
- **After this amendment:** the committed-mirror rebuild, the second-node rebuild, the embedding scan,
  any publication and any r2 build.

Nothing from b1–b4 was published or trained on. No C1 registry dataset (any split) is used. The sealed C1
directory is never read.

## 1. Disclosure: what the draft builds changed

- The H7 caps were raised after the first audit (b2) showed about 50% of HoVer and 23% of NQ rows
  quarantined:
  - HoVer twins 2,000 → 4,400, HoVer coverage 1,500 → 3,300, NQ 1,800 → 2,600 rows.
  - NQ extended from 32 to 64 TRAIN shards.
- From b3 on, the builder sizes states with the tokenizer loaded as `row_tokens` loads it
  (`freeze.load_tokenizer`).
- Content rules did not change. These settings are frozen as the b4 settings in `v2.data.m3.src_gap` at
  `360f7c6d1`, where the recovery commit changed formatting only.

## 2. Gap arms H7 and H8

- **Constructions, sources, pins, caps and group keys:** as in the draft record, sections 1 and 2.
  - H7 (English, long evidence): HoVer `hover_answerable` (Noul) and `hover_coverage` (Score); NQ
    `nq_window_removal` (Noul).
  - H8 (multilingual): TyDi QA `tydi_window_removal_<lang>`, MIRACL es / fa / fr / hi / zh
    `miracl_relevance_<lang>` and `miracl_pool_<lang>`, JCommonsenseQA `jcqa` (Choice), and SentiMix
    Spanglish (Score, L = 3).
  - Slicing: AHO `sha256(group_id) % 10 == 0`; SHO `sha256("sho-v2:" + group_id) % 50 == 0` among the
    rest.
  - Budget: 8,192 native tokens.
- **Gates.** Every gate must pass for an arm to be published. A failing cell is dropped as built; a
  shortfall is recorded, never filled by relaxing a rule.
  1. Rights: `license-registry-m3b.json`.
  2. C1 registry check by name and upstream (the eval track's
     `sealed-c1-source-registry-2026-09-28.json`, every grade).
  3. id / group / `input_sha256` isolation against the 69 existing partitions of `existing.json` and
     SELECT / CAL / CAL698.
  4. Four-method lexical overlap against PI-v4 (section 3), with whole-group quarantine.
  5. Embedding scan: Qwen3-Embedding-0.6B@`97b0c614`, v2 thresholds (quarantine at cosine ≥ 0.93, a
     20-pair review sample in [0.85, 0.93)), against the PI-v3 embedding manifest plus the PI-v4 A7
     held-out roles, with whole-group quarantine.
  6. Shortcut gates per (source, family, task type) cell of ≥ 30 rows: state-removed and option-only at
     majority + 5 points. The state-length logistic baseline is reported per cell.
  7. Held-out dedup against TRAIN.
  8. Canonical freeze on node A from an exact mirror of a pushed commit.
  9. Byte-identical rebuild of the build outputs on node B, run on CPU from the same commit, with sources
     downloaded there and hash-checked against node A's receipts.
  10. Dataset-level same-source check of every H7 / H8 source against the CSS pilot and CSS15 tasks, as
      in `m3b-same-source-2026-09-28.md`. A SAME DATASET or SAME PARENT CORPUS source is excluded.
- **Publication.** TRAIN and AHO go to the private HF dataset under `m3/arms/H7/` and `m3/arms/H8/`, with
  per-slice hashes, token files, gate receipts (public parts only) and the licence registry. SHO slices
  move to `/data/dev2/private/sealed/m3b/` on node A (mode 0600). Only their hashes are published; they
  are never uploaded and never read by size tracks.

## 3. PI-v4 (protected inventory for H7 / H8 and every later research & data arm)

- **Unchanged from PI-v3:** the 56 roles and their flags (`fc09b2bd…`).
- **Added as quarantining roles:** every A7 held-out slice. That means the AHO slices of all eleven A7
  sub-arms (A7g, A7h, A7i, A7k, A7m, A7o, A7p, A7q, A7r, A7s, A7x), with hashes equal to the A7 v3
  registry, plus any A7 sealed slice if one exists.
- **Added as report-only roles:** every A7 TRAIN file, and the nine v2 AHO slices.
  - The v2 AHO slices are for information only: v2 TRAIN already shares Wikipedia paragraphs with its
    own AHO by construction.
  - For models trained with H7 / H8, H1 / H3 / H5 / H6 / E11 AHO readouts are therefore disclosed as
    passage-familiar.
  - The private receipts name the groups, so a stricter policy can be re-applied without a rebuild.
- **Manifest:** built on node A from hash-verified files. Its hash is recorded before the first PI-v4
  scan.
- **A7k:** M3b rebuilds no H6 or A6h2 held-out or sealed slice. The record reports the number of A7k
  pairs found in the existing H6 / A6h2 AHO and SHO slices (expected 0). A later rebuild must keep A7k
  pairs out of them.

## 4. XL recipe revision r2 (built only if H7 and/or H8 are published)

- **`mx-xl-full-r2`** = every row of `mx-xl-full` r1 (`db5f4a24…`), plus H7 and H8 TRAIN whole groups.
  - Groups are taken per pool in `sha256("mx-xl-r2:full:<pool>:" + group)` order, up to targets of
    H7 18.8M and H8 12.6M native tokens (all of both arms).
  - Caps: 8% of the r2 total per human source, where the total counts r1 plus the candidates, and
    English ≤ 60% of tokens. A group that would break a cap is skipped in order.
  - r1 rows are never removed, so r2 ⊇ r1 and every r1 teacher target is reused.
- **`mx-xl-short-r2`** = every row of `mx-xl-short` r1 (`67f440a6…`), plus the H7 / H8 whole groups
  whose rows are all ≤ 1,024 native tokens, with the same order and caps.
- **Control.** r1 is the matched control for the gap arms, trained to r2's token budget by repetition.
  The cx-xl-* controls remain r1 controls.
- **Teacher targets.** H7 / H8 rows have none and train on gold labels (coordinator decision 23:50 UTC+8).
  M3b produces no own-Lux targets for them; a later wave would be a new preregistration.
- **Reported:**
  - everything reported for r1 (`m3b-prereg-2026-09-28.md` §2);
  - the long-evidence share: tokens in rows of ≥ 2,000 native tokens (floor 20%, stretch 30%; r1
    XL-full is 19.55%, 29.58M of 151.29M, and A7 generated rows supply 17.6M of those);
  - the human long-evidence share without generated pools;
  - the shift in type shares against r1.

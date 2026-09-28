# M3b preregistration — amendment 3: quarantining-only rescreen of the XL rows; r2 controls; control-row teacher waves (2026-09-28)

Committed at 2026-09-28 ~17:45 UTC. It comes after H7 / H8 were published (`m3b-gap-sources-2026-09-28.md`) and
before any rescreen of XL rows, any r2 build and any control-row teacher wave. It amends amendment 2 §4 only where
stated.

## 1. Why

- The gap-arm audit found that the four-method lexical scan (`v2.data.overlap`) misses hits against quarantining
  roles when large report-only roles sit in the same protected inventory. The report-only rows raise the posting
  counts used for near-duplicate candidate selection and the protected-row boilerplate counts (`d14dff8e8`).
- H7 / H8 are therefore quarantined on the union of the full PI-v4 scan and a scan against PI-v4's quarantining roles
  only.
- Every earlier arm feeding the XL recipes was scanned only in the single, diluted way:
  - v1 and v2 against PI-v3, whose report-only roles are v1 TRAIN and A0 TRAIN;
  - A7 against its own inventories.
- How many hits were missed is unknown.

## 2. Rescreen (CPU, node A, exact mirror)

- **Rows scanned:** every row of the union of the six published r1 recipes (`mx-xl-full`, `mx-xl-short`,
  `cx-xl-a7v1-{full,short}`, `cx-xl-v2v1-{full,short}`, revision `ba848147…`), rendered exactly as in its pool file.
- **Protected inventory:** only the quarantining roles of PI-v4. This is the `manifest.quarantining.json` derived
  from PI-v4 `24b060da…`; its hash is recorded before the scan.
- **Method:** the same four methods, thresholds and code as every arm's audit.
- **Flagged group:** any group of any pool with a hit against a quarantining role. Counts are reported by pool, source,
  role and method. Hits against report-only roles are not produced by this scan and are not reported here.
- **Embedding scans:** not repeated, because pairwise cosine is not diluted by report-only roles.

## 3. r2 recipes (replaces "r2 ⊇ r1" in amendment 2 §4)

- **`mx-xl-full-r2`** = (`mx-xl-full` r1 minus every flagged group) + H7 / H8, selected exactly as amendment 2 §4
  (order, targets, caps).
- **`mx-xl-short-r2`**: the same rule, applied to the short recipe.
- **Controls, published as id lists:**
  - `cx-xl-r2-nogap-{full,short}` = r2 minus its H7 / H8 rows. This is the matched control for the gap arms, trained
    to r2's token budget by repetition.
  - `cx-xl-r2-a7v1-*` and `cx-xl-r2-v2v1-*` = the r1 controls minus flagged groups.
- **r1 stays published.** It is marked superseded for new runs, and its flagged counts are disclosed. Candidates
  already trained on r1, or on the underlying arms, disclose the roles their flagged rows touch.
- **Shares reported:**
  - the long-evidence share (tokens in rows of ≥ 2,000 native tokens) with and without generated pools;
  - type, English and language shares;
  - teacher coverage per recipe.

## 4. Own-Lux targets on the control rows (completes M3b prereg §3)

- **The gap:** prereg §3 covers every row of XL-full ∪ XL-short ∪ their controls, but waves w1–w5 were cut from the
  XL-full and XL-short missing lists only. The 100,957 control-only rows (64.9M native tokens) got no wave. This
  amendment records that deviation.
- **Control waves:** `lux-xl-c-w1` (60,000 rows) and `lux-xl-c-w2` (the rest). They use the same launcher, image,
  mirror and node-B autotune cache on node B GPU7, one at a time, after w5. The guard, the conversion with per-row
  attestation and publication under `m3/teachers/lux1/xl/` follow w1–w5.
- **Flagged rows:** targets for rows the rescreen later flags are published but not referenced by any r2 recipe.

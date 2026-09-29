# HT-DEV v1 preregistration, amendment 1 (2026-09-29): isolation-check scope details

Committed before the source-level checks (§3 items 2, 3, 5(ii), 6, 7) run. Thresholds are unchanged.

1. **Source snapshot scope.** "Every split of the source" means every split of the dataset the task
   draws from:
   - BRIGHTER: the `eng` configuration only (the other 27 languages are separate annotation sets).
   - Hyperpartisan: the four by-article files of Zenodo 10.5281/zenodo.5776081 (the by-publisher set,
     1.3 GB of distantly labelled articles, is a different labelled set and is not used).
   - Moral Stories: `data/moral_stories_full.jsonl` only; the 100+ classification/generation configs
     are re-arrangements of the same 12k stories.
   - NYCC: the whole repository (all explanation, matching and ranking configs).
   - GitHub sources: every data file under the listed paths of the commit tarball (UKPConvArg: all of
     `data/`, not only the Strict CSVs).
2. **PubHealth** is taken from the Hugging Face parquet conversion `refs/convert/parquet`
   (`7f898a16838e708a4597986e3fab8af05b710ba4`) of `ImperialCollegeLondon/health_fact@57995242`, the
   loader that downloads the authors' archive; the archive's own sha256 is not re-derived.
3. **IBM-ArgQ-9.1kPairs** (persuasion backup) has no reachable pinned archive (IBM download URLs return
   404; no mirror). It is recorded as not checked and cannot replace UKPConvArg.
4. **Converted formats.** Tab-separated files named `.csv` (UKPConvArg), header-less TSV (XED), the
   WiC-TSV line-aligned text files, the HateXplain post dictionary, Swords contexts, Diplomacy
   dialogues (one row per message) and the hyperpartisan XML are converted to JSON lines by
   `v2/eval/htdev_iso/prepare.py` so that `independence rows` sees every text.
5. **Embedding scan (item 5(ii)) scope.** One GPU job of at most 30 minutes: the 14 primary sources'
   rows (deduplicated normalised row text per source) against the training rows of the
   social-text-adjacent sources selected by provenance. Backup sources are embedded only if their
   primary fails an admission check (a backup can only enter the panel in that case).
6. **Flagged rows** handed to the builder: lexical REVIEW or OVERLAP (containment ≥ 0.2, or any exact
   match) and embedding cosine ≥ 0.93, identified by the sha256 of `schema.normalized(text)` of each
   ≥ 20-character string leaf, so the builder can drop them independently of its own row order.

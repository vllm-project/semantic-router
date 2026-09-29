# HT-DEV v1 preregistration, amendment 2 (2026-09-29): build decisions before the freeze

Committed after the source-level isolation checks (`htdev-isolation-2026-09-29.md`) and a local dry run
of the builder (counts only), and before the node-A build, the item-level scans, the freeze and any
model scoring. No model has been run on HT-DEV. Thresholds, the scoring functional, the validation
metrics and the decision rule are unchanged.

1. **Formality (`formality/pavlick`) fails admission check 1 and is dropped.** The only licence
   statement is the re-uploader's card (CC BY 3.0); the authors' release states none and its archive
   link no longer serves the data. There is no backup, so the task is dropped.
2. **Diplomacy grouping.** The per-game cap (≤ 10 per game, 12 test games) makes the 150 floor
   impossible (dry run: 120 items, 14 lies). Caps apply per dialogue (game + player pair, ≤ 10 per
   dialogue); the paired bootstrap resamples games. If the task still misses the floor it is dropped
   (its backup, CaSiNo, was rejected by the isolation check).
3. **Backups also replace a primary that misses the floor or keeps fewer than two gold classes after
   caps**, in the preregistered order, before the freeze.
4. **Split order.** Test first; validation is added only if the test pool cannot fill the 250 cap;
   train only if the task is still below the 150 floor. A single-split source (Circa) uses its only
   split.
5. **Fig-QA lineage check.** Six Fig-QA rows match FLUTE training rows (CSS15's `flute` is FLUTE).
   Before the build, the FLUTE paper and data card are checked for any portion derived from Fig-QA.
   If one exists, Fig-QA fails the dataset-level rule and MAGPIE replaces it (after its embedding
   scan). Otherwise Fig-QA stays and the six rows are dropped with the other flagged rows.
6. **Flagged rows.** Every source row flagged in `ADMISSION.json` (lexical REVIEW or worse, embedding
   ≥ 0.93) is removed from the pool before item-level scans and selection.
7. **Item-level lexical scan scope.** The pool is scanned against every training corpus in the frozen
   training manifest, the six named gold-free evaluation prompt files, and the SELECT / CAL rows.
   The historical training-file versions missing on node A are fetched first, or listed in the build
   record if unobtainable.
8. **Rendering details.**
   - NYCC entity links are shown as their page titles; no URLs appear in states.
   - Choice options are keyed by display position (A, B, …), and the display order comes from the
     salted item hash.
   - Noul criteria and Score levels (ascending) are fixed per task.
   - Circa groups by (situation, question).
   - Leftover class quota goes to the larger classes within group caps.
9. **Length cues.** The gate stays Choice-only. A Choice task that still fails it after its single
   length-stratified re-draw is excluded. Length-only baselines are reported for every task. The
   source-level length correlations of hyperpartisan (0.587 macro-F1 vs chance 0.5) and Measuring
   Hate Speech (0.405 vs 0.333) are disclosed, not gated.
10. **Validation scope.** 27B is not run: its weights, caches and kernel image are only on node B. The
    runner's lease option is written `--shared-lease eval-htdev`, which creates the preregistered
    entry `owner.eval-htdev`.

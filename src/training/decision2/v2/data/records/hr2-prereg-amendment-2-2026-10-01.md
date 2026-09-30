# HR2 preregistration — amendment 2: review sample after the G4 drops (2026-10-01)

Committed after the audits G1–G4 and finalize pass 1, **before any row is sampled for review or shown to a
reviewer**. Nothing in §2 or the audit rules changes.

G4 dropped two families by its preregistered rule (gated view above majority + 0.05; group-disjoint 5 folds on
pass-1 TRAIN): `indonli` (hypothesis-only .613 vs threshold .535) and `kob_boolq` (the question alone, i.e. the
`state_removed` view, .559 vs .534). They are not part of HR2, so §4 samples the **9 remaining families**:

1. **Sample:** 24 TRAIN rows per family × 9 families = **216 rows**, drawn from pass-1 TRAIN without the two
   families (finalize with `--drop-families indonli kob_boolq`), otherwise exactly as §4 (one row per group,
   stratified by gold, `hash("hr2-review-v1:" + id)` order). Packets: `hash("hr2-packet-v1:" + id)` order, two
   packets of 108 items per reviewer order.
2. **Thresholds unchanged as rules:** P1 pooled error ≤ 5.0% and exact 95% upper bound ≤ 8.0% — for n = 216 that is
   **≤ 9 errors** (9 / 216 = 4.17%, upper 7.76%; 10 / 216 gives 8.35%); P2 population-weighted ≤ 5.0% over the 9
   families; P3 no family with ≥ 5 errors of 24; fix rule F1 as preregistered.
3. **Split (operational):** R1 and R2 split when exactly one of them agrees with gold (for Choice / Noul this is
   R1 ≠ R2; for Score an answer agrees within one level). R3 answers only the split items; the split list is computed
   on node A, where the keys stay.

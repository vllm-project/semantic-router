# IB3 preregistration — amendment 1: G4 audit-copy fix; `maud` leaves (2026-10-01)

Committed **after G4 on run `d1` (pass 2) and before any screen or review sample was drawn**; no IB3 row has been shown
to a reviewer. The prereg (`b30bcae5c`) is unchanged except where this amendment says so. Only counts were read.

## A. G4 audit copy (tool fix, no rule change)

The G4 step renames each family's declared hypothesis field to `claim` in the audit copy (prereg §3). `v2.data.shortcut`
validates every row's `input_sha256` against its payload, so it refused the renamed copies of `fdial`, `haluqa`,
`esci`, `mqa` and `maud`; only `wpd` and `phiu` (no renamed field) were audited. The audit copy now recomputes
`input_sha256` after the rename (the training rows are not touched). G4 runs again on the same pass-2 TRAIN file into a
fresh directory (the first receipts are kept as `shortcut-v1/`); the gate, views and threshold are unchanged.

## B. `maud` leaves IB3 (too few groups to review)

After G0 and G2, `maud` keeps **86 TRAIN rows in 11 contract groups and no DEV row**: 100 of the contract groups were
quarantined because their excerpts near-duplicate Decision Bench v4 rows (a protected panel source; G2 method N) or
held-out development-panel rows. Stage R needs `max(18, ceil(216 / F))` rows per family from distinct groups, which
this family cannot supply, and a family of 86 rows adds nothing a track could measure. As IB1's `sumedit` addendum,
the family leaves before the screen (listed in `amendment-drop-families.txt` on the node; `final` drops it), and
**contracts stay uncovered** in IB3. With F = 6 surviving families (if G4 passes them all), stage R takes 36 rows per
family (216).

## C. Run

Run `d1` continues. Re-scan 1 dropped 42 more groups and re-scan 2 (of pass 2) dropped 11 more (G2 quarantine; G0 and
G0u found nothing new), so **pass 3** (every drop list through re-scan 2) is the sample pass (`IB3_SAMPLE_PASS=3`): G4
runs on pass 3 with the fixed audit copy (the pass-2 receipts are kept as `shortcut-v1/`), re-scan 3 must be clean
(at most three rounds, prereg §3), then the screen and the review run as preregistered.

# 0.6B Milestone 4 amendment (M4e): in-distribution Noul check replaces the typed-DEV Noul floor

Written 2026-09-28 after the T soup's readout and **before** the V2 readouts and any formal
run. It changes only the formal-run gate of part 2 §3 / M4c; it is a post-readout change
and is disclosed as such for the coordinator's review.

## What happened

The T soup (uniform average of T s1–s3, same init) has P 32.74 ≥ the T seed mean 32.42, so
it is T's candidate artifact under the 17:15 rule. It clears the P ≥ 30.5 gate and the
CSS pilot H (.319 ≥ .281) and typed Choice (255 ≥ 186) floors, but typed-DEV Noul is
182 < 186.

## Why the typed-DEV Noul floor is replaced

- Part 1 (recorded before this readout, `m4b-prereg` §0): the typed-DEV Noul slot is one
  family (`rule_precedence`) on which every 0.6B checkpoint has no ranking signal (AUC
  .44–.53) and every model up to 4B in the eval matrix scores .39–.56. Values between about
  182 and 200 are the constant answer plus noise (the constant "false" answer scores 192,
  always "true" 208).
- The eval track's calibrated guidance is never to select on typed-DEV Noul; the
  coordinator's Milestone 4 candidate rule (seed mean → soup if ≥ mean → formal runner) has
  no Noul floor. The floor was inherited from the Milestone 2 bar, set when a constant Noul
  answer was still suspected to be a bug.
- What the floor was meant to catch — a broken Noul readout — is measured directly in
  distribution: T soup SELECT Noul 267/290 (.92), CAL Noul 273/290 (.94), equal to every T
  and C seed (.90–.93).

## New gate (applies to every Milestone 4 candidate)

CSS pilot H ≥ .281, typed-DEV Choice ≥ 186/800, **SELECT Noul ≥ .85 and CAL Noul ≥ .85**
(replacing typed-DEV Noul ≥ 186/400), and P ≥ 30.5. Typed-DEV Noul stays reported, and the
formal report's typed FINAL Noul (different families) is read for any Noul regression.
Candidate choice (seed mean, tie rule, soup vs median seed, at most two formal runs) is
unchanged.

# M3b preregistration — amendment 4: which rescreen hits exclude rows from r2 (2026-09-28)

**Written AFTER the amendment-3 rescreen results** (`m3b-xl-r2-2026-09-28.md` §2) and after the strict r2 build was
published at HF `9f83ce9b940b6aa945845166e97079eb387faeb1`. Nothing has trained on that build. This amendment changes
one rule and states why. Both builds and all counts stay on record.

## 1. What the strict rule did

Amendment 3 §3 removed every group with a hit against any of PI-v4's 58 quarantining roles. On the r1 union that is
44,249 groups: 15.5% of rows and 21.0% of tokens. `mx-xl-full` loses 21.6% of its tokens:

- 22% of the A0s-strict anchor;
- 28% of A7g;
- 39–42% of G2 / G6;
- 80–85% of V1:A2 / A6g / A3.

### Where the hits land

- **Rare n-grams only:** 93.6% of the flagged groups.
- **Held-out slices of the same lineage:**
  - A7 sub-arms against their own and sibling A7 AHO slices;
  - G2 / G6 / G4h and the v1 generated arms against the v1 A2 / A4 / A6g AHO slices of the same generator
    lineage;
  - H3 / H6 / V1:A3 against the v1 A3 AHO, which holds MuSiQue paragraphs drawn from the same Wikipedia pages as
    HotpotQA / 2Wiki;
  - the anchor against the A7 Stage 4 generator's AHO slice.
- **The anchor against SELECT700 / CAL700:** these are A0's own development splits, built by the same generators.

### Why a quarantining-only scan over-flags these

The scan leaves out the report-only TRAIN rows that let `v2.data.overlap` recognise template and passage boilerplate.
It therefore treats n-grams shared by construction as rare.

- **Held-out slices:** they are in-family diagnostics. Their separation from TRAIN is group-level isolation, which
  holds. Template and passage familiarity with their own lineage is inherent, as for the v2 AHO slices (amendment 2
  §3).
- **SELECT700 / CAL700:** in-distribution by design.

## 2. Rule for r2 (replaces amendment 3 §3's "every flagged group")

- **A group is excluded from every r2 recipe and control if it has a hit, by any method, against an evaluation or
  benchmark role.** That covers every PI-v4 quarantining role except:
  - the v1 AHO slices;
  - the A7 AHO slices;
  - SELECT700 and CAL700.

  The excluding roles are, for example:
  - CSS15;
  - Decision Bench v4;
  - `mlx-diag`, JevBench-231, `ml-parallel-dev`;
  - the typed panels and the CSS pilot (no hits).
- **Hits whose only roles are held-out slices or SELECT700 / CAL700 are disclosed, not excluded.** They are reported:
  - by role and pool;
  - with the number that include exact (E) or long-leaf-near (L) evidence.

  The readouts they touch are disclosed as familiar for models trained on these pools.
- Everything else in amendments 2 §4 and 3 §3 is unchanged:
  - H7 / H8 selection, order, caps and English cap;
  - the controls `cx-xl-r2-nogap-*` and `cx-xl-r2-{a7v1,v2v1}-*`, now "minus excluded groups";
  - coverage and reporting.
- The published strict build stays retrievable at `9f83ce9b…` and is recorded as **r2-strict (superseded)**.
- The final r2 is published at a new revision of `m3/mixtures/xl-r2/`.
- H7 / H8 themselves keep their stricter union quarantine (`m3b-gap-sources-2026-09-28.md`).

# 9B M10 amendment 11: drops after the KIB4W2 / KIB4L2 results (2026-10-03)

Written 2026-10-02 ≈19:50Z (10-03 03:50 UTC+8) by the M10 continuation worker, before any KIB4-a50, KIB4R-a40,
KIB4W3-a40 or Y result exists. At writing, KIB4-a50's shards 0–3 were running, and X7-a40's gate bootstrap was
running. Amendments 7–10 are unchanged, except as noted here.

## Evidence (values private)

- **KIB4W2-a40** (IB4 phase-1 rows at twice the loss weight) is significantly below KIB4-a40. Its RAGTruth and
  PhishNChips scores, the benchmarks behind KIB4's gain, fell back to about K-a13IB's level.
- **KIB4L2-a40** (backbone LR 2e-5) is significantly below K-a13IB.
- **AF-KF-a40** (all nine KIB4-family seeds) is significantly below KIB4-a40.

## Changes

1. **`KIB4W3-a40` is dropped** (IB4 dose ×3). It moves further in the direction that KIB4W2 measured as harmful.
2. **`KIB4R-a40` stays** (KIB4's recipe with a different x60 cut: data diversity, not a dose change). It is measured
   once when the factory's node B soup exists.
3. **`Y1` / `Y2` become conditional.** F* is the best measured arm-factory point. They are built only if F* is not a
   KIB4 s1 / s2 point (KIB4-a50 would make Y1 an α interpolation, not a cross-arm average) and F*'s point delta vs
   KIB4-a40 is above −0.3. Otherwise they are dropped.
4. **The one more KIB4-family arm** (amendment 7, item 4) is not planned. The two recipe variants measured so far are
   both significantly worse, so no measured lever points to an arm with a plausible gain. That changes only if
   KIB4-a50 or KIB4R-a40 has a positive point delta vs KIB4-a40.
5. **`KIB4Q-a50` and `KFxKIB-a40`** stay last (amendment 10), budget permitting.

# Arm factory — amendment 9: a standing backlog of ready-to-launch arms (2026-10-02 ≈21:10Z)

Written before any backlog weights file was built and before any backlog GPU job. COORDINATION 2026-10-03 04:58
(UTC+8): keep a standing backlog of at least four ready-to-launch 4B and 9B arms, and launch onto any GPU idle for more
than 20 minutes without waiting for a note (claim order: 4B owner, 27B #6, arm factory, 9B publisher). Every arm
reuses an audited TRAIN file; only loss weights or the recipe change. Two seeds each (20260926 / 20260927 for 4B;
9B seeds as listed), launched in this order onto idle, lease-checked GPUs.

| # | Arm | Size | TRAIN (audited) | Change | Why |
| --- | --- | --- | --- | --- | --- |
| 1 | `4b-SDMLIB4-UP2` | 4B | `4b-SDMLIB4` `b51fab6a…` | kept released rows ×2.0, IB rows ×1 | a stronger UP dose (RAGTruth rides on UP, 4B owner 03:35) |
| 2 | `4b-LHS17IB4-UP2` | 4B | `4b-LHS17IB4` `dfed3944…` | the same | as 1 |
| 3 | `4b-LHS17IB4ML-UP` | 4B | `4b-LHS17IB4ML` `3ca6a795…` (amendment 1) | UP ×1.5 | UP on the factory's IB4 + ML mixture |
| 4 | `4b-LHS23IB4-UP` | 4B | `4b-LHS23IB4` `68301f88…` (amendment 1) | UP ×1.5 | UP on the S23 swap dose |
| 5 | `KIB4-lrh` | 9B | `KIB4` `2e72bcfd…` | backbone LR 5e-6 (half) | the LR direction opposite to KIB4L2 (seeds 15 / 16) |
| 6 | `KIB4R-W2` | 9B | `KIB4R` `e95e32ce…` | IB4 p1 rows ×2 | the re-cut data with the IB4 dose (seeds 17 / 18) |
| 7 | `KIB4-e2` | 9B | `KIB4` `2e72bcfd…` | two epochs; seed cap 6.0 GPU-h | an epoch variant (seed 19, one seed) |

- UP / IB4 weights come from `af_weights.py` (teacher-target rows, or IB4 p1 ids), as in amendments 4 and 8.
- 4B arms train on node C (or node F); 9B arms on node A or node B. Node gates: A 32, B 28, C 44 GPU-h.
- Hand-off as before: 4B seeds to the 4B owner (it builds soups and reads the Index), 9B seeds and their two-seed
  α = .4 points to the 9B publisher. The 9B arms are recipe variants inside the KIB4 family; new-family 9B arms
  follow the 9B publisher's deficit amendment (COORDINATION 04:58) when it names them.
- Budget: ≈ 11 (4B) + ≈ 16 (9B) GPU-h; the factory total stays ≤ 130.

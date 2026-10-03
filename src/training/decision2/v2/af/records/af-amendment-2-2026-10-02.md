# Arm factory — amendment 2: exact candidates, members and measurement order (2026-10-02 ≈15:05Z)

Written before any arm-factory soup was built and before any arm-factory Index result exists (all 15 seeds are in
their full runs). Prereg [`af-prereg-2026-10-02.md`](af-prereg-2026-10-02.md), amendment 1
[`af-amendment-1-2026-10-02.md`](af-amendment-1-2026-10-02.md). Known at writing: M17's `4b-SDMLxALL9` is equal within
noise to `4b-SDMLxALL` (private values), so more members alone did not help; `4b-SDMLxALL15` is still scoring.

## Change: one arm, one vote

The prereg added each factory seed-extension soup as its own member. Instead, a factory seed extends the owner's arm
soup to the exact uniform soup of all its seeds, and that arm soup is one member (the `SDMLxALL` design). FP32 soups
are linear, so `[x3 soup × 3, factory 2-seed soup × 2]` is the uniform five-seed soup (M10's KXP construction).

## 4B (node F holds M17's soups; the factory's node C soups are shipped there, SHA-256 lists equal)

| Candidate | Members (uniform FP32 average) |
| --- | --- |
| `4b-LHS17IB4-x5` | M17 `4b-LHS17IB4-x3` × 3, factory `4b-LHS17IB4-s45` (s4, s5) × 2 |
| `4b-SDMLIB4-x5` | M17 `4b-SDMLIB4-x3` × 3, factory `4b-SDMLIB4-s45` × 2 |
| `4b-LHS17UP-x4` | M17 `4b-LHS17UP` × 2, factory `4b-LHS17UP-s34` (s3, s4) × 2 |
| **`4b-AFxALL`** (prereg rule) | the better of M17's `SDMLxALL9` / `SDMLxALL15` member sets by their Index runs (`SDMLxALL15` only if ≥ 0.10 above `SDMLxALL9`), with `4b-LHS17IB4-x3` → `-x5`, `4b-SDMLIB4-x3` → `-x5`, `4b-LHS17UP` → `-x4`, plus `4b-LHS17IB4-lrh`, `4b-LHS17IB4ML`, `4b-LHS23IB4` (12 members on the `SDMLxALL9` set) |
| **`4b-AFxALL2`** (the prereg's one more weighting) | `SDMLxALL`'s seven (SDML, S17, UP, IB4, IB4X, SDMLIB4, S17ML) with IB4 → `-x5`, SDMLIB4 → `-x5`, UP → `-x4`, plus the three new arms (10 members) |
| `4b-LHS17IB4-lrh`, `4b-LHS17IB4ML`, `4b-LHS23IB4` | each arm's two seeds |

- Seed soups: each LoRA seed's BEST checkpoint merged into FP32 first (`m10_merge.py`, SELECT agreement check).
- **Measured, in this order:** `4b-LHS17IB4ML`, `4b-LHS23IB4` (as soon as built), `4b-AFxALL`, `4b-AFxALL2`,
  `4b-LHS17IB4-lrh`. The `-x5` / `-x4` / `-s45` soups are members only (handed over unmeasured).
- 4B Index runs on node C (panel-8, reference `IS-4b-LHA10SDML-bf16`) and on node F with the scoring environment and
  that reference run's merged results copied from node C (results SHA-256 pinned `a459ce7c…`).

## 9B (node A)

| Candidate | Members (uniform FP32 average; Lux = the pinned Lux 1.0 zero-step member) |
| --- | --- |
| `KF` (family soup, 9 seeds) | M10 `KIB4P` × 3 (= KIB4 s1–s3, imported from node B, lists equal), factory KIB4 s4, s5, KIB4W2 s1, s2, KIB4L2 s1, s2 |
| **`KF-a40`** / **`KF-a50`** | `[KF, KF, Lux, Lux, Lux]` / `[KF, Lux]` |
| **`KIB4W2-a40`** / **`KIB4L2-a40`** | `[s1, s2, Lux, Lux, Lux]` of the arm (= its two-seed soup at α = .4) |
| `KFK` | `[KF, KF, KIB]` (KIB = K-a13IB's arm soup, imported from node B) |
| **`KFxKIB-a40`** | `[KFK, KFK, Lux, Lux, Lux]` |
| `KIB4Q` | `[KIB4P × 3, KIB4 s4, s5]` (five-seed KIB4), member / hand-over only |

- **Measured, in this order:** `KF-a40`, `KF-a50`, `KIB4W2-a40`, `KIB4L2-a40`, `KFxKIB-a40` (panel-7, reference
  `K-a13IB-bf16`, node A's copy). A failed or capped seed is left out of every soup (disclosed).

## Budget

Training ≈ 32, merges ≈ 1, Index ≈ 5 × 2.2 + 5 × 2.7 ≈ 24.5 GPU-h: ≈ 57.5 of 60. The last run in each order is
dropped if the running total would pass 57 at its start.

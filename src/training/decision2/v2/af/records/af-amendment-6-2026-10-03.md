# Arm factory — amendment 6: a 9B α ladder for the seed soups (CPU builds and staging only; 2026-10-02 ≈18:25Z)

Written after the factory's only 9B measurement (`AF-KF-a40-bf16`, value private) and before any point below was
built. It is an adaptive choice made on Index results (the factory's `KF-a40` and M10's KIB4 points), disclosed as
such; the 9B publisher decides what to measure and applies its gate.

## Why

In M10's runs a two-seed soup beat a three-seed soup of the same recipe at the same α (`KIB4` vs `KIB4P` at α = ⅓),
the KIB4 curve still rose from α = ⅓ to .4, and the factory's nine-seed `KF` at α = .4 measured lower than both.
Averaging more full-fine-tuning seeds cancels part of each seed's step away from Lux 1.0, so a soup of more seeds may
need a larger α. No α above .4 has been measured for any 9B soup.

## Points (uniform FP32 soups with the pinned Lux 1.0 zero-step member L; node A, CPU)

| Point | Members | α |
| --- | --- | --- |
| `KIB4-a50` | `[KIB4, L]`, KIB4 = M10's two-seed soup (the released `KIB4-a40`'s arm soup), imported from node B, lists equal | .5 |
| `KIB4Q-a50` / `KIB4Q-a60` | `[KIB4Q, L]` / `[KIB4Q × 3, L × 2]`, KIB4Q = five pure KIB4 seeds | .5 / .6 |
| `KF-a60` | `[KF × 3, L × 2]` | .6 |

- Each point is staged as its BF16 release copy (`af-stage.sh`, `AF-<point>-bf16`) for the 9B publisher; the factory
  runs no Index for them. `KF-a50` (amendment 2) is staged the same way.
- No GPU time: soups, BF16 copies and restaging are CPU jobs.

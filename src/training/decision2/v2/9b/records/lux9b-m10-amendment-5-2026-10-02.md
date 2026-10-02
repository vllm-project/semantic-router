# 9B M10 amendment 5: cross-arm points (X1–X6) and the speculative formal path (2026-10-02)

Written ≈17:50 UTC+8 (09:50Z), after COORDINATOR UPDATE 17:25 (KUP dropped; formal runs started speculatively in parallel
with each candidate's Index run). It was written before any KIBM, KSW, KIB4 or KX point was measured and before any
cross-arm point was built.

## Cross-arm points (no training)

For k frozen arm soups S1…Sk and the pinned Lux 1.0 zero-step member L, X(α) is the uniform FP32 soup of
[S1, …, Sk, L × m], so each arm carries α / k and Lux 1.0 carries 1 − α. At a33, m = 2k; at a25, m = 3k; at a40,
m = 3k / 2 (even k only). For a33, X equals the per-tensor mean of the arms' own a33 points, which is the "CPU weight
average of the best M10 soups". Building and checking it is `m10/xarm.sh` (CPU, node A or B). Arm soups move between
nodes with `ix.sh soupcopy`, and SHA-256 lists must be equal on both sides.

| Name | Arms | When |
| --- | --- | --- |
| X1 | KX, KIB4 | once both arm soups exist (prior: both target the K-a13IB deficits) |
| X2 | KX, KIB (K-a13IB's arm soup, i.e. an average with the current release) | once KX's soup exists |
| X3 | KIB4, KIB | once KIB4's soup exists |
| X4 | the two M10 arms with the highest measured a33 lower bounds, if that pair is not KX + KIB4 | after those arms' a33 runs |
| X5 | the three M10 arms with the highest measured a33 lower bounds | after those arms' a33 runs |
| X6 | the best measured M10 arm with KIB, if X2 / X3 does not already cover it | after those arms' a33 runs |

- **Point.** a33 is primary. A cross-arm point is also built at a25 or a40 only when that α gave the best measured
  single-arm point of a member arm.
- **Measurement.** Each point is measured once, on its BF16 release copy (`M10-X<n>-<aNN>-bf16`, registered in the IX1
  launcher), with the same gate as every other candidate.
- **Order.** X points are measured in the order they become buildable, after the a33 points of KIB4 and KX.
- **Audit.** A point that contains KIB is released only after the contamination audit is rerun with K-a13IB's TRAIN
  added to the audited sets. Its planted control must be complete. K-a13IB's TRAIN was audited for its own release.

## Speculative formal path

The formal path (`m9/formal.sh`, through `m10/formal.sh`) starts for the KIB4-a33 and KX-a33 points when their soups
are built, in parallel with their Index runs. Later candidates start it as soon as their Index lower bound is > 0.

It runs on node A, because the CAL698 inputs, the comparator runs, the frozen formal Triton cache and the Lux 1.0
package exist only there. It can use any node A GPU whose lease is free; GPU6 / GPU7 were only M9's allocation.

A formal run counts toward the budget whether or not its candidate passes. A formal result never replaces the Index
gate.

## Budget

90 GPU-h. Cross-arm points cost no GPU time to build and ≈ 2.7 GPU-h per Index run. Each formal run costs ≈ 2 GPU-h.
No new Index or formal run starts once the M10 total reaches 85 GPU-h.

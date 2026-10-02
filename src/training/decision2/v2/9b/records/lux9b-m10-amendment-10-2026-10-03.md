# 9B M10 amendment 10: the arm factory's α ladder and the revised order (2026-10-03)

Written 2026-10-02 ≈19:25Z (10-03 03:25 UTC+8) by the M10 continuation worker, after COORDINATION 03:04. That note
puts the 9B publisher's α ladder (arm-factory amendment 6, `6b11fb647`) ahead of the factory's batch 3.

At writing, KIB4W2-a40 was scored but its bootstraps were running and no value of it had been read. KIB4L2-a40 and
X7-a40 were still running. The known verdicts were AF-KF-a40 and X8-a40, both FAIL (state 19:05Z). Amendments 7–9
are unchanged, except as noted here.

## The α ladder (factory points on node A; FP32 soups built by the factory)

- **`KIB4-a50`** = `[KIB4, Lux]`: M10's two-seed KIB4 soup (seeds s1 / s2; model `2406b058…`, imported by the factory
  from node B) at α = 1/2. These are the released point's seeds. KIB4-a33 → a40 moved the point by less than its
  interval, so a50 is the ladder's only point with a plausible chance. It is measured first, once, as
  `M10-KIB4-a50-bf16` (linked with `xpts.sh link`, BF16 copy by `ix.sh bf16`).
- **`KIB4Q-a50`** (five-seed KIB4) is measured last, only if the budget allows. KIB4Q's seeds s3–s5 are in `KF`, whose
  a40 point is significantly below KIB4-a40. KIB4P-a33, with s3, was below KIB4-a33.
- **`KIB4Q-a60` and `KF-a60` are dropped**, for the same reason plus the KF-a50 drop (state 19:05Z).

## Revised order of the remaining measurements

1. Running: X7-a40, KIB4W2-a40, KIB4L2-a40.
2. `KIB4-a50`.
3. The batch-2 points `KIB4W3-a40` and `KIB4R-a40` (amendment 9).
4. `Y1`, then `Y2` (amendment 7; F* = the best measured arm-factory point).
5. `KIB4Q-a50`, then `KFxKIB-a40`.

## One more KIB4-family arm (amendment 7, item 4)

X8-a40's positive point delta meets the condition. The design waits for items 1–3: the KIB4W2 dose result and the
batch-2 points are the evidence for a dose or recipe change. It needs its own amendment and a budget line, since the
current budget does not fit it beside items 2–4.

## Budget

At writing, about 12.2 GPU-h are committed, including the running shards. Items 2–3 take ≈ 8.1, and a release about 2,
for ≈ 22. Items 4–5 fit only partly under the 27 GPU-h stop rule. Whatever does not fit waits for a budget line from
the coordinator.

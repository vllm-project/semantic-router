# 9B M10 amendment 9: the arm factory's full 9B hand-off (2026-10-03)

Written 2026-10-02 ≈18:25Z (10-03 02:25 UTC+8) by the M10 continuation worker, after the factory's hand-off
(COORDINATION 02:00 / 02:22). The factory has stopped its node A Index script, and the 9B publisher measures its 9B
points. Node A GPU1–4 are M10's (COORDINATION 02:00).

At writing, `AF-KF-a40-bf16` was scored and its bootstraps were running. No arm-factory 9B Index value and no
continuation gate result had been read. X7-a40 and KIB4W2-a40 were still running, and X8-a40's bootstraps were
running. Amendments 7 and 8 are unchanged, except as noted here.

## Candidates and order

1. `AF-KF-a40`: the factory's run on node A, gated vs `M10-KIB4-a40-bf16` (amendment 7).
2. `KIB4W2-a40` (node C GPU1–4) and `KIB4L2-a40` (node A GPU1–4): amendment 8. KIB4L2-a40 moves to node A, where
   it starts at once. Node C's chain keeps only KIB4W2-a40, so node C GPU1–4 free up when it ends.
3. `KF-a50` (amendment 7's list): measured by M10 as `M10-KF-a50-bf16`, once, on node A GPU1–4 after KIB4L2-a40.
4. **The factory's batch-2 points** (KIB4W3, KIB4R; their seeds end ≈ 19:40Z): each arm's two-seed soup at α = 2/5,
   built on node B where the seeds are (`[s1, s2, Lux, Lux, Lux]`, the factory's construction). They are measured once
   each as `M10-KIB4W3-a40-bf16` and `M10-KIB4R-a40-bf16`, on the first free M10 GPUs. KIB4R's TRAIN is in audit
   `m10c` (0 item rows); KIB4W3's TRAIN is KIB4's with other loss weights.
5. `Y1`, then `Y2` (amendment 7). F* ranges over every measured point of items 1–4.
6. `KFxKIB-a40`: last, only if the budget allows it. It is X3's construction (KIB4 family plus KIB), and X3 measured
   below its KIB4 parent.

Each point is measured once on its BF16 release copy, and the gate is amendment 7's IF1. A point that the factory
measures after all is not measured again.

## Budget

The 30 GPU-h and the 27 GPU-h stop rule are unchanged. Items 1–5 take about 8 Index runs. If items 4–6 do not fit,
they wait for a budget line from the coordinator.

# 9B M10 amendment 3: arm (d) KIB4, and two-seed KSW / KIB4 (2026-10-02)

Written ≈13:55 UTC+8 (05:55Z) by worker 7e1c9ce8, after COORDINATION 13:45 (IB4 phase 1 published; training workers
may start IB4 arms) and before any KIB4 row was built or any KSW / KIB4 seed ran.

## Arm (d) KIB4

- **Change vs K-a13IB:** IB1 `sentfin` dropped (IB4's `sentfin3` replaces it) and IB4 phase 1 added:
  `llm-semantic-router/decision-2.0-training-data` `m6/ib4/p1` @ `76cea510`, TRAIN `6045b456…` (9,459 rows: `sqa2`
  3,694, `sentfin3` 2,661, `isarc2` 2,636 (in-distribution), `fc_pick` 468; tokens `2af4f8a5…`). The x60 cut is
  re-done to K-a13's 60,183,732 native tokens with K-a13IB's cut seed (`prep.sh kib4`, `m9_data.py --ib4`), exactly
  as KIBM did for IB3. Own-Lux KL on the x60 rows, IB rows gold only; the rest of the recipe is K-a13IB's.
- **Release:** IB4 is "release-safe pending C1": a KIB4 candidate can ship only after the custodian's C1 recheck r3
  passes (`v2/eval/records/c1-recheck-r3-2026-10-02.md`). It is measured on the Index like any other candidate.
- **Placement:** phase-2 chains on node B GPU6 / GPU7 after KIBM-s1 / s2 (≈ 07:35Z).

## Two seeds for KSW and KIB4

KSW (amendment 1 had three seeds) and KIB4 train **two seeds each** (20260926 / 1), the M17 4B design, so both arms
finish together (≈ 10:30Z) on four GPUs instead of a third seed ending ≈ 13:00Z. The arm soup is the uniform FP32 soup
of both seeds' BEST; a single finished seed is the arm artifact (disclosed). KSW-s1 / s2 run on GPU2 / 4 after the
labeling (unchanged); the labeling uses GPU2 / 4 only (two shards).

## Budget

Training ≈ 7.8 (KUP) + 7.8 (KIBM) + 5.2 (KSW) + 5.2 (KIB4) + ≈ 0.6 labeling ≈ 27 GPU-h; Index ≈ 2.7 per candidate.
The M10 total stays ≤ 60; the node gate (50 node-B GPU-h) still stops new seeds.

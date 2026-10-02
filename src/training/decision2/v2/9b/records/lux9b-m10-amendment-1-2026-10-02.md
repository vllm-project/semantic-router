# 9B M10 amendment 1: arm (c) KSW, the 4B swap recipe at 9B (2026-10-02)

Written ≈13:15 UTC+8 (05:15Z) by the 9B frontier worker (7e1c9ce8), before any KSW row was built, any teacher target
was computed or any KSW GPU job ran. KUP and KIBM (prereg) are unchanged and running. Index values stay private.

## Arm (c) KSW

M17's 4B swap (`4b-LHS17SD`, the highest 4B candidate) differs from K-a13IB in three ways; KSW carries all three to
9B and changes nothing else in the K-a13IB recipe:

| | K-a13IB | KSW |
| --- | --- | --- |
| x60 cut to the matched tokens | every stratum (all languages) | **English rows only**: every x60 recipe group with a non-English row is kept whole; the cut runs over the all-English groups (same strata, same seed `20261001:m9-s3:keep`, same 1% tolerance) to 60,183,732 − kept IB − kept non-English native tokens |
| IB rows | IB1-r3 + IB2 | IB1-r3 **without `sentfin`** + IB2 (COORDINATION 01:05); no IB3, so KSW isolates the swap from KIBM's maths block |
| teacher on kept x60 rows | own Lux 1.0, KL 1.0 | **the released K-a13IB** (self-distillation, KL 1.0, T = 1: the release ships `calibration: null`) |

- IB share ≈ .16 (K-a13IB .168 with `sentfin`; M17's 4B winner .17).
- **SD targets** (`m10/ksw.sh teach`, `v2.dec.teacher_label --teacher-kind dec --uncalibrated`, source path Lux 1.0):
  the teacher is K-a13IB's FP32 soup, rebuilt on node B as `[KIB-soup, Lux zero-step, Lux zero-step]` from node C's
  M9 KIB soup and M10's pinned Lux member. It is accepted only if its model SHA-256 equals K-a13IB's (`4701ba41…`, the
  identity the release's BF16 copy was made from). The labels cover exactly the kept x60 rows, each once (four shards
  on node B GPU2 / 4 / 6 / 7 after a one-shard pre-warm); per-type agreement with gold is reported. IB rows get no
  teacher (`--teacher-partial`).
- **Placement:** labeling starts once the phase-1 chains on GPU2 / 4 / 6 / 7 finish (≈ 07:30Z); then three KSW seeds
  on GPU2 / 4 / 6 (seeds 20260926 / 1 / 2) under the same preflights, seed cap, node gate and markers; GPU7's lease
  goes idle. The arm soup, the a33 / a25 / a40 points and their Index measurement follow the prereg.
- **Budget:** labeling ≈ 1 GPU-h, training ≈ 8 GPU-h; the M10 total stays ≤ 60 (the node gate of 50 stops new seeds).
- A failed labeling or teacher-identity check stops the arm (no rerun).

## Arm (d) + IB4

Not started: written as its own amendment only after the IB4 record announces a release-safe phase.

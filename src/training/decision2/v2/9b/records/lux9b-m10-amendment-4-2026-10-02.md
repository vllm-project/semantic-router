# 9B M10 amendment 4: arm KX (IB4 phase 1 + IB3-r2 + the M15 ML block) on node A; budget 90 GPU-h (2026-10-02)

Written ≈15:55 UTC+8 (07:55Z) by worker 7e1c9ce8 after COORDINATOR UPDATE 15:50 (0.6B / 0.8B / 2B paused; node A
GPU1 / 2 / 7 for 9B training as the M18 worker releases them; IB4 phase 1 release-safe, C1 r3 PASS; budget 90 GPU-h),
before any KX row was built or any KX GPU job ran. KUP, KIBM, KSW and KIB4 continue unchanged.

## Arm KX

- **TRAIN** (`prep.sh kx`, node B, CPU): K-a13IB's construction with IB1 `sentfin` dropped and both new blocks added:
  IB1-r3 − `sentfin` + IB2 + IB3-r2 `mqa` (`1c8452da`, `9d92d92a…`) + IB4 phase 1 (`76cea510`, `6045b456…`: `sqa2`,
  `isarc2` (in-distribution), `sentfin3`, `fc_pick`), x60 cut to K-a13's 60,183,732 native tokens with K-a13IB's cut
  seed (`m9_data.py --ib3 --ib4`).
- **ML block** (`m10_ml.py`, M15's construction): one `~m2` copy of whole kept x60 groups whose every row is
  non-English, per language in proportion to the kept multilingual tokens (M15's `upsample`, seed 20261002), adding
  U = (s·T − ML − I_ml) / (1 − s) tokens so the TRAIN multilingual token share equals x60's s (M15's tolerance .002).
  Copies carry their originals' own-Lux targets. This adds tokens beyond the matched 60.18M, as M15 did.
- **Recipe:** K-a13IB's (own-Lux KL 1.0 on x60 rows and their copies, IB rows gold only, CE + 0.5·Brier, backbone 1e-5
  / head 1e-4, `even8`, SELECT700 `matrix-v1`), **three seeds** (20260926 / 1 / 2), uniform FP32 soup, then the
  a33 / a25 / a40 points against the Lux zero-step member.
- **Placement:** node A GPU1 (KX-s1, pre-warm) / GPU2 / GPU7, each seed waiting ≤ 4 h for its GPU's lease to be
  released (`track=9b-m10` then). Node A's Lux 1.0 copy (`/data/decision20-20260926/...`) equals node B's `bd45a30a`
  file for file (74 SHA-256 equal); node A's Triton cache is M9's `f83b1d10`. The TRAIN is copied node B → C → A and
  re-hashed (`prep.sh kx-lock`). The points' Lux member is KX-s1's zero-step checkpoint, checked against M9's
  `m9-KIB-s1-zero` SHA-256 list on node A.
- **Release:** IB4 is release-safe (C1 r3 PASS, COORDINATION 15:50), so KX and KIB4 candidates ship like the others.

## Index placement

The shared pool now includes node A GPU0 / 3–6 as they free up; M10 Index chains run greedy over node C GPU1–7 and node
A GPU3–6 (packages restaged on node C, copied with `ix.sh pkgcopy`).

## Budget

90 GPU-h (COORDINATION 15:50). Planned: ≈ 26 (KUP, KIBM, KSW, KIB4, labeling) + ≈ 8.5 (KX) + ≈ 2.7 per Index run.

# 9B M10 state (running log, newest first; prereg `lux9b-m10-prereg-2026-10-02.md`)

Index values stay private (node private run directories and the coordinator's private folder); this file has none.

## 2026-10-02 04:55Z (12:55 UTC+8), worker 7e1c9ce8

- **Training (node B, image `f83b1d10`, leases `track=9b-m10`):** chains launched 04:40Z from mirror `240c8ca79`.
  - KUP-s1 GPU3 (pre-warm done 04:45Z, preflight PASS 04:46:53Z), KUP-s2 GPU2, KUP-s3 GPU4, KIBM-s1 GPU6,
    KIBM-s2 GPU7: zero-step and one-step done 04:47–04:50Z, full runs in progress. KIBM-s3 is queued on GPU3 after
    KUP-s1.
  - Expected ≈ 2.6 GPU-h per seed: the first five end ≈ 07:30Z, KIBM-s3 ≈ 10:10Z.
- **Data locked** in `$M/data/READY-m10.json`: KUP weights `ae696d93…` (x60 weight share .677 → .758); KIBM TRAIN
  `aadca52f…` / teacher `b4cfc3be…` (151,375 rows, 60,391,810 tokens, IB share .1815; 6,808 `sentfin` rows out, 8,752
  IB3-r2 `mqa` rows in).
- **Lux member pinned:** M10's zero-step checkpoint (KUP-s1) is byte-identical to M9's `m9-KIB-s1-zero` (10 weight
  files; `$M/inputs/lux-zero-m9-KIB-s1.sha256`), K-a13IB's Lux member.
- **Post chains (node B, CPU) launched 04:52Z from mirror `ed69b9657`:** `post.sh KUP` / `post.sh KIBM` wait for their
  seeds, then build the 3-seed soup and the a33 / a25 / a40 points.
- **Index measurement:** node C (package `DEV2.0-9B-e51f9881`, panel-7, reference run `K-a13IB-bf16` local); the pool
  GPU1–7 is busy with other Index runs until ≈ 08:10Z. Driver `m10/ix.sh` (ship → bf16 → chain → status → fetch).
- **GPU-h:** ≈ 1 so far (5 seeds started); projection ≈ 16 training + ≈ 2.7 per Index run.
- **Next:** at ≈ 07:30Z ship KUP's points to node C; BF16 copies; Index chain on the free node C GPUs; then KIBM.
  Arms (c) 9B swap and (d) + IB4 are amendments, not yet written.

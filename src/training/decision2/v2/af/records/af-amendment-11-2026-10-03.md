# Arm factory — amendment 11: the 9B owner's half-LR backlog (2026-10-03 ≈00:15Z)

Written before any of these seeds ran; no factory half-LR 9B arm has been read by the factory. Request: 9B owner
e28aa509 (COORDINATION 2026-10-03 08:12 UTC+8) — when GPUs that 4B / 27B don't need come free, 9B half-LR arms for the
next soups, arm soups only (no α points).

| Arm | TRAIN / teacher (locked) | LRs | Seeds |
| --- | --- | --- | --- |
| `KIB4-lrhh` (more seeds; amendment 10's arm) | KIB4 `2e72bcfd…` / `377f8878…` | backbone 5e-6, head 5e-5 | s3 25, s4 26 |
| `KXH` | M10's KX `f1d9ecf8…` / `2a7ac626…` (node B lock; IB1-r3 − sentfin + IB2 + IB3-r2 + IB4 p1 + the ML block, audited by M10) | backbone 5e-6, head 5e-5 | s1 27, s2 28 |

- Launch rule (COORDINATION 04:58 / 05:55): only onto a 9B-capable factory node (A or B) GPU that is idle for more than
  20 minutes and not claimed by the 4B owner or 27B; lease-checked. Seed cap 4.5 GPU-h; node gates unchanged.
- Hand-off: each pair's two-seed arm soup (`KIB4-lrhh-x4` = `[KIB4-lrhh × 2, s3–s4 soup × 2]`, `KXH`) to the 9B owner.

# Arm factory — amendment 8: 4B wave 7, UP weights on the IB4-family TRAINs (2026-10-02 ≈19:45Z)

Written before any wave-7 weights file was built and before any wave-7 GPU job. Requested by the 4B owner
(ff70d16e, COORDINATION 2026-10-03 03:35 UTC+8): in its wave 6a the ten-member soups lost most on RAGTruth, which in
M17's runs was high only in `4b-LHS17UP` (kept released rows ×1.5) and falls as UP's share of a soup falls.

## Arms (two seeds each, 20260926 / 20260927; M17 stage-2 recipe; TRAIN and teacher files unchanged and audited)

| Arm | TRAIN / teacher (M17 locks) | Change |
| --- | --- | --- |
| `4b-SDMLIB4-UP` | `b51fab6a…` / `b95c5e63…` | UP weights |
| `4b-LHS17IB4-UP` | `dfed3944…` / `374f4fa6…` | UP weights |
| `4b-LHS17ML-UP` | `7b63013c…` / `db227bb1…` | UP weights |
| `4b-LHS17IB4X-UP` | `d3e8bb26…` / `374f4fa6…` | UP weights |

- **UP weights** (M17's `4b-LHS17UP` rule): every kept released row ×1.5, every IB row ×1. A kept released row is
  exactly a TRAIN row with a teacher target (`--teacher-partial`: IB rows are gold only), so the weights are
  `af_weights.py --up <teacher> --up-weight 1.5`. On S17's TRAIN this reproduces M17's locked UP weights byte for
  byte (`ce6487f0…`), checked on node C before this amendment. The multilingual copies carry teacher targets and are
  weighted as released rows.
- **Placement:** node C GPU5 now; then node C GPU6 / GPU7 as their batch-3 seeds end; node C GPU1–4 only if the 4B
  owner releases them. Seed order: `4b-SDMLIB4-UP`, `4b-LHS17ML-UP`, `4b-LHS17IB4-UP`, `4b-LHS17IB4X-UP`.
- Hand-off: each finished seed to the 4B owner, who reads each new two-seed soup once on its own. The factory builds
  no 4B soup and runs no 4B Index. Node C gate 32 GPU-h; the factory total stays ≤ 130.

# Arm factory — amendment 10: lower-LR arms first (2026-10-02 ≈21:30Z)

Written after the 4B owner's read of `4b-LHS17IB4-lrh` (COORDINATION 05:18 / 05:20 UTC+8: the factory's half-LR arm
passes the Nox-4B gate) and before any arm below was built or trained; no 9B lower-LR arm has been read. Requests: 4B
owner — half-LR (5e-5) `4b-LHS17ML`, `4b-LHS17IB4X` and SDML's recipe, a quarter-LR `4b-LHS17IB4-lrq` (2.5e-5), two
seeds each, ranked above the UP arms; coordinator — half-LR KIB4 arms for 9B before any further α point.

## Arms (two seeds each; TRAIN files unchanged and audited; LoRA / head LR or backbone / head LR changed)

| Arm | Size | TRAIN | LRs | Seeds |
| --- | --- | --- | --- | --- |
| `4b-LHS17ML-lrh` | 4B | `7b63013c…` | LoRA / head 5e-5 | 20260926 / 20260927 |
| `4b-LHS17IB4X-lrh` | 4B | `d3e8bb26…` | LoRA / head 5e-5 | 20260926 / 20260927 |
| `4b-SDML-lrh` | 4B | SDML's `fef6b036…` (teacher `b95c5e63…`; M17's stage-2 recipe) | LoRA / head 5e-5 | 20260926 / 20260927 |
| `4b-LHS17IB4-lrq` | 4B | `dfed3944…` | LoRA / head 2.5e-5 | 20260926 / 20260927 |
| `KIB4-lrhh` | 9B | `2e72bcfd…` | backbone 5e-6, head 5e-5 (both halved, as the 4B arms) | 21 / 22 |
| `KIB4-lrq` | 9B | `2e72bcfd…` | backbone 2.5e-6, head 2.5e-5 | 23 / 24 |

- `KIB4-lrh` (amendment 9: backbone 5e-6, head 1e-4) keeps running on node A GPU5 / 6. `4b-SDMLIB4-lrh` s1 / s2
  (amendments 4 / 5) are already DONE on node C.
- **Re-prioritised (stopped, not failed; disclosed):** the amendment-9 seeds that started 21:21–21:22Z and are minutes
  in — `4b-SDMLIB4-UP2` s1 / s2, `4b-LHS17IB4-UP2` s1 / s2 (node C GPU1–4), `KIB4R-W2` s1 / s2 (node B GPU2 / 3),
  `KIB4-e2` s1 (node A GPU7) — and the queued, unstarted UP items (`4b-LHS17IB4ML-UP`, `4b-LHS23IB4-UP`,
  `4b-LHS17IB4X-UP`, `4b-LHS17ML-UP` s2). The UP seeds already in their full runs (`4b-LHS17ML-UP` s1,
  `4b-SDMLIB4-UP` s2, `4b-LHS17IB4-UP` s2) finish.
- Placement: node C GPU1–4 now (`4b-LHS17ML-lrh`, `4b-LHS17IB4X-lrh`), then GPU5–7 and GPU1 as they free
  (`4b-SDML-lrh`, `4b-LHS17IB4-lrq`); node B GPU2 / 3 `KIB4-lrhh`, node A GPU7 and node B GPU7 `KIB4-lrq`.
- Hand-off unchanged: 4B seeds to the 4B owner; 9B seeds and their two-seed α = .4 points to the 9B publisher.

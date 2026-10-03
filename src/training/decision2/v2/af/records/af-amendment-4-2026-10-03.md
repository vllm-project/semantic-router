# Arm factory — amendment 4: batch 2 (+30 GPU-h, more GPUs; 2026-10-02 ≈16:15Z)

Written before any batch-2 file was built and before any batch-2 GPU job; no arm-factory Index result exists yet.
COORDINATION 2026-10-03 00:00 (UTC+8): the factory may expand onto any GPU idle for more than 15 min (lease-checked;
never node C GPU0, node E GPU4–5, node F GPU0–1, nor node D / E GPUs that 27B #6 holds), with +30 GPU-h (90 total),
for more diverse 4B and 9B arms. Idle at 16:05Z (released leases): node A GPU7, node B GPU2–4 / 6–7, node C GPU1–2 /
6–7, node F GPU6–7.

## Arms (owners' recipes, one stated change each)

| Arm | Size / node | Data | Change | Seeds (run sN) |
| --- | --- | --- | --- | --- |
| `4b-LHS17IB4X` | 4B / C | M17 lock `READY-m17s3` (`d3e8bb26…`, teacher `374f4fa6…`) | none: more seeds (its m50 point was among M17's best) | s3 20260928, s4 20260929 |
| `4b-LHS17ML` | 4B / C | M17 lock `READY-m17s4` (`7b63013c…`, teacher `db227bb1…`) | none: more seeds | s3 20260928, s4 20260929 |
| `4b-SDMLIB4-lrh` | 4B / F | `4b-SDMLIB4`'s TRAIN | LoRA / head LR 5e-5 (half) | s1 20260926, s2 20260927 |
| `KIB4W3` | 9B / B | `KIB4`'s TRAIN + `af_weights.py`, IB4 p1 rows ×3 | IB4 dose ×3 | s1 seed 9, s2 seed 10 |
| `KIB4R` | 9B / B | `af-prep9b.sh kib4r`: KIB4's construction with x60 cut seed `20261003:af-r1:keep` | a different kept subset of the released rows | s1 seed 11, s2 seed 12 |

- `KIB4R` is a new TRAIN file: it gets the row-level Index contamination audit (IX1 method, planted controls) and its
  hashes in the state record before its seeds start (its chains wait for the lock entry).
- Node B trains 9B with M10's node B inputs (Lux 1.0 `bd45a30a`, SELECT700 / CAL698, a copy of M10's node B Triton
  cache; node B's first 9B seed pre-warms). Node A GPU7 and node B GPU7 stay free (budget).
- Node gates (training GPU-h; no seed starts above): node A 26, node C 17, node F 9, node B 15 (9B).

## Candidates (amendment-2 rules; measured once each, BF16 release copy)

- 4B `4b-AFxALL3`: `SDMLxALL`'s seven with every arm at its most seeds (`4b-LHS17IB4-x5`, `4b-LHS17IB4X-x4`,
  `4b-SDMLIB4-x5`, `4b-LHS17UP-x4`, `4b-LHS17ML-x4`, plus SDML and `4b-LHS17SD`), plus the factory's new arms
  (`4b-LHS17IB4-lrh`, `4b-LHS17IB4ML`, `4b-LHS23IB4`, `4b-SDMLIB4-lrh`): 11 members. `-x4` = `[M17 two-seed soup × 2,
  factory s3–s4 soup × 2]`.
- 9B `KF2-a40`: `KF`'s nine seeds plus `KIB4W3` s1–s2 and `KIB4R` s1–s2 (13 seeds, uniform), at α = .4.
- Measured in this order after the amendment-2 candidates: `4b-AFxALL3`, `KF2-a40`. Budget: batch 2 ≈ 8 (4B
  training) + 14 (9B training) + 5 (two Index runs) ≈ 27 GPU-h; the factory total stays ≤ 90.

# Arm factory — amendment 5: batch 3 under the new split (2026-10-02 ≈18:15Z)

Written before any batch-3 file was built and before any batch-3 GPU job. COORDINATION 2026-10-03 01:38–02:00 (UTC+8):
+40 GPU-h (130 total); the factory trains arms only (the 4B owner ff70d16e builds 4B soups and runs the 4B Index; the
9B publisher 9087b208 measures 9B); batch 3 runs on node C GPU5–7 and node B GPU4 and aims at the remaining gaps (4B
≈ +0.6 to JPT-4B, 9B ≈ +0.6 to JPT-9B) with new seeds, IB4 / ML dose and LR / epoch variants.

## Arms (owners' recipes, one stated change each)

| Arm | Size / GPU | Data | Change | Seeds |
| --- | --- | --- | --- | --- |
| `4b-SDMLIB4W2` | 4B / C5, then C6 | `4b-SDMLIB4`'s TRAIN + `af_weights.py`: IB4 p1 rows ×2 | IB4 dose ×2 on the SDML base | s1 20260926, s2 20260927 |
| `4b-SDMLIB4-e2` | 4B / C7 | `4b-SDMLIB4`'s TRAIN | two epochs (`--epochs 2`); seed cap 3.5 GPU-h | s1 20260926 |
| `KIB4R2` | 9B / B4, then B6 | `af-prep9b.sh kib4r2`: KIB4's construction with x60 cut seed `20261003:af-r2:keep` | a third kept subset of the released rows | s1 seed 13, s2 seed 14 |

- `4b-SDMLIB4-lrh` s1 / s2 (amendment 4, lost at the node F return) run first on C7 / C6 (since 18:08Z); the batch-3
  4B seeds follow on those GPUs' flocks.
- `KIB4R2` is a new TRAIN file: row-level audit before its seeds start (their chains wait for the lock entry). Its
  seeds follow `KIB4R` s1 / s2 on node B GPU4 / GPU6.
- Node gates: node C 21, node B 22 (9B). The factory total stays ≤ 130.
- Hand-off: every finished arm (seeds + two-seed soup) goes to the 4B owner / 9B publisher, who decide soups and
  measurements. The factory measures nothing in batch 3.

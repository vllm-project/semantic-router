# Decoder M17 stage 2 — amendment 1: arm (a) data, wave-1 data lock (2026-10-02 ≈06:10Z)

Written after wave 1 started training and before any arm-(a) file was built or trained; no stage-2 result exists.
Prereg [`dec-m17-stage2-prereg-2026-10-02.md`](dec-m17-stage2-prereg-2026-10-02.md) (`5bedf6f6f`).

## Wave 1 data (built on node F by `m17-prep2.sh`, `1843bbee7`; locked in `m17/data/READY-m17s2.json`)

| Arm | TRAIN | Teacher | Weights |
| --- | --- | --- | --- |
| `4b-LHS17UP` | `14bce13c…e25a0` (= the locked `4b-LHS17SD` TRAIN, hard link), 71,088 rows | `374f4fa6…` (= S17's) | `ce6487f0…4089`: 48,838 released rows ×1.5, 22,250 IB rows ×1; released share of the loss weight .687 → .767 |
| `4b-LHS23SD` | `c629913c…f6c8`, 75,576 rows, T+11 tokens, IB share .2299, 39.3% of the English candidate tokens removed | `a012a5e1…3f5c` | — |

- Deviation (operational, disclosed): the first `m17-prep2.sh` run (`5bedf6f6f`) pre-created the output directory,
  which `m17_data.py` refuses, so it stopped before writing any row. The empty directory and the failure marker were
  removed (its log kept as `data/4b-s2.void-mkdir.log`) and the fixed script (`1843bbee7`) built the data. Stage 1's
  cross-node data-lock comparison was not repeated: `m17_data.py` is deterministic and stage 1 reproduced it byte
  for byte on nodes E and F.
- Wave 1 started 06:00:56Z on node F GPU2 / 3 (`4b-LHS17UP` s1 / s2) and GPU6 / 7 (`4b-LHS23SD` s1 / s2).

## Arm (a) is two arms (IB4 guidance: report `isarc2` separably)

| Arm | TRAIN (built by `m17-prep3.sh`, locked in `READY-m17s3.json`) |
| --- | --- |
| **`4b-LHS17IB4`** | the locked `4b-LHS17SD` TRAIN byte for byte, then every row of IB4 phase 1 TRAIN (`m6/ib4/p1/ib4.train.jsonl` @ `76cea510`, `6045b456…06fb`, 9,459 rows: `sqa2` 3,694, `sentfin3` 2,661, `isarc2` 2,636, `fc_pick` 468), then every row of IB3-r2 TRAIN (`m6/ib3/ib3.train.jsonl` @ `1c8452da`, `9d92d92a…2dea`, 8,752 `mqa` rows; node F copy staged by M18) |
| **`4b-LHS17IB4X`** | the same without IB4's `isarc2` rows |

- Both are additive (above T, ≈ +3.5M tokens); the new rows are gold only (the S17 teacher, `--teacher-partial`).
  IB1 `sentfin` is already absent (dropped in stage 1), as IB4 requires with `sentfin3`.
- IB4 TRAIN was fetched on node F from the dataset revision; its hash equals the published LFS object id.
- Wave 2 chains (`M17_STAGE=3`): GPU2 / 3 `4b-LHS17IB4` s1 / s2, GPU6 / 7 `4b-LHS17IB4X` s1 / s2, each queued on
  its GPU's flock behind the wave-1 chain. Recipe, seeds, caps and stop rules as the prereg.
- Release of either needs the C1 recheck r3 PASS, plus the prereg's gate and integrity checks. The contamination
  audit covers the new TRAIN files.
- `4b-LHS23UP` (the prereg's alternative wave) is not trained unless a wave-2 arm stops early.

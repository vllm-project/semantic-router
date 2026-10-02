# Arm factory — amendment 1: 4B wave-2 data lock and audit; node F placement (2026-10-02 ≈14:40Z)

Written before any wave-2 GPU job; no arm-factory result exists. Prereg
[`af-prereg-2026-10-02.md`](af-prereg-2026-10-02.md) (`75da74ff6`).

## Wave-2 TRAIN files (`ops/af_prep4b.py`, `35130dd49`; built on node C, rebuilt on node F byte for byte)

| Arm | TRAIN | Rows (base + added) | Added families | Teacher |
| --- | --- | --- | --- | --- |
| `4b-LHS17IB4ML` | `3ca6a795…6005abe` | 93,551 (71,088 + 22,463) | `sqa2` 3,694, `sentfin3` 2,661, `isarc2` 2,636, `fc_pick` 468, `mqa` 8,752, ML copies 4,252 | `db227bb1…` (= M17 `4b-LHS17ML`'s: S17's, then the copies' SDML teacher rows) |
| `4b-LHS23IB4` | `68301f88…df38b4ce1` | 93,787 (75,576 + 18,211) | `sqa2` 3,694, `sentfin3` 2,661, `isarc2` 2,636, `fc_pick` 468, `mqa` 8,752 | `a012a5e1…` (= `4b-LHS23SD`'s) |

- Inputs hash-checked against the M17 / M15 locks (S17 `14bce13c…`, S23 `c629913c…`, SDML `fef6b036…` / teacher
  `b95c5e63…`, IB4 p1 `6045b456…`, IB3-r2 `9d92d92a…`); `4b-LHS17IB4ML`'s first 89,299 rows are byte-equal to M17's
  locked `4b-LHS17IB4` TRAIN (`dfed3944…`). Ids unique; every copy has its teacher row.

## Row-level Index contamination audit (IX1 method; node C, CPU, `v2.eval.ix1.contamination`, panel-7)

- 120,226 Index rows; planted control **200 / 200 found, 0 missed**.
- `4b-LHS17IB4ML`: 93,551 lines, **0 item rows**, 90 duplicate-class rows. `4b-LHS23IB4`: 93,787 lines, **0 item
  rows**, 81 duplicate-class rows (the same class M17's audits reported for its locked files).
- Output private (node C `ix1/audit/af4b-w2/out`, `audit.json` `79bede43…`).

## Placement

- The four wave-2 seeds (prereg seeds 20260926 / 20260927) run on node F GPU2–5. Those GPUs held stale harness leases
  of M17's finished `SDMLxALL15` Index run (all eight shards exit 0 at 14:14Z, no pool process left); the owner files
  are moved to `owner.prev-af-*` and replaced by `track=arm-factory` leases. Node F GPU6–7 are left to M17.
- Node F's first wave-2 seed pre-warms node F's arm-factory Triton cache (a copy of M17's node F `4b-train` cache).

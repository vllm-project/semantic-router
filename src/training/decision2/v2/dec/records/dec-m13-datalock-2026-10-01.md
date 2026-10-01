# Decoder Milestone 13 — data lock (2026-10-01)

Prereg [`dec-m13-prereg-2026-10-01.md`](dec-m13-prereg-2026-10-01.md) (`c2610143b`). Built from mirror `c2610143b` by
`ops/m13/m13-prep.sh` (CPU, decoder image `dbe5f32b`) on nodes E and F, 13:21–13:22Z; self-distillation targets by
`ops/m13/m13-teach.sh` (13:23–13:29Z). No M13 training job has run. After this record is pushed,
`m13/data/READY-m13.json` (arms and per-arm teachers below) is written on both nodes; every seed re-hashes its TRAIN and
teacher against it.

## TRAIN (byte-identical on nodes E and F)

| Arm | TRAIN SHA-256 | Rows | Built as |
| --- | --- | --- | --- |
| `08b-RASD` | `12bd63d8b215877b6f471c12652e534e0f0be794c15bff74ad5ac50e98018bf3` | 309,225 | hard link of M12 `08b-RA` (locked) |
| `08b-RAAG` | `69b8d00d6cdd2d48ee0c0d840ba07954eba404aa48ffbaf34a6cd3d8aefbb6cd` | 359,919 | M12 `08b-RA` + copies 2, 3 of 25,347 proxy-family rows |
| `2b-RASD` | `08140409a6afc439e35bc4e0aefc9b5812162c4a2972e5e1da5b137a99d0e592` | 91,146 | hard link of M12 `2b-RA` (locked) |
| `4b-LHA5` | `4e5316aab361b8ce15b0494c235577b549645ffb3e5a8d917f3098c416a18628` | 65,789 | `m12_data.py` `4b-LHA5=all:0.05`, seed 20261001 |
| `4b-LHA10SD` | `d41cdd1aa1e63b47394ed8fa61a87bf461cb4d2cea08cc7ff12f2e58b36dc9d5` | 72,847 | hard link of M12 `4b-LHA10` (locked) |

- `4b-LHA5`: 30,873,747 tokens; IB 7,050 rows / 1,469,208 tokens = **5.0%** of T₄ (29,404,539).
- `08b-RAAG` proxy rows (each now at weight 3): `stage4_scope` 9,067, `scoped_sources` 7,202, `stage3_evidence_scope`
  6,212, `stage4_replay_stage3_evidence_scope` 837, `stage4_replay_policy` 836, `stage4_replay_authorization` 829,
  `authorization` 364; all are released TRAIN rows (checked by the builder). +50,694 rows (+16.4% over `08b-RA`).

## Self-distillation targets (`teacher_label --teacher-kind dec --uncalibrated`, T = 1, one per tier)

| Tier (node) | Teacher | Rows (= released TRAIN) | SHA-256 | Gold agreement choice / Noul / Score |
| --- | --- | --- | --- | --- |
| 0.8B (E) | DEV2.0-0.8B (`bede7938`, source Eos 1.0) | 162,696 | `18827abcc49ffa73fb06ffbda68661848f373417d279f544d6c255fadbade4f6` | .869 / .824 / .823 |
| 2B (F) | DEV2.0-2B (`a53cf66a`, source Sol 1.0) | 56,141 | `2b9858d89fb650db9bc3e7fa608755135b81d7b20ff7b1b6fa186c86bf2e49f2` | .884 / .892 / .772 |
| 4B (F) | released LH (M10 LH soup, source Qwen3.5-4B-Base) | 58,739 | `7639fab17c719bb3ed7a18bd16397130f109c8bd204d17c45f6743d6f0c17496` | .919 / .905 / .702 |

- Every target file covers each released TRAIN id exactly once (checked at the join); four shards per tier after a
  pre-warm shard alone; ≈ 200 s (0.8B), 160 s (4B), 70 s (2B) per shard.
- Agreement is the four shards' range midpoint (per-shard values in `m13/OPERATIONS.log`). The teachers' Score
  agreement on TRAIN is the lowest (4B .70, 2B .77): self-distillation pulls the Score head towards the release's own
  Score behaviour, not towards gold.

## Per-arm teachers (`READY-m13.json`)

| Arm | Teacher | SHA-256 |
| --- | --- | --- |
| `08b-RASD` | 0.8B SD targets | `18827abc…` |
| `08b-RAAG` | none | — |
| `2b-RASD` | 2B SD targets | `2b9858d8…` |
| `4b-LHA5` | own-Lux (LH's) | `e2ff27ce2fc397c8c203e37133dcfac52ac3d1f172c6f3e9412ec28c2ffe296c` |
| `4b-LHA10SD` | 4B SD targets | `7639fab1…` |

# Decoder M14 — data lock (2026-10-01)

Prereg [`dec-m14-prereg-2026-10-01.md`](dec-m14-prereg-2026-10-01.md) and ops `e48a66f46` (committed and pushed 14:59Z;
the header's "≈15:20Z" was an estimate, the commit time is authoritative). Built from mirror `e48a66f46` by
`ops/m14/m14-prep.sh` (CPU, decoder image `dbe5f32b`) on node A (15:04Z) and node B (15:05Z). No M14 training job has
run. After this record is pushed, `m14/data/READY-m14.json` (per arm: `train`, `weights`, `teacher`) is written on
both nodes; every seed re-hashes its files against it.

## TRAIN (M12's locked files, unchanged) and weights (byte-identical on nodes A and B)

| Arm | TRAIN = M12 arm (SHA-256, M12 data lock `d506595c2`) | Weights SHA-256 | Rows | Released / IB weight share | Score share |
| --- | --- | --- | ---: | --- | ---: |
| `08b-RAUP` | `08b-RA` `12bd63d8b215877b6f471c12652e534e0f0be794c15bff74ad5ac50e98018bf3` | `99b77df529a3d566ee92caa85431f76bea817d8379943bf8d5ad2d50664d2a94` | 309,225 | .625 / .375 (rows .526) | .083 |
| `2b-RAUP` | `2b-RA` `08140409a6afc439e35bc4e0aefc9b5812162c4a2972e5e1da5b137a99d0e592` | `e1ae614465cee0dc654761a9c4fbf7ac5bfe0a02973c12a3c02ae3f6a4300c0d` | 91,146 | .721 / .279 (rows .616) | .203 |
| `4b-LHA10UP` | `4b-LHA10` `d41cdd1aa1e63b47394ed8fa61a87bf461cb4d2cea08cc7ff12f2e58b36dc9d5` | `3f741da8ab4b973d78194cc8695e82e180b17c50d33a073bfde9d0e708155562` | 72,847 | .862 / .138 (rows .806) | .209 |

- TRAIN files were pulled from node E with M12's `train.ids.jsonl` (`5b831161…`, `22854cfc…`, `e429ce9b…`, equal to the
  M12 lock prefixes) and hash-checked; each arm's `train.jsonl` is a hard link of the staged file.
- `m14_weights.py` verified on both nodes that each `base` block is exactly the first N lines (0.8B 162,696, 2B 56,141,
  4B 58,739), that its bytes hash to the released TRAIN (`f9f3c022…`, `1527b38b…`, `c385406e…`) and that no released
  line follows an IB line. Weights: released Choice / Noul / Score 1.5 / 1.5 / 1.5 (0.8B, 4B) and 1.5 / 1.5 / 2.0 (2B);
  every IB row 1.0. The shares equal the prereg's table.

## Teachers, references, panels (node B unless noted)

| Input | Source | Check |
| --- | --- | --- |
| 2B own-Sol teacher (`m14/inputs/teachers/2b/teacher.jsonl`) | node E `m11/data/2b/teacher.jsonl` | `3f92e8c108c5a9949fda4b08638907fcc80175f68efec18bece912d0edc8eea1` |
| 4B own-Lux teacher (`m14/inputs/teachers/4b/lux-teacher.jsonl`) | node E `m10/inputs/n4xf-teacher/m4-xl-full-29m/lux-teacher.jsonl` | `e2ff27ce2fc397c8c203e37133dcfac52ac3d1f172c6f3e9412ec28c2ffe296c` |
| DEV2.0-0.8B C0 package `bede7938…` (node A) | node E `/data/dev2/models/DEV2.0-0.8B/` | content manifest `26baab0177827efd…`, 97 files, equal on both sides |
| DEV2.0-2B C0 package `a53cf66a…` | node E `/data/dev2/models/DEV2.0-2B/` | manifest `ed89b9c4b4659748…`, 106 files, equal on both sides |
| LH soup (released LH weights) | node F `m10/soup/LH/build/LH-soup` | manifest `d5e0c085d911fca9…`, 13 files; **equal to node A's M10 package `checkpoint.files.sha256`** |
| `ib-dev` / `m10-probes` prompts (both nodes; gold-free) | node E `panels/` | `7cf53e47…` / `a324f1e2…` |

- Sources in each node's HF cache (revisions as M12): Eos 1.0 `363c4a5e`, Sol 1.0 `ce0c018a`, Qwen3.5-4B-Base
  `1001bb4d`. SELECT700 / CAL698 are node A / B's `m3/data-sel700-cal698` (`32a4352d…` / `19cc1a8c…`, equal to node E's
  `m10/inputs/sel700-cal698`). The other six panel prompt files on A / B equal node E's.
- Triton caches (`m14/triton-cache/<tier>-{train,read}`): `cp -a` copies of the node's `triton-cache/dbe5f32b2263`
  (node A 0.8B: 11,215 files; node B 2B / 4B: 30,182 files).
- Trainer tests (`test_dec`, `test_m12`–`test_m14`, 39 tests) pass in the decoder image on node B.

## READY

`m14/data/READY-m14.json` = `{"arms": {"08b-RAUP": {"train": "12bd63d8…", "weights": "99b77df5…"}, "2b-RAUP":
{"train": "08140409…", "weights": "e1ae6144…", "teacher": "3f92e8c1…"}, "4b-LHA10UP": {"train": "d41cdd1a…",
"weights": "3f741da8…", "teacher": "e2ff27ce…"}}, "record": "dec-m14-datalock-2026-10-01.md"}` (full hashes above).

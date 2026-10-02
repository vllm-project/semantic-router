# Decoder M17 — hand-over of the two 4B candidates to the Nox-4B publisher (2026-10-02 04:25Z)

Per the coordinator's interrupt of 12:25 UTC+8, M17 does **not** publish. The single Nox-4B publisher is the 4B
Index-first worker 5e7b8132 (branch `xunzhuo/decision-2-training-4b-indexfirst`), which compares these runs with M13's
and releases the highest. This record lists the integrity evidence with exact paths and hashes. It holds no Index
value; Index values stay in the node private directories.

Context: prereg `da770d98a`, amendment 1 (Index-first; measurement = the 4B worker's method) `367cdfa67`, data lock
`4f68f1eb0`, IX1 entries `db00429a1`, state [`dec-m17-state.md`](dec-m17-state.md).

## Candidates

| | `DEV2.0-4B-LHS10SD-bf16` | `DEV2.0-4B-LHS17SD-bf16` |
| --- | --- | --- |
| Arm (FP32 soup, node F) | `/data/dev2/runs/dec/m17/soup/4b-LHS10SD/build/4b-LHS10SD-soup`, model `537553da252d79e9fd4aca7731e74367cf15899788b80c912c3d4fc07a3f28ea` | `/data/dev2/runs/dec/m17/soup/4b-LHS17SD/build/4b-LHS17SD-soup`, model `8995bd9d58a3aab7c83baee47adfb1ce0e4782f51097b98a3d0a2247ddf14079` |
| TRAIN (data lock) | `m17/data/4b/4b-LHS10SD/train.jsonl` `72fa2d84…ef025` (66,004 rows) | `m17/data/4b/4b-LHS17SD/train.jsonl` `14bce13c…e25a0` (71,088 rows) |
| BF16 copy receipt (node F) | `/data/dev2/models/ix1/dec-m17/DEV2.0-4B-LHS10SD-bf16-bf16-copy.json` `c46e0448733a300bca12c678c5efbd8c7b4d750c61ccf60d1fe58fbbe9658eeb`; source = the soup's model SHA; BF16 model `d44a336a2d61918ac7b09b182649e12325673be2f358ac669c8ac3ad88239fdf` | `/data/dev2/models/ix1/dec-m17/DEV2.0-4B-LHS17SD-bf16-bf16-copy.json` `09dd39ed9b36f2b28bbe28936bce99c6a94313c44f2b01bc355e82e573550e49`; BF16 model `74ec8b2f838df1b8c26e94f83845f389e963da3a4a0576111a71cf8a617297d7` |
| BF16 checkpoint (node F) | `/data/dev2/models/ix1/dec-m17/DEV2.0-4B-LHS10SD-bf16-ckpt` | `/data/dev2/models/ix1/dec-m17/DEV2.0-4B-LHS17SD-bf16-ckpt` |
| Restaged package (nodes F and E, per-file lists equal) | `/data/dev2/models/ix1/dec-m17/DEV2.0-4B-LHS10SD-bf16-r13d42143`, `MODEL_MANIFEST.json` `171bc9295f02ed4fe98bd549b619003cb3416ccc9c8d7f4077bb539caf314132` (identity = BF16 model, loaded 4,208,383,488, calibration null) | `/data/dev2/models/ix1/dec-m17/DEV2.0-4B-LHS17SD-bf16-r13d42143`, `MODEL_MANIFEST.json` `024a02cc1b43b843785c758132c2d03ad10037c4276c4932cc4941ba87eb9982` (same checks) |
| IX1 parity gate (node E; 17SD's copied to node F with its frozen cache) | `/data/dev2/private/eval/index021/ix1/parity/DEV2.0-4B-LHS10SD-bf16/parity.json` `8fdb9811…d0c7c3`: PASS, 86 / 86 ok, max \|Δp\| 0.0 | `…/parity/DEV2.0-4B-LHS17SD-bf16/parity.json` `8fdb9811…d0c7c3` (byte-identical content): PASS, 86 / 86, 0.0 |
| Index run (node C, panel-8, mirror `d80933b6c` / `3e1b30664`) | `…/ix1/runs/DEV2.0-4B-LHS10SD-bf16/merged/`: 120,226 rows (120,224 ok, 2 unsupported), `results.jsonl` `cf3ddb991595fb5283d499c94b29c1af5a21a85329e29f12bc4c5a4b45124e4a`, receipt `ee73ab4e…`, compare `9cf4ebca…`, dual scoring PASS, panel run IDs `6455d7be…` (= LH's); 2.14 GPU-h (node E GPU0–2) | `…/ix1/runs/DEV2.0-4B-LHS17SD-bf16/merged/`: 120,226 rows (120,224 ok, 2 unsupported), `results.jsonl` `bb726d5d782ffd2e2fcd967b96eb34e4322a3978b72b11b2b48f17d281db3706`, receipt `c2c72651…`, compare `e5784876…`, dual scoring PASS, panel run IDs `6455d7be…`; 2.14 GPU-h (node F GPU2, 3, 6, 7) |
| Paired bootstrap vs `DEV2.0-4B-LH` (node C, CPU, 2,000 replicates, seed 20261002) | running since 04:09Z → `…/runs/DEV2.0-4B-LHS10SD-bf16/m17-boot-full-vs-lh.json`, exit file `…/ix1/logs/m17-boot-full-DEV2.0-4B-LHS10SD-bf16.exit` | running since 04:16Z → `…/runs/DEV2.0-4B-LHS17SD-bf16/m17-boot-full-vs-lh.json`, exit file `…/logs/m17-boot-full-DEV2.0-4B-LHS17SD-bf16.exit` |
| Formal typed-FINAL, item 3 (node A `formal/m17`) | `m17-4b-LHS10SD/TYPES.json` `bd01227e…27724e`: choice / Noul / Score OK / OK / OK; REPORT `fca321ab…` | `m17-4b-LHS17SD/TYPES.json` `8a65cdfa…0404b5`: OK / OK / OK; REPORT `a492bcfd…` |

- **Index contamination audit (node C, CPU; IX1 method, planted controls):** `/data/dev2/private/eval/index021/ix1/audit/m17/out/audit.json`
  `ad29695905a8ceb937bebda03bc7e10b5d4a6ac6df113b5f51fb9cb17dc310ff` (exit 0; also `items.json`, `duplicates.json`,
  `planted.jsonl`; inputs `audit/m17/train/FILES.txt`, the two TRAIN files hash-checked). 120,226 Index rows; planted
  200 / 200 found; **item rows 0 / 0**; duplicate-class rows 100 / 90.
- **Formal path:** parity run `m17-4b-LH` exact vs the stored LH run (0 / 0 / 0; TYPES `7feaf59b…`); successor files
  `formal/m17/successor/4b-m17-4b-LHS1{0,7}SD.{json,md}` (node A).
- **References the publisher should carry (not blockers under the 09:55 rule; material regressions to flag):** formal
  v3 vs LH −3.09 [−7.49, +0.93] / **−5.25 [−11.26, −1.67]**; mlx-diag card-eligible **−.0257 [−.0358, −.0161] /
  −.0279 [−.0395, −.0167]** (MLX-DEV2 −.023 / −.026); public 231 174 vs 172 (both); HT-DEV v2 −.010 TIE / **−.026
  FLAG**; retention −.048 (CI below 0) / −.026; typed DEV C / N / S 699 / 314 / 333 and 735 / 340 / 370 (LH 728 / 290
  / 371).
- GPU leases: node E GPU0–2 and node F GPU2, 3, 6, 7 are back to `track=dec-m17 status=released`; no M17 container runs.

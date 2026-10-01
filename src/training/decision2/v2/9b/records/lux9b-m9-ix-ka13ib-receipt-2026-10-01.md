# 9B M9 follow-up A: private Index diagnostic of the frozen K-a13IB — receipt (no Index values)

COORDINATION 2026-10-02 00:35, follow-up A. K-a13IB is the stage-3 soup (`model_sha256` `4701ba41…`; not a release).
Harness: IX1 (`v2/eval/ix1`, image host2 `f83b1d10`, kit `87d4650b`, panel-7). Tooling `lux9b/m9/ix.sh` and
`v2/eval/ix1/paired_boot.py` (`78b01441e`, mirrored to node C before use). Every Index value, per-benchmark value,
area value, frontier comparison and bootstrap interval is private: node C `/data/dev2/private/eval/index021/ix1/` and
the local private folder. This record holds procedure, counts, hashes and GPU-hours only. Label: **independent
provisional 0.2.1 reproduction**.

| Step | Result |
| --- | --- |
| Restage | the FP32 soup restaged into DEV2.0-9B `e51f9881` (the package IX1 scored; T = 1, calibration none) with `v2.eval.ix1.restage`: 16 files copied node A → node C (SHA-256 lists equal), loaded parameters 7,940,895,744, identity `4701ba41…`, manifest `ee09a792…` |
| FP32-restage control | DEV2.0-9B's own FP32 soup (`m4/K-a13-build/soup`) restaged the same way (manifest `9e7dbf79…`), read by the package's entry point over the 86 compatibility requests with DEV2.0-9B's frozen cache, against IX1's DEV2.0-9B kit parity results: **86 / 86 `ok`, identical choices, max \|Δp\| = 0.0** (`8fdb9811…`) |
| 86-request parity gate | **PASS**: 86 / 86 `ok`, identical choices, max \|Δp\| = 0.0 (`parity/K-a13IB/parity.json` `8fdb9811…`) |
| Full run | 7 shards on node C GPU1–7 (node C GPU0 never), DEV2.0-9B's frozen autotune cache; **120,226 rows: 120,224 `ok`, 2 `unsupported` (`max_length_exceeded`, as DEV2.0-9B), 0 errors**; results `df70173a…`, panel run IDs `6455d7be…` |
| Dual scoring | port vs kit `87d4650b`: **PASS** (`merged/compare.json` `d90eaf6c…`) |
| Paired bootstrap vs DEV2.0-9B | `paired_boot.py`: 2,000 replicates, seed `20261002`, 138,645 scoring cases resampled within each of the benchmarks, both runs scored on the same cases with the port; the identity resample reproduces both headlines exactly (`paired-boot-vs-dev20-9b.json` `92fd2fb0…`) |
| Contamination audit (CPU) | `v2.eval.ix1.contamination` against the K-a13IB TRAIN (`2cd09292…`, 151,015 lines): **0 item duplicates**; familiar text in ANLI (71 rows), BANKING77 (1) and HoVer (140), the same three benchmarks as DEV2.0-9B's own audit (88 / 1 / 175); planted control 200 / 200 |

GPU-hours (node C): parity 0.06, control 0.02, full run 2.74 (launcher receipt): **≈ 2.82 GPU-h**. Leases on node C
GPU1–7 were taken as `track=eval-ix1` and released at 17:52Z. Nothing went to HF or the gist.

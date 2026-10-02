# Decoder M18 state (prereg dec-m18-prereg-2026-10-02.md)

Index values stay private (node private dirs and the program's private folder); this file records operations only.

## 2026-10-02T05:10Z

- Part A: all eight BF16 copies built and restaged (identity, loaded-count and calibration checks pass). Index pools
  running: node A GPU1 / 2 / 7 (panel-3, the four 0.8B candidates), node E GPU0-3 / 6-7 (panel-8, 2b-RAUP-a75,
  2b-UPRA, 2b-RAUP-a50), node F GPU4-5 (panel-8, 2b-UPRAa75). Every parity gate so far 86 / 86.
- `m18-cand.sh`: 2B candidates are built and copied on node E (`WORK=e`), 0.8B on node A; node-to-node copies start
  on node A / B (their key is authorized on C-F only); the pool is started detached.
- Part B ops (`ops/m18`): `m18-prep.sh` (data on node F, in the decoder image), `m18_swap.py` (M17's builder),
  `m18-chains.sh` / `m18-arm.sh` / `m18-launch.sh` (node F GPU4-5, after the node's Index pool), `m18-soup.sh`,
  `m18_gpuh.py`. IB3-r2 TRAIN fetched to node F from the dataset revision `1c8452da` (hash matches `9d92d92a…`).
- Next: build the data, commit the data lock, write READY-m18.json, launch the chains.

## 2026-10-02T05:50Z

- Part A 2B: all four runs scored on node C (scorer gates pass); bootstraps vs DEV2.0-2B running on node C.
  `2b-UPRAa75`: one `invalid_model_output` row, scored with `--allow-errors` (amendment 1).
- Part A 0.8B: `08b-RAUP-a75`, `08b-UPRAa75` scored, bootstraps vs the current Eos release (`M16-08b-RA-a75`) running;
  `08b-UPRA` shards running on node A; `08b-RAUP-a50` dropped (amendment 1; node A pool stopped before its parity).
- Hub: Eos `main` = `3de61185` (b49d1f36's `08b-RA-a75` release, 04:44Z); Sol `main` still `8ed41433`.
- Part B: data lock committed; node F chains started 05:32Z (`2b-RS17UP` seeds 1 / 2, then `2b-RAUPM`).
- `m18-cand.sh` hop: per-item staging directory (two concurrent relays to one parent collided on `.m18-part`; the
  copy itself was complete and verified by SHA-256 list).
- Amendment 1 arms (`2b-RAM`, `08b-RAM`) on node A GPU1-2 / 7: inputs copied node F -> node A (pulled by node A).

## 2026-10-02T06:25Z

- Part A complete: 2B four runs scored and bootstrapped vs DEV2.0-2B (all exit 0); 0.8B three runs scored
  (`08b-UPRA` with one `invalid_model_output` row counted wrong, as `2b-UPRAa75`); no 0.8B Part A candidate is
  above the current Eos release, so their bootstraps vs it are report-only.
- Part B node F: `2b-RS17UP` seeds DONE (preflights PASS), soup built 06:03Z; `2b-RAUPM` seeds running.
  Node A: `2b-RAM` seeds 1 / 2 and `08b-RAM` seed 1 running (preflights PASS); `08b-RAM` seed 2 follows on GPU1.
- Candidates `2b-RS17UP` (soup) and `2b-SWRA` (½ RS17UP soup + ½ `2b-RA`, built on node E, lineage check PASS):
  BF16 copies restaged (identity / loaded / calibration checks pass); IX1 pool on node E GPU0-3 / 6-7 since 06:08Z
  (node E leases of the finished M18 runs set to released first).
- Bootstraps of `2b-UPRA` / `2b-UPRAa75` vs the `IS-2b-RAUP` run (the likely next Sol release) running on node C.

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

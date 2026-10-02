# Arm factory — state (running log, newest first; prereg `af-prereg-2026-10-02.md`)

Index values stay private (node private stores and `decision2-program/private/arm-factory/`); this file has none.

## 2026-10-02 14:20Z (22:20 UTC+8) — prereg and ops

- Prereg and ops (`v2/af/ops/`: `af-launch.sh`, `af-arm.sh`, `af-chain.sh`, `af_arms.py`, `af_gpuh.py`,
  `af_weights.py`) committed before any arm-factory GPU job.
- Free at 14:05Z: node A GPU1–6 (GPU7 runs M10's KIB4-a40 prerelease), node C GPU3–7. Node F GPU2–7 and node B
  GPU2 / 4 / 6 / 7 hold the M17 `SDMLxALL15` and M10 `KIB4P-a33` Index runs.
- Assets relayed node to node: M17's 4B locks, data, Triton caches and Qwen3.5-4B-Base (node F → node C through
  node A); M10's KIB4 TRAIN / teacher and IB4 p1 (node B → node C → node A).

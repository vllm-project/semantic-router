# Arm factory — state (running log, newest first; prereg `af-prereg-2026-10-02.md`)

Index values stay private (node private stores and `decision2-program/private/arm-factory/`); this file has none.

## 2026-10-02 14:35Z (22:35 UTC+8) — 15 GPUs training

| Node / GPU | Chain (items in order) | Since |
| --- | --- | --- |
| A1 | `KIB4` s4 (seed 3; pre-warm, preflight PASS 14:29Z) | 14:23Z |
| A2 / A3 / A4 / A5 / A6 | `KIB4` s5 (4); `KIB4W2` s1 (5) / s2 (6); `KIB4L2` s1 (7) / s2 (8) | 14:30Z |
| C3 | `4b-LHS17IB4` s4 (20260929; pre-warm, PASS 14:27Z), then `4b-LHS17IB4-lrh` s1 (20260926) | 14:22Z |
| C4 | `4b-LHS17IB4` s5 (20260930), then `4b-LHS17IB4-lrh` s2 (20260927) | 14:28Z |
| C5 / C6 / C7 | `4b-SDMLIB4` s4 (20260929) then `4b-LHS17UP` s4 (20260929); `4b-SDMLIB4` s5 (20260930); `4b-LHS17UP` s3 (20260928) | 14:28Z |
| F2 / F3 / F4 / F5 | `4b-LHS17IB4ML` s1 (pre-warm) / s2; `4b-LHS23IB4` s1 / s2 (amendment 1) | 14:33Z |

- Data locks (`READY-af.json` per node): wave-1 4B entries equal M17's locks; `KIB4` equals M10's node B lock
  (`2e72bcfd…` / `377f8878…`); `KIB4W2` adds weights `ea81c621…` (9,459 IB4 rows ×2; IB4 loss-weight share .063 →
  .118); wave-2 4B files per amendment 1 (audit PASS).
- Node F GPU2–5: stale harness leases of M17's finished `SDMLxALL15` run moved to `owner.prev-af-*`.

## 2026-10-02 14:20Z (22:20 UTC+8) — prereg and ops

- Prereg and ops (`v2/af/ops/`: `af-launch.sh`, `af-arm.sh`, `af-chain.sh`, `af_arms.py`, `af_gpuh.py`,
  `af_weights.py`) committed before any arm-factory GPU job.
- Free at 14:05Z: node A GPU1–6 (GPU7 runs M10's KIB4-a40 prerelease), node C GPU3–7. Node F GPU2–7 and node B
  GPU2 / 4 / 6 / 7 hold the M17 `SDMLxALL15` and M10 `KIB4P-a33` Index runs.
- Assets relayed node to node: M17's 4B locks, data, Triton caches and Qwen3.5-4B-Base (node F → node C through
  node A); M10's KIB4 TRAIN / teacher and IB4 p1 (node B → node C → node A).

# ~27B M4 state (resume file)

Updated: 2026-09-29 16:50 UTC+8 (M4 worker; **preregistered, launching**)
Branch: `xunzhuo/decision-2-training-27b` (merge-only into `xunzhuo/decision-2-training`)
Prereg: `m4-prereg-2026-09-29.md`. Code mirror on both nodes: `1b56e482b` (drivers in `v2/27b/m4/`, run from
the mirror; no uploads).

## Done

- Frozen mixtures (node B `/data/dev2/private/27b/m4-data/mixtures-m4-1` = `-2`; node A copy `mixtures-m4-1`):
  `a20.train.jsonl` `4aa0dc96…` (56,969 rows, 25,043,392 tokens, 3,561 updates), `ar.train.jsonl` `aadeef1a…`
  (64,363 rows, 25,043,494 tokens, 4,023 updates). `MIXTURES.json` `4839c27e…`.
- T0 training cache `/data/dev2/runs/27b/m4-train-cache-T0` (`1933eb36…`, read-only) on both nodes.
- Base tree `c457c994…` equal on both nodes. C1 source check clean (`m4-logs/c1-sources-{a20,ar}.json`, node B).
- Leases: node B GPU5–7 and node A GPU2–4 owned by 27b (M4), previous owner files kept as `owner.prev-*`.

## Placement (one attempt each)

| Run | Node / GPU | Mixture | LoRA |
| --- | --- | --- | --- |
| M4-A20r-s1 / -s2 | B5 / B6 | a20 | r32 α64 |
| M4-A20-s1 | B7 | a20 | r8 α16 |
| M4-xnode-A20-s1 (onestep probe), then M4-A20-s2 | A2 | a20 | r8 α16 |
| M4-Ar-s1 / -s2 | A3 / A4 | ar | r8 α16 |

## How to operate

- Launch (node host): `bash /data/dev2/src/1b56e482bf1a3c5c14bd74ff30acf97ee302a99f-src_training_decision2/src/training/decision2/v2/27b/m4/m4-launch-arm.sh NODE ARM GPU MIX SEED`.
- Status: same dir, `m4-status.sh NODE` (JSON per arm-seed); heartbeat loop `m4-status.sh NODE --loop 600`
  → `/data/dev2/runs/27b/m4-logs/heartbeat-<node>.json`.
- Liveness: `docker ps --filter name=d2-27b-M4-` (never `pgrep -f`).
- Run dirs: `/data/dev2/runs/27b/M4-*` on the node that trains them; driver log `driver.log`, receipts `receipts/`.

## Next

1. Launch the five node-B/node-A arm-seeds and the probe; verify containers and first log lines.
2. G1 budget check after ~300 full updates of every run.
3. As node A runs finish: relay BEST checkpoints (no `trainer_state.pt`) to node B, check SHA-256.
4. Soups → readouts (sanity) → CAL698 / 23:15 → packages → formal → gates / overlap → mlx-diag → contrasts.

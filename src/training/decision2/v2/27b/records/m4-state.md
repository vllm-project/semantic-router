# ~27B M4 state (resume file)

Updated: 2026-09-29 18:25 UTC+8 (M4 worker; **all six arm-seeds in full training; G1 passed; amendment 1**)
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

## Launch receipts (prereg `727de1149`)

- Launched 08:44Z (A20-s2 08:52Z, after the probe). Every driver's first log line, T0 copy check (`1933eb36…`) and
  pipeline check are in its `driver.log`; containers confirmed by name.
- Preflights, all six: admission all admitted; one-step exit 0 (0.07–0.08 GPU-h); reload 32 rows, 0 argmax
  changes, passed. Full containers `d2-27b-M4-<ARM>-full` running on B5/B6/B7 and A2/A3/A4 by 08:58Z.
- **Cross-node probe** `M4-xnode-A20-s1` (node A GPU2, 0.071 GPU-h): step-1 adapter `628b7825…` and head
  `02c858ad…` are byte-identical to node B's M4-A20-s1 one-step; loss 1.6794943176209927 and gradient norm
  13.559420585632324 are equal. Training is node-invariant on this image / base / T0.
- Heartbeats: `m4-status.sh {a,b} --loop 600` running on both nodes (pids in `m4-logs/heartbeat-*.driver.log`).

## G1 and amendment 1 (18:20 UTC+8)

- G1 at ~350 updates: projected full attempts 10.25 / 10.30 (A20), 10.59 / 10.35 (A20r), 11.07 / 11.48 (Ar);
  training ≈ 64.7 + evaluation ≈ 2.95 ≈ 67.6 of 70 → continue.
- Amendment 1: post-training drivers (`32718523d`, `d815066f9`), group-level family CIs, CAL698 not optional,
  typed-DEV collapse check refined (chance test only where F1 ≥ chance + .10; modal share ≥ .95), `M4_LEFT` from
  `m4-budget.sh` gates every tail GPU stage.
- Tail tools (workstation): `v2/27b/m4/m4-budget.sh` (prints `M4_LEFT`), `m4-relay-best.sh ARM` (node A BEST → node B),
  `m4-mlx-score.sh NAME...`. Node B: `m4-tail.sh {soup|readout|guard|prepkg|formal|gates|overlap|mlx|contrast}` from
  the amendment-1 mirror (see below).

## Next

1. Monitor (heartbeats); training ends ≈ 19:10–20:30Z.
3. As node A runs finish: relay BEST checkpoints (no `trainer_state.pt`) to node B, check SHA-256.
4. Soups → readouts (sanity) → CAL698 / 23:15 → packages → formal → gates / overlap → mlx-diag → contrasts.

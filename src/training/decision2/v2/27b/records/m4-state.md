# ~27B M4 state (resume file)

Updated: 2026-09-30 06:40 UTC+8 (M4 worker; **M4 complete: successor M4-A20r soup**)
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

- Node B tail mirror: `ff660322a76b04c4774fdb6aaa31e5af2a45743e` (content manifest `edb7f5f8…`, 3,640 files).
- **Relay watcher (workstation, started 09:56Z):** `~/.cache/m4-work/relay-watch.sh` (pid in `pgrep -af relay-watch`)
  runs the committed `m4-relay-best.sh` (SHA-256 `7bdb8167…` = `ff660322a`) for M4-A20-s2, M4-Ar-s1, M4-Ar-s2 in
  order, retrying every 5 min until each run is complete; log `~/.cache/m4-work/relay-watch.log` and
  `relay-<ARM>.log`. If it is not running, restart it the same way (the relay is idempotent).

## Resume after the usage-limit stop (20:45 UTC+8)

- Runs untouched and healthy (5.8 h into full training at 14:37Z; projected full attempts 9.9–11.1).
- Amendment 2 (`53685d65e`): the 16:05 seven-item successor rule vs DEV2.0-27B (F1's scored run), the beats-AutoJev
  check, and JevBench item 7 (`gates public231`, no selection on it). Merged integration → `44a94e581`, mirrored to
  both nodes (use it for `gates public231` on node B and `v2.06b.m8_scorebias mlx-paired` on node A).

## Training done (20:00Z)

- All six full attempts: exit 0, no watchdog, all planned updates, 0 recoveries, training cache frozen_check passed
  (0 added / 0 changed). Full GPU-h: A20 9.998 / 10.029, A20r 9.994 / 10.076, Ar 10.974 / 11.081. Receipts total
  63.29 GPU-h (incl. preflights, probe, 2 soup readouts).
- BEST: A20-s1 1784, A20-s2 2676, A20r-s1 3561, A20r-s2 2230, Ar-s1 3018, Ar-s2 3018.
- Relays (workstation → node B), SHA-256 lists equal: A20-s2 (19:02Z, 276.8 MB), Ar-s1, Ar-s2 (20:00Z).
- Soups exact: A20r r64 α128 (max rel 5.2e-7), A20 r16 α32 (2.5e-7); Ar building.
- Soup readouts (sanity only): A20r P 78.99 (T .916, H .681; Noul 270/400), A20 P 76.30 (T .876, H .665; Noul 202/400).
- Node B chain `m4-logs/m4-chain-formal.sh` (uploaded, size+SHA checked, pid confirmed 20:28Z): waits for the Ar
  readout, runs `guard`, then per finalist `prepkg` + `formal` (A20r GPU5, A20 GPU6, Ar GPU7); logs
  `m4-logs/chain-formal.log`, `final-<ARM>.log`.

## Milestone 4 complete (2026-09-30 06:40 UTC+8)

- Results: `m4-results-2026-09-29.md`. **Successor = M4-A20r soup** (post-key v3 72.360, +5.15 (+2.19, +8.02) vs
  DEV2.0-27B; all seven 16:05 items pass; tie-break over M4-A20). Not "beats AutoJev" (LB −1.60).
- Package node B `/data/dev2/runs/27b/M4-A20r-soup/package` (T = 1, 32K), SEAL `9d60e611…`. Nothing on HF.
- GPU-h 65.68 of 70. No chain left running; heartbeat loops ended on their own; leases set back to reserved-idle (node A GPU2–4, node B GPU5–7).

## Next (coordinator)

- Release step for the successor as a new DEV2.0-27B revision (card disclosures listed in the results record).

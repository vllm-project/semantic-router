# ~27B M5 state (resume file)

Updated: 2026-09-30 10:00 UTC+8 (01:57Z; M5 worker). **Continuation workers: read "Next steps" first.**
Branch: `xunzhuo/decision-2-training-27b` (worktree `/home/xunliu/code/vllm-sr-dev2-27b`; merge-only into
`xunzhuo/decision-2-training`). Gist file: `06-decision-2-27b.md`. Assignment: COORDINATION 2026-09-30 07:20.
Prereg: `m5-prereg-2026-09-30.md` (+ amendment 1). Latest node mirror (both nodes): **`e76e56d4c`**.

## Target, reference, budget

- **Goal:** post-key v3 paired lower bound > 0 vs AutoJev-27B (72.133), human transfer not significantly below.
- **Successor reference:** A20r, released 09:50 as DEV2.0-27B `main` @ `53233103` (same weights as the scored run
  node B `/data/dev2/runs/27b/M4-A20r-soup/formal`). Its C1 post-key run (57.56) is now the 27B C1 baseline; another
  27B C1 attempt needs coordinator approval (one per baseline).
- **Coordinator hint (09:50, not a rule change):** among passing finalists prefer an HT-DEV v2 GAIN / human-transfer
  gain; typed gains inflate v3 without moving C1. Report it with the choice; the preregistered choice rule stands.
- **Budget:** 72 GPU-h. Receipts at 00:52Z: 1.54 (+ running training). G1 projection ≈ 53.5. Sum receipts with
  `ssh <node> python3 - < ~/.cache/m5-work/budget.py` on both nodes (workstation helper).
- **Platform rule (COORDINATION 07:35):** poll ≤ 30 min with one-line state updates; hand off through this file near
  5 h of turn time.

## Running now (all detached; liveness by container name or PID)

| What | Where | Log / PID | ETA |
| --- | --- | --- | --- |
| Lane B: FF20-s1 full → FF20H-s1 | node B GPU5–7, `d2-27b-m5-FF20-s1-full` | `/data/dev2/runs/27b/m5/logs/lane-b.log` | FF20-s1 ≈ 03:40Z, FF20H-s1 ≈ 08:10Z |
| Lane A: FF20-s2 full → FF20H-s2 | node A GPU2–4, `d2-27b-m5-FF20-s2-full` | node A `/data/dev2/runs/27b/m5/logs/lane-a.log` | ≈ 03:55Z, ≈ 08:25Z |
| Chain FF20: pull FF20-s2 → soup M5-FF20 → readout (GPU0) → devgates | node B | `logs/chain-FF20.log`, PID 2425535 | ≈ 04:50Z |
| Chain FF20H: pull FF20H-s2 → soups M5-FF20H + M5-SX → readouts (GPU0, GPU1) → devgates (all three) | node B | `logs/chain-FF20H.log`, PID 2432764 | ≈ 09:30Z |

- Leases: node B GPU5–7 and GPU0–1 (aux) and node A GPU2–4 are track `27b`.
- Status: `~/.cache/m5-work/status.sh` (updates, s/update, projection per running attempt).

## Infrastructure

- **Private node link (node B → node A):** temporary ed25519 key on node B `/data/dev2/tmp/27b-m5-xfer/` (mode 700;
  `peer` holds node A's private address), authorized on node A for rsync only (`from=<node B private address>,
  restrict,command="/usr/bin/rrsync /data/dev2/xfer/27b-m5"`, comment `dev2-27b-m5-xfer-temp`; node A's previous
  `authorized_keys` backed up as `authorized_keys.bak.27b-m5-<UTC>`). ≈ 850 MB/s. rsync paths are relative to node A's
  `/data/dev2/xfer/27b-m5`. **At milestone end remove the key line on node A and `/data/dev2/tmp/27b-m5-xfer` on node B.**
- **Mirrors:** `~/.cache/m5-work/mirror2.sh <commit>` (node A by stream, node B over the link, then re-verified).
- **Staged on node A** (`/data/dev2/xfer/27b-m5/stage/`, hash-equal to node B): A20r soup checkpoint + package; F-b
  checkpoint (96 GB FP32) + package.
- **ht-dev2 installed on node B** (`/data/dev2/private/panels/{goldfree,gold}/ht-dev2.*`, registry hashes verified).
- **Mixtures** (built twice on node B, identical; `MIXTURES.json` `9ad3da72…`): `a20` `4aa0dc96…` (= M4), `a20h`
  `4a9d93f5…` (71,753 rows, 33,033,018 tokens); node A copy hash-equal at the same path. C1 source check on `a20h`:
  clean (`m5-logs/c1-sources-a20h.json` `53af68da…`, node B).

## Done

- **Step 1, HT-DEV v2 references** (node A; 0.486 GPU-h): F1 .5779 (+.0125 TIE vs A20r), **A20r .5655**, F-b .5527
  (−.0128 TIE), Eikos .5558 (TIE), AutoJev .5443 (−.0211 FLAG). Outputs node A `/data/dev2/runs/27b/m5/htdev2/<key>/`.
- **A20r node B reference readout** (0.249 GPU-h; `readouts/m4-a20r-soup`, symlink `readouts/A20r-ref`): P_dev 78.99,
  T_dev .91625, H_pilot .6811, H_dev2 .56547; HT-DEV v2 answers equal node A's exactly.
- **Prereg** `b31cd76e9`; amendment 1 (`7f07c4904`: launch3 co-tenant statuses; first launch refused before any
  container, 0 GPU-h). Preflights all passed: FF20-s1 one-step 0.251 + reload 0.025 (0 / 32 argmax changes); FF20-s2
  0.225 + 0.026; FF20X-s1 probe 0.278 (update-1 loss and gradient norm equal on both nodes; parameters differ bytewise
  in 25 / 27 backbone shards — float order, report only).
- **G1 (00:52Z):** continue as planned (FF20 ≈ 3.5 h per attempt, FF20H ≈ 4.3 h).

## Next steps (in order)

1. **When `chain-FF20.log` ends with devgates** (`/data/dev2/runs/27b/m5/readouts/DEVGATES-<UTC>.json`): apply
   **B1** to M5-FF20's gates 1–3 (collapse, HT-DEV v2 not FLAG vs A20r, typed guard).
   - **Pass (default):** launch M5-FF20's formal run on node B GPU0:
     `bash /data/dev2/src/<mirror>-src_training_decision2/src/training/decision2/v2/27b/m5/m5-tail.sh formal <mirror> M5-FF20 /data/dev2/runs/27b/m5/M5-FF20/checkpoint 0`
     (detached with setsid/nohup, log in `logs/`). FF20H continues.
   - **Fail:** `echo "B1: <reason>" > /data/dev2/runs/27b/m5/STOP-FF20H`, same text to `BRANCH-B1`; `docker stop` the
     running `d2-27b-m5-FF20H-s*-full` containers (the lanes then stop); launch L128:
     `m5-l128.sh b 5 s1` (node B) and `m5-l128.sh a 2 s2` (node A) from mirror `e76e56d4c`. The L128 soup / readout /
     formal use M4's LoRA tooling (`v2/27b/run_finalist.sh`, `m4/m4-tail.sh` pattern) — write an M5 wrapper then.
2. After M5-FF20's formal: mlx-diag (`m5-tail.sh lease <mirror> 2`, `m5-tail.sh mlx <mirror> M5-FF20 2`, then
   `m5-tail.sh mlx-push <mirror> M5-FF20`; on node A `m5-mlx-nodeA.sh <mirror> M5-FF20`; then `m5-tail.sh mlx-pull
   <mirror> M5-FF20`), and gates on node B: `m5-gates.sh <mirror> gates M5-FF20`, `... overlap`, `... verdicts M5-FF20`.
3. When `chain-FF20H.log` ends with devgates: formal (≤ 3 finalists total, every passer), mlx-diag and gates for
   M5-FF20H and M5-SX as in step 2 (use `EXTRA_COMPARATOR` for the other finalists' formal runs).
4. Verdicts → results record `m5-results-2026-09-30.md`, gist 06, merge, report. A successor needs a release hand-off:
   stage its frozen package on node A over the link (full weights: 96 GB FP32; the release worker converts with
   `bf16_copy` + `bf16z`), write a C1 post-key successor spec (format: `v2/eval/sealed/c1-postkey/*.json`, role
   successor, 27B kernel adapter, the formal run's post-run cache and stored typed-final / public231 predictions), and
   note the privacy-screen rule: soup `decision_config.json` records node paths; the release must write track-relative
   paths.
5. Cleanup at the end: remove the temporary key (see Infrastructure), set leases to reserved-idle, and delete non-BEST
   full checkpoints of the FF runs only after the results are recorded (keep every BEST and soup).

## Poll log (newest first)

- 02:53Z: FF20-s1 701/891, FF20-s2 672/891 (≈ 13 s/upd); chains waiting; no incident.
- 02:27Z: FF20-s1 593/891 (12.4 s/upd), FF20-s2 553/891 (12.3 s/upd); both chains waiting; no incident.

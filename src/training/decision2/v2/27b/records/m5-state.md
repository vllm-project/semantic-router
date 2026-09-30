# ~27B M5 state (resume file)

Updated: 2026-09-30 17:20 UTC+8 (09:20Z; M5 continuation worker, started 09:10Z; COORDINATION 17:15). **Continuation
workers: read "Next steps" first.**

**Deviation (prereg amendment 2):** M5-FF20 failed development gate 2 at 04:26:53Z (HT-DEV v2 .516 vs A20r .565, Δ −.049
[−.070, −.029], FLAG; gate 4 failed too), so B1 triggered. No worker was alive, so FF20H was not stopped. FF20H trained
to completion and M5-SX was built. Both go through their preregistered gates as usual, and the results must disclose the
deviation. B1's L128 branch was launched late, at 09:16Z.
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
| L128-s1 (admit → onestep → reload → full; cap 12.0 GPU-h) | node B GPU5, `d2-27b-M5-L128-s1-*` | `/data/dev2/runs/27b/M5-L128-s1/driver.log`, PID 2459358 | ≈ 19:40Z |
| L128-s2 (same) | node A GPU2, `d2-27b-M5-L128-s2-*` | node A `/data/dev2/runs/27b/M5-L128-s2/driver.log`, PID 3882884 | ≈ 19:40Z |
| Chain FF20H: readouts M5-FF20H (GPU0) + M5-SX (GPU1) → devgates (M5-FF20, M5-FF20H, M5-SX) | node B | `logs/chain-FF20H.log`, PID 2432764 | ≈ 09:35Z |

- Done lanes and chains: FF20H-s1 (node B, 13.32 GPU-h), FF20H-s2 (node A, 13.49), chain FF20 (devgates 04:26Z).
- `BRANCH-B1` is on both nodes (`/data/dev2/runs/27b/m5/BRANCH-B1`). `STOP-FF20H` was not written because there was
  nothing left to stop.
- Leases: node B GPU0–1 (aux, running readouts) and GPU5 (L128-s1), node A GPU2 (L128-s2) are track `27b`. Node B
  GPU6–7 plus GPU2 are lent to 2B / 0.8B M8-small, and node A GPU3–4 to 9B M8 (COORDINATION 17:05 / 17:15). Never
  co-tenant them.
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
- **FF20 trained (both seeds exit 0, BEST = final update 891):** FF20-s1 full 10.255 GPU-h, SELECT700 family macro .9033; FF20-s2 (node A) full 10.468 GPU-h, .8693. FF20H-s1 preflights passed (reload 0.025). The FF20 chain started pulling FF20-s2 at 03:52Z.

## Next steps (in order)

1. **B1 is done** (amendment 2): M5-FF20 is not a finalist, and L128 is training (see "Running now").
2. **When `chain-FF20H.log` ends with devgates** (`readouts/DEVGATES-<UTC>.json` over M5-FF20, M5-FF20H and M5-SX):
   each passer goes formal (≤ 3 finalists in total, L128 included) on node B GPU0 or GPU1:
   `m5-tail.sh formal <mirror> NAME /data/dev2/runs/27b/m5/NAME/checkpoint GPU` (detached, log in `logs/`), with
   `EXTRA_COMPARATOR` naming the other finalists' formal runs. Check the budget first (amendment 2). Then mlx-diag
   (`m5-tail.sh mlx <mirror> NAME GPU`, then `mlx-push`; on node A `m5-mlx-nodeA.sh <mirror> NAME`; then `mlx-pull`)
   and gates on node B (`m5-gates.sh <mirror> gates NAME`, `... overlap`, `... verdicts NAME...`).
3. **L128** (both seeds ≈ 19:40Z): relay L128-s2's BEST from node A, pull it, build the soup `M5-L128` (exact rank-256
   concatenation), read it out on GPU0 / GPU1 and run devgates. If it passes, formal, mlx-diag and gates as in step 2.
   The tooling is being written now; its commands will appear here.
4. Verdicts → results record `m5-results-2026-09-30.md`, gist 06, merge, report. A successor needs a release hand-off:
   stage its frozen package on node A over the link (full weights: 96 GB FP32; the release worker converts with
   `bf16_copy` + `bf16z`), write a C1 post-key successor spec (format: `v2/eval/sealed/c1-postkey/*.json`, role
   successor, 27B kernel adapter, the formal run's post-run cache and stored typed-final / public231 predictions), and
   note the privacy-screen rule: soup `decision_config.json` records node paths; the release must write track-relative
   paths.
5. Cleanup at the end: remove the temporary key (see Infrastructure), set leases to reserved-idle, and delete non-BEST
   full checkpoints of the FF runs only after the results are recorded (keep every BEST and soup).

## Poll log (newest first)

- 09:20Z: continuation worker. B1 recorded late (amendment 2). Leases node B GPU5 / node A GPU2 retaken 09:13Z. `BRANCH-B1` written; L128-s1 / s2 launched 09:16Z (admit). M5-FF20H / M5-SX CAL fits done, readout collections running. Receipts 49.46 GPU-h.
- 04:00Z: FF20 both seeds complete (BEST 891; SELECT .903 / .869; 10.25 / 10.47 GPU-h); FF20H-s1 full running, FF20H-s2 preflights; chain-FF20 pulling FF20-s2.
- 03:19Z: FF20-s1 831/891, FF20-s2 795/891; both ≈ 3.45 h per attempt (cap 4.5); chains waiting.
- 02:53Z: FF20-s1 701/891, FF20-s2 672/891 (≈ 13 s/upd); chains waiting; no incident.
- 02:27Z: FF20-s1 593/891 (12.4 s/upd), FF20-s2 553/891 (12.3 s/upd); both chains waiting; no incident.

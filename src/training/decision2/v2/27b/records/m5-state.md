# ~27B M5 state (resume file)

Updated: 2026-09-30 17:20 UTC+8 (09:20Z; M5 continuation worker, started 09:10Z; COORDINATION 17:15). **Continuation
workers: read "Next steps" first.**

**Deviation (prereg amendment 2):** M5-FF20 failed development gate 2 at 04:26:53Z (HT-DEV v2 .516 vs A20r .565, Δ −.049
[−.070, −.029], FLAG; gate 4 failed too), so B1 triggered. No worker was alive, so FF20H was not stopped. FF20H trained
to completion and M5-SX was built. Both go through their preregistered gates as usual, and the results must disclose the
deviation. B1's L128 branch was launched late, at 09:16Z.
Branch: `xunzhuo/decision-2-training-27b` (worktree `/home/xunliu/code/vllm-sr-dev2-27b`; merge-only into
`xunzhuo/decision-2-training`). Gist file: `06-decision-2-27b.md`. Assignment: COORDINATION 2026-09-30 07:20.
Prereg: `m5-prereg-2026-09-30.md` (+ amendments 1–2). Latest node mirror (both nodes): **`c8b1d9128`** (L128 tail;
the L128 training runs from `e76e56d4c`).

## Target, reference, budget

- **Goal:** post-key v3 paired lower bound > 0 vs AutoJev-27B (72.133), human transfer not significantly below.
- **Successor reference:** A20r, released 09:50 as DEV2.0-27B `main` @ `53233103` (same weights as the scored run
  node B `/data/dev2/runs/27b/M4-A20r-soup/formal`). Its C1 post-key run (57.56) is now the 27B C1 baseline; another
  27B C1 attempt needs coordinator approval (one per baseline).
- **Coordinator hint (09:50, not a rule change):** among passing finalists prefer an HT-DEV v2 GAIN / human-transfer
  gain; typed gains inflate v3 without moving C1. Report it with the choice; the preregistered choice rule stands.
- **Budget:** 72 GPU-h. Receipts at 09:50Z: **50.05** (node B 25.00, node A 25.06, including the L128 preflights).
  Projection: ≈ 70.5 after L128's training, ≈ 70.8 after its readout, and ≈ 71.7 if it also goes formal (+ mlx-diag).
  The chain checks receipts + 0.9 ≤ 72 before the formal run. Sum receipts with
  `ssh <node> python3 - < ~/.cache/m5-work/budget.py` on both nodes (workstation helper; now includes `M5-L128-s*`).
- **Platform rule (COORDINATION 07:35):** poll ≤ 30 min with one-line state updates; hand off through this file near
  5 h of turn time.

## Running now (all detached; liveness by container name or PID)

| What | Where | Log / PID | ETA |
| --- | --- | --- | --- |
| L128-s1 full (preflights passed; 3,561 updates, ≈ 10.1 s/upd; cap 12.0 GPU-h) | node B GPU5, `d2-27b-M5-L128-s1-full` | `/data/dev2/runs/27b/M5-L128-s1/driver.log`, PID 2459358 | ≈ 19:30Z |
| L128-s2 full (same, ≈ 10.4 s/upd) | node A GPU2, `d2-27b-M5-L128-s2-full` | node A `/data/dev2/runs/27b/M5-L128-s2/driver.log`, PID 3882884 | ≈ 19:50Z |
| Relay watcher: L128-s2 BEST → node A `xfer/27b-m5/relay/M5-L128-s2` (+ `BUDGET-nodeA.json`) | node A | node A `logs/relay-M5-L128-s2.log`, PID 3889492 | at s2's end |
| mlx watcher: scores M5-L128's mlx-diag after `mlx/M5-L128.PUSHED` (ends on `.SKIP`) | node A | node A `logs/mlx-watch-M5-L128.log`, PID 3889493 | if a finalist |
| **Chain L128** (`RESERVE=0.9`): pull → `lsoup` M5-L128 → readout (GPU0) → devgates → [formal → mlx-diag → push → pull → gates / overlap / verdicts] | node B | `logs/chain-L128.log`, PID 2495533 | ≈ 20:30Z (devgates); ≈ 22:00Z (verdicts) |

- Done lanes and chains: FF20H-s1 (node B, 13.32 GPU-h), FF20H-s2 (node A, 13.49), chain FF20 (devgates 04:26Z),
  chain FF20H (devgates 09:25:43Z, no finalist).
- The chain was first started with `RESERVE=1.0` and restarted at 09:49Z with 0.9 while it was still waiting (log
  `logs/chain-L128.reserve1.0-stopped.log`). 0.9 is the prereg's planned evaluation per finalist: formal
  2.2 / 3 + mlx-diag 0.5 / 3.
- `lsoup` was tested on M4-A20r's two seeds at 09:39–09:47Z (CPU). It reproduced M4-A20r-soup's model hash
  `2e074511…` exactly, and the relay-list check ran. The test output was deleted; the log is
  `logs/test-lsoup-A20r.log`.
- `BRANCH-B1` is on both nodes (`/data/dev2/runs/27b/m5/BRANCH-B1`). `STOP-FF20H` was not written because there was
  nothing left to stop.
- Leases: node B GPU0 (reserved-idle, held for chain L128), GPU1 (reserved-idle, spare) and GPU5 (L128-s1), node A
  GPU2 (L128-s2) are track `27b`. Node B
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
- **FF20H trained** (the B1 deviation): FF20H-s1 full 13.073 GPU-h (node B), FF20H-s2 full 13.216 (node A). Soups
  M5-FF20H (`1e5b6e8a…`) and M5-SX (`9d853b03…`, four members) verified to 6e-8.
- **Development gates: no FF finalist.** Files are `readouts/DEVGATES-20260930T042653Z.json` (M5-FF20) and
  `DEVGATES-20260930T092543Z.json` (all three). Every candidate is HT-DEV v2 FLAG vs A20r (.5655):

  | Candidate | T_dev | H_pilot | P_dev | H_dev2 (Δ [95%]) | Gates failed |
  | --- | ---: | ---: | ---: | --- | --- |
  | M5-FF20 | .9569 | .4970 | 68.96 | .5161 (−.0494 [−.0703, −.0291]) | 2, 4 |
  | M5-FF20H | .8538 | .4998 | 65.33 | .5203 (−.0452 [−.0659, −.0239]) | 1 (typed-DEV Noul .49, 99% "False"), 2, 3, 4 |
  | M5-SX | .9281 | .5252 | 69.82 | .5257 (−.0397 [−.0596, −.0202]) | 2, 4 |

  A20r reference: T_dev .9163, H_pilot .6811, P_dev 78.99. Full fine-tuning with more typed dose lowers human
  transfer on every screen. Round-2 data (HS1 / PN1-r2) collapses FF20H's rule-precedence Noul to "False".
- **L128 preflights passed** on both nodes: s1 one-step and reload exit 0; s2 one-step 0.068 and reload 0.019 GPU-h.
  The full runs have been going since ≈ 09:22Z, at ≈ 9.7 s per update (3,561 updates).
- **L128 tooling** (`b2af94a9e` drivers, `c8b1d9128` M5 tail; mirror `c8b1d9128` on both nodes). M4b's `run_readout.sh`
  and `run_formal.sh` take `CHECKPOINT_FORMAT=peft-lora/1` (default `full` unchanged; `m4b/ckpt_format.py`). The
  L128 soup loads 27,497,508,864 parameters: base text 25,624,600,064 + rank-256 adapter 1,867,644,928 + head
  5,263,872. The helper reproduces A20r's released 26,096,775,168. `m5-tail.sh lsoup` does the exact rank
  concatenation. `m5-l128-relay.sh` / `m5-mlx-watch.sh` run on node A, and `m5-l128-chain.sh` on node B.

## Next steps (in order)

1. **B1 is done** (amendment 2): M5-FF20 is not a finalist, and L128 is training (see "Running now").
2. **FF20H / SX gates are done: no FF finalist** (see "Done"). No FF formal run is made.
3. **L128 is automated end to end; just poll.** Once L128-s2 finishes, node A's relay watcher relays it. Node B's
   `chain-L128.log` then pulls it, builds `M5-L128`, reads it out on GPU0 and runs devgates over all four candidates.
   If L128 is a finalist and the budget allows, formal, mlx-diag and gates follow; node A's mlx watcher scores the
   mlx-diag. How the chain can end:
   - **`not a finalist: …`** (exit 0): there is no finalist in M5. Record the attribution only (prereg "No finalist"),
     then do steps 4–5.
   - **`m5 l128 chain complete`** after `m5 gates verdicts`: read `gates/VERDICTS-<UTC>.json` (items 1–7,
     beats-AutoJev, choice) and do step 4.
   - **exit 3** (a seed ended without a finished run) or **exit 4** (the budget would pass 72): record it, and the
     coordinator decides. Nothing reruns.
   - Any other failure: fix the code, mirror it, and re-run only the missing stages by hand. The chain skips finished
     outputs: soup manifest, `READOUT-M4B.json`, `SEAL.json`, `COLLECT.json`. Relaunch with
     `setsid nohup bash /data/dev2/src/<mirror>-src_training_decision2/src/training/decision2/v2/27b/m5/m5-l128-chain.sh <mirror> 0 <s1 driver pid>`.
     Once s1 has finished, the pid can be any dead pid.
   - Manual equivalents (as in M4b): `m5-tail.sh lsoup|readout|devgates|formal|mlx|mlx-push|mlx-pull`, with
     `CHECKPOINT_FORMAT=peft-lora/1` for readout / formal / mlx and `LOADED_PARAMETERS=27497508864` for formal. On
     node A, `m5-mlx-nodeA.sh <mirror> M5-L128`. Then `m5-gates.sh <mirror> gates|overlap|verdicts M5-L128`.
4. Verdicts → results record `m5-results-2026-09-30.md` (disclose amendment 2's deviation), gist 06, merge, report.
   A successor needs a release hand-off:
   - Stage its frozen package on node A over the link. An M5-L128 successor is an adapter package like A20r's: the
     rank-256 adapter + head on the pinned base, T = 1 or CAL698, 27,497,508,864 loaded parameters, no `bf16z`
     needed. Full weights (96 GB FP32) would need `bf16_copy` + `bf16z`, but no full-weight candidate is left.
   - Write a C1 post-key successor spec: format `v2/eval/sealed/c1-postkey/*.json`, role successor, 27B kernel
     adapter, the formal run's post-run cache and stored typed-final / public231 predictions.
   - Note the privacy-screen rule: soup `decision_config.json` records node paths, so the release must write
     track-relative paths.
5. Cleanup at the end:
   - Remove the temporary key (see Infrastructure) and set the leases to reserved-idle: node B GPU0–1, GPU5; node A
     GPU2.
   - Check that the node A watchers have ended: the mlx watcher ends on `mlx/M5-L128.SKIP` or after scoring.
   - Delete non-BEST full checkpoints of the FF runs only after the results are recorded; keep every BEST and soup.

## Poll log (newest first)

- 10:03Z: L128-s1 202 / 3,561, s2 200 / 3,561 (9.8–9.9 s/upd; ETA ≈ 19:10–19:30Z). Validated on node B with `DRY_RUN=1` (no GPU, no lease change): `run_readout.sh` with `CHECKPOINT_FORMAT=peft-lora/1` on A20r's soup passes every stage and argcheck. The default `full` refuses it, and an unknown format is refused; scratch removed. Integration merged at `64b608bb0`. Disk: node B 64%, node A 41%.
- 09:55Z: interim results record `m5-results-2026-09-30.md` (FF arms final; development attribution: dose at full FT −.037 [−.053, −.021], round-2 and cross-soup TIE; files `readouts/attribution/` on node B). L128 training; chain L128 and node A watchers waiting.
- 09:50Z: FF20H / SX gates: no FF finalist (every candidate HT-DEV v2 FLAG). L128-s1 141 / 3,561, s2 138 / 3,561 (≈ 10 s/upd). L128 tooling `b2af94a9e` + `c8b1d9128` mirrored; `lsoup` test exact. Node A watchers and chain L128 running. Receipts 50.05 GPU-h.
- 09:20Z: continuation worker. B1 recorded late (amendment 2). Leases node B GPU5 / node A GPU2 retaken 09:13Z. `BRANCH-B1` written; L128-s1 / s2 launched 09:16Z (admit). M5-FF20H / M5-SX CAL fits done, readout collections running. Receipts 49.46 GPU-h.
- 04:00Z: FF20 both seeds complete (BEST 891; SELECT .903 / .869; 10.25 / 10.47 GPU-h); FF20H-s1 full running, FF20H-s2 preflights; chain-FF20 pulling FF20-s2.
- 03:19Z: FF20-s1 831/891, FF20-s2 795/891; both ≈ 3.45 h per attempt (cap 4.5); chains waiting.
- 02:53Z: FF20-s1 701/891, FF20-s2 672/891 (≈ 13 s/upd); chains waiting; no incident.
- 02:27Z: FF20-s1 593/891 (12.4 s/upd), FF20-s2 553/891 (12.3 s/upd); both chains waiting; no incident.

# ~27B M5 state (resume file)

Updated: 2026-10-01 10:50 UTC+8 (02:50Z; M5 continuation worker 4, started 02:20Z).
**M5 is CLOSED: no successor.** Chain L128 completed unattended at 21:01:55Z. Worker 3 had stopped silently after its
16:50Z poll, and every detached process finished normally. Results are final in **`m5-results-2026-09-30.md`**.
Nothing of track 27b is running. Cleanup is done; only the coordinator decisions under "Next steps" remain.

**Deviation (prereg amendment 2):** M5-FF20 failed development gate 2 at 04:26:53Z (HT-DEV v2 .516 vs A20r .565, Δ −.049
[−.070, −.029], FLAG; gate 4 failed too), so B1 triggered. No worker was alive, so FF20H was not stopped. FF20H trained
to completion and M5-SX was built. Both went through their preregistered gates as usual and failed, and the results
disclose the deviation. B1's L128 branch was launched late, at 09:16Z.
Branch: `xunzhuo/decision-2-training-27b` (worktree `/home/xunliu/code/vllm-sr-dev2-27b`; merge-only into
`xunzhuo/decision-2-training`). Gist file: `06-decision-2-27b.md`. Assignment: COORDINATION 2026-09-30 07:20.
Prereg: `m5-prereg-2026-09-30.md` (+ amendments 1–2). Latest node mirror (both nodes): `c8b1d9128` (L128 tail; the
L128 training ran from `e76e56d4c`).

## Outcome

- **M5-L128 was the only finalist; it is not a successor and does not beat AutoJev-27B.** Post-key v3 74.73 (T .948,
  H .589; seal `93042ce0…`; package T = 1, model `95d61175…`, 27,497,508,864 loaded parameters). Node B
  `gates/VERDICTS-20260930T210155Z.json`:
  - item 1 fails: +2.37 [−0.56, +3.82] vs A20r;
  - item 4 fails: mlx-diag card-eligible −.013 [−.024, −.002] (Noul −.026);
  - items 2, 3, 5, 6 and 7 pass;
  - beats-AutoJev fails: +2.59 [−0.29, +6.28];
  - choice: none. Item 8 is not handed off, and no release hand-off is needed.
- The gain is typed only: T +.052 [+.036, +.068], H +.005 [−.038, +.026], HT-DEV v2 −.011 (TIE).
- No FF candidate was a finalist (every one HT-DEV v2 FLAG).
- Same-data development reading: L128 − M5-FF20 on HT-DEV v2 is +.039 [+.021, +.057], a GAIN. The adapter keeps the
  human transfer that full fine-tuning loses.
- **DEV2.0-27B stays A20r** (`main` @ `53233103`, 72.36).

## Budget (final)

- **70.99 of 72 GPU-h:** node B 35.995, node A 34.998. Node A's `mlx-diag/M5-L128/GPU-TIME.json` is a copy of node B's
  run and is counted once.
- Sum receipts with `ssh <node> python3 - < ~/.cache/m5-work/budget.py`. Items are listed in the results record.

## Running now

Nothing for track 27b; checked 02:24Z by container names and PIDs.

- L128-s1 / s2 drivers ended at 19:21Z / 19:18Z.
- Node A's relay watcher ended at 19:19:43Z and its mlx watcher at 20:59:38Z, after scoring.
- Chain L128 (PID 2495533) ended at 21:01:55Z with `m5 l128 chain complete`.

**Node A GPU3–5 and node B GPU6–7 belong to the 27B MoE worker** (track `27b-moe`, worktree `vllm-sr-dev2-27b-moe`,
COORDINATION 23:50). Never touch its GPUs, files, link key (`27b-moe-xfer`, node A → node B) or branch, and never
co-tenant.

## Infrastructure (after cleanup)

- **Private node link (node B → node A): REMOVED at 02:31Z.**
  - On node A, the `authorized_keys` line with comment `dev2-27b-m5-xfer-temp` is gone. The file is now byte-equal to
    its pre-key backup `authorized_keys.bak.27b-m5-20260929T233147Z` (mode 600).
  - A test rsync with the key from node B was refused ("Permission denied").
  - Node B's `/data/dev2/tmp/27b-m5-xfer` was deleted; no process was using it.
  - `~/.cache/m5-work/mirror2.sh` relies on that link and no longer works. Mirror each node directly with
    `v2/common/mirror_to_node.sh --path src/training/decision2 node-a|node-b <sha>`.
- **Leases (02:33Z):** node B GPU0, GPU1 and GPU5 and node A GPU2 are `status=reserved-idle`, track `27b`
  ("27b M5 closed 2026-10-01 (no successor); reserved-idle for track 27b until the coordinator reassigns"). Each
  previous owner file is kept as `owner.prev-<UTC>`, and other tracks' co-tenant entries are untouched.
- **Staged on node A** (`/data/dev2/xfer/27b-m5/stage/`, hash-equal to node B): A20r soup checkpoint + package, and the
  F-b checkpoint (96 GB FP32) + package. These are kept.
- **ht-dev2 is installed on node B** (`/data/dev2/private/panels/{goldfree,gold}/ht-dev2.*`).
- **Mixtures:** `a20` `4aa0dc96…` (= M4) and `a20h` `4a9d93f5…` (`MIXTURES.json` `9ad3da72…`); the node A copy is
  hash-equal.

## Done

- **Step 1, HT-DEV v2 references** (node A; 0.486 GPU-h): F1 .5779, **A20r .5655**, F-b .5527, Eikos .5558, AutoJev
  .5443 (FLAG). Outputs are in node A `/data/dev2/runs/27b/m5/htdev2/<key>/`.
- **A20r node B reference readout** (0.249 GPU-h; `readouts/m4-a20r-soup`, symlink `readouts/A20r-ref`).
- **Prereg** `b31cd76e9`; amendment 1 `7f07c4904`; amendment 2 (B1 applied late).
- **FF20 and FF20H trained** (all four seeds exit 0). Soups M5-FF20 / M5-FF20H / M5-SX verified to 6e-8.
- **FF development gates:** no FF finalist (`DEVGATES-20260930T042653Z`, `…092543Z`).
- **L128** (B1; s1 node B GPU5, s2 node A GPU2):
  - Both seeds completed, BEST = final update 3561. SELECT700 .932 / .927. Full runs 9.99 / 9.94 GPU-h.
  - The relay and soup were exact (`95d61175…`, 8.9e-7).
  - Readout P_dev 76.99, H_dev2 .5548.
  - **Devgates `DEVGATES-20260930T201334Z`: finalist** (HT-DEV v2 TIE −.011; typed guard .8875 ≥ .8863).
- **M5-L128 formal and verdicts** (node B GPU0):
  - CAL698 not adopted, so T = 1.
  - Smoke, then collection: 0 invalid, autotune 0 added, seal `93042ce0…`, v3 74.725.
  - mlx-diag: collected on node B, scored on node A at 20:59Z.
  - Gates, overlap and verdicts ran at 21:01–21:02Z.
- **Attribution:** `readouts/attribution/htdev2-l128-vs-ff20.json` (new, CPU) plus the four FF files;
  `gates/contrast.json` (L128 − A20r, formal).
- **Results record final.** The gist 06 update and the integration merge are recorded in the poll log.

## Next steps

1. **Nothing is left to run in M5.** A continuation worker only needs to check that the results commit is on the
   integration branch.
2. **Coordinator decisions:**
   - **Delete the non-BEST FF checkpoints?** They are 96 GB each, ≈ 1.8 TB in total, and are kept until approved:
     - node B `/data/dev2/runs/27b/m5/FF20-s1/full/run/checkpoint-{0000112,0000336,0000560,0000672,0000784}` and
       `FF20H-s1/full/run/checkpoint-{0000141,0000564,0000705}`;
     - node A `FF20-s2/full/run/checkpoint-{0000112,0000224,0000336,0000448,0000560,0000784}` and
       `FF20H-s2/full/run/checkpoint-{0000141,0000282,0000423,0000705,0001122}`.
     - Every BEST, every soup and all L128 artifacts stay.
   - **Lease reassignment:** node B GPU0, GPU1 and GPU5 and node A GPU2 are reserved-idle for 27b.
   - **Next 27B milestone:** typed accuracy is near its ceiling and the interval is H-dominated, so a successor needs a
     human-transfer lever on the adapter recipe. Add an mlx-diag Noul guard for L128-derived arms.

## Poll log (newest first)

- 02:52Z: M5 CLOSED, no successor. Results record final (all arms FF20 / FF20H / SX / L128, verdict table, attributions incl. new development reading L128 − FF20 HT-DEV v2 +.039 [+.021, +.057] GAIN, B1 deviation, final receipts 70.99 GPU-h). Cleanup: temporary node link key removed (node A authorized_keys back to its pre-key backup, key refused; node B key directory deleted); leases node B GPU0 / 1 / 5 and node A GPU2 reserved-idle (02:33Z). Non-BEST FF checkpoints (≈ 1.8 TB) kept pending the coordinator. Next: gist 06, integration merge
- 02:27Z: continuation worker 4 (worker 3 stopped silently after its 16:50Z poll). Chain L128 COMPLETED 21:01:55Z (exit 0): L128-s1 / s2 complete (BEST = final update 3561; 9.99 / 9.94 GPU-h); soup M5-L128 95d61175… finalist (DEVGATES-20260930T201334Z: HT-DEV v2 TIE −.011); formal v3 74.725; VERDICTS-20260930T210155Z: items 1–7 false (item 1 lower bound −0.559, item 4 mlx-diag card −.013 [−.024, −.002]), beats-AutoJev false (lower bound −0.295): no successor. Node A watchers ended; no 27B container running. Receipts 70.99 GPU-h. Results record next
- 16:50Z: L128-s1 2,652 / 3,561 (9.9 s/upd; ETA ≈ 19:18Z), s2 2,687 / 3,561 (ETA ≈ 19:13Z); s2 SELECT700 at 2676 .915 (new BEST, checkpoint written). All processes alive; no incident
- 16:25Z: L128-s1 2,490 / 3,561 (9.9 s/upd; ETA ≈ 19:18Z), s2 2,533 / 3,561 (ETA ≈ 19:11Z). SELECT700 family macro at 1338 / 1784 / 2230: s1 .909 / .913 / .925, s2 .906 / .912 / .914 (each a new BEST). Chain L128, node A watchers and drivers alive. rrsync on node A allows --mkpath (the chain's SKIP / mlx push)
- 15:55Z: continuation worker 3 (Cursor restart stopped worker 2 ≈ 12:40Z). L128-s1 2,303 / 3,561 (9.7 s/upd; ETA ≈ 19:15Z), s2 2,345 / 3,561 (9.2 s/upd; ETA ≈ 19:00Z). Chain L128, node A relay and mlx watchers and both drivers alive. FF20H / SX gates confirmed (DEVGATES-20260930T092543Z: no FF finalist). Receipts 50.05 GPU-h; projection ≈ 70.7 with L128's formal. Integration merged at 8793d333b
- 12:39Z: L128-s1 1,154 / 3,561, s2 1,152 / 3,561 (≈ 10.0 s/upd; ETA ≈ 19:20Z). Chain L128 and node A watchers alive; no incident
- 12:20Z: L128-s1 1,038 / 3,561, s2 1,036 / 3,561 (9.7–9.9 s/upd; ETA ≈ 19:00Z). SELECT700 at 892: s1 .894, s2 .903 (M4-A20r seeds .859 / .880). All processes alive
- 11:48Z: L128-s1 858 / 3,561, s2 848 / 3,561 (9.5–9.8 s/upd; ETA ≈ 19:00Z). Chain L128 and node A watchers alive; no incident
- 11:17Z: L128-s1 669 / 3,561, s2 656 / 3,561 (9.8–9.9 s/upd; ETA ≈ 19:05Z). Leases node B GPU5 / node A GPU2 running the L128 full containers; chain L128 and node A watchers alive
- 10:51Z: L128-s1 512 / 3,561, s2 497 / 3,561 (10.0 s/upd; projected ≈ 10.0 h per attempt). First SELECT700 at update 446: s1 .824, s2 .869 family macro (M4-A20r seeds at 446: .840 / .820). All processes alive
- 10:26Z: L128-s1 369 / 3,561, s2 355 / 3,561 (9.9 s/upd; projected 9.8–9.9 h per attempt, ETA ≈ 19:10Z). Chain L128, node A relay and mlx watchers, and both drivers alive; no incident
- 10:03Z: L128-s1 202 / 3,561, s2 200 / 3,561 (9.8–9.9 s/upd; ETA ≈ 19:10–19:30Z). Validated on node B with `DRY_RUN=1` (no GPU, no lease change): `run_readout.sh` with `CHECKPOINT_FORMAT=peft-lora/1` on A20r's soup passes every stage and argcheck. The default `full` refuses it, and an unknown format is refused; scratch removed. Integration merged at `64b608bb0`. Disk: node B 64%, node A 41%.
- 09:55Z: interim results record `m5-results-2026-09-30.md` (FF arms final; development attribution: dose at full FT −.037 [−.053, −.021], round-2 and cross-soup TIE; files `readouts/attribution/` on node B). L128 training; chain L128 and node A watchers waiting.
- 09:50Z: FF20H / SX gates: no FF finalist (every candidate HT-DEV v2 FLAG). L128-s1 141 / 3,561, s2 138 / 3,561 (≈ 10 s/upd). L128 tooling `b2af94a9e` + `c8b1d9128` mirrored; `lsoup` test exact. Node A watchers and chain L128 running. Receipts 50.05 GPU-h.
- 09:20Z: continuation worker. B1 recorded late (amendment 2). Leases node B GPU5 / node A GPU2 retaken 09:13Z. `BRANCH-B1` written; L128-s1 / s2 launched 09:16Z (admit). M5-FF20H / M5-SX CAL fits done, readout collections running. Receipts 49.46 GPU-h.
- 04:00Z: FF20 both seeds complete (BEST 891; SELECT .903 / .869; 10.25 / 10.47 GPU-h); FF20H-s1 full running, FF20H-s2 preflights; chain-FF20 pulling FF20-s2.
- 03:19Z: FF20-s1 831/891, FF20-s2 795/891; both ≈ 3.45 h per attempt (cap 4.5); chains waiting.
- 02:53Z: FF20-s1 701/891, FF20-s2 672/891 (≈ 13 s/upd); chains waiting; no incident.
- 02:27Z: FF20-s1 593/891 (12.4 s/upd), FF20-s2 553/891 (12.3 s/upd); both chains waiting; no incident.

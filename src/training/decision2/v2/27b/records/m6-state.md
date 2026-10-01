# ~27B M6 state (resume file)

Updated: 2026-10-01 15:12 UTC+8 (07:12Z; M6 worker 1, started 06:17Z).
Prereg `m6-prereg-2026-10-01.md` (`90d38aba7`). Tooling: latest mirror **`482cb0ddb`** on node A and node B
(`20af2e4a1` ran step 0).
Assignment: COORDINATION 2026-10-01 14:25 (27B M6, worker 11741ee2). Branch `xunzhuo/decision-2-training-27b`
(worktree `/home/xunliu/code/vllm-sr-dev2-27b`; merge-only into `xunzhuo/decision-2-training`). Gist file
`06-decision-2-27b.md`. Budget 140 GPU-h. Index numbers are private: never in this file, commits or the gist.

## Goal

A successor to DEV2.0-27B (A20r, `main` `4e89288d`, post-key v3 72.36) that passes successor items 1–8 and also
improves the private Decision Index standing (deficit families: tool-call decisions, phishing; private report only).
Index runs only on frozen finalists, never for selection.

## Inputs (status)

- IB1: round 2 (`82bf70a7`) NOT release-safe; IB1-r3 in progress (data worker 1bad770e). Stage 1 waits for a
  release-safe IB1 record on the integration branch.
- IB2: being built (data worker 24a520c1). Stage 2 waits for it.
- IX1 follow-up (c0ce08eb): runtime fix + private M5-L128 Index diagnostic; holds node C / D GPUs until released.

## Done

- **Step 0, PN1-guard validation: VALIDATED → G5 is a gate.** Node B, mirror `20af2e4a1`, 06:49–06:54Z; A20r on GPU0
  (0.080 GPU-h), M5-L128 on GPU1 (0.086 GPU-h); PN1 dev `c3b68ac1…` (1,974 rows), raw T = 1; caches: no autotune
  entry added. `m6/slices/M5-L128/pn1-vs-A20r.json`:

  | PN1 dev yes-rate | A20r | M5-L128 | Δ [95%] |
  | --- | ---: | ---: | --- |
  | clean gold-no (850 rows) | .1541 | .1729 | **+.0188 [+.0095, +.0294]** |
  | hop, true paraphrases (236) | 1.000 | 1.000 | .000 [.000, .000] |
  | all eight languages | .5760 | .5866 | +.0106 |
  | accuracy | .9200 | .9103 | |

  The guard sees L128's yes-bias significantly, in the same direction as its formal mlx-diag failure (PAWS-X gold-no
  yes .200 → .294).

## Running now

Nothing (no GPU job).

## Leases

node B GPU0, GPU1, GPU5 and node A GPU2: track 27b, `reserved-idle`. Node D: IX1 follow-up (not ours yet; IX1 is
still fixing the long-input runtime bug and staging its M5-L128 Index diagnostic on node C).

## Infrastructure

- **M6 node link (node B → node A): UP since 07:07Z** (`m6/m6-link.sh setup`, then `check` passed). Key in node B
  `/data/dev2/tmp/27b-m6-xfer/` (mode 700; `peer`, `known_hosts` with node A's host key, verified against node A's own);
  node A `authorized_keys` line `dev2-27b-m6-xfer-temp` = `from=<node B source>`, `command="/usr/bin/rrsync
  /data/dev2/xfer/27b-m6"`, `restrict`; backup `authorized_keys.bak.27b-m6-20261001T070730Z`. **Remove at milestone end
  with `m6/m6-link.sh remove`.** Never touch the MoE worker's link (`27b-moe-xfer`, node A → node B).
- Drivers (`v2/27b/m6/`): `m6-build.sh` (node B), `m6-arm.sh` (either node), `m6-relay.sh` + `m6-mlx-watch.sh`
  (node A), `m6-chain.sh` (node B), `m6-tail.sh` / `m6-gates.sh` (node B stages), `m6_devgates.py` (integration-tested on
  node B with real files: reproduces M5's L128 readout values, fails L128 on G5).

## Stage-1 launch runbook (when a release-safe IB1 record is on the integration branch)

1. Read the IB1 record: dataset revision, `ib1.train.jsonl` / `ib1.dev.jsonl` SHA-256, `release_safe: true`. Commit the
   data-lock amendment skeleton (revision and hashes) **before** any GPU job.
2. Node B: `bash <mirror>/v2/27b/m6/m6-build.sh REV TRAIN_SHA DEV_SHA` (detached, log `m6/logs/build.log`; ≈ 30–40
   min, two builds). Fill the data lock from `/data/dev2/private/27b/m6-data/BUILD.json` (mixture SHA-256, tokens,
   updates, `SAVE_EVERY`, projection); commit + push + mirror.
3. Push `a20ib1x.train.jsonl` to node A over the link (`rsync -e "$X" … root@peer:relay/mix/`), move it to
   `/data/dev2/private/27b/m6-data/mixtures-m6-1/` on node A and check its SHA-256.
4. Node B GPU0 / GPU1: IB DEV reference slices `A20r-ib1` (A20r soup) and `M5-L128-ib1` (`m6-tail.sh slices … ib=…`).
5. Launch (`m6-arm.sh`): M6-IB-s1 node B GPU5, M6-IB-s2 node B GPU1, M6-IBX-s1 node B GPU0, M6-IBX-s2 node A GPU2
   (`DEV2_NODE` set by the script). Record the four driver PIDs; confirm containers `d2-27b-M6-*` and first log lines.
6. Node A: `m6-relay.sh M6-IBX-s2 <pid>`, `m6-mlx-watch.sh <sha> M6-IB`, `m6-mlx-watch.sh <sha> M6-IBX` (detached).
7. Node B: `m6-chain.sh <sha> M6-IB-s1=b:<pid> M6-IB-s2=b:<pid> M6-IBX-s1=b:<pid> M6-IBX-s2=a` with `PN1_ROWS`,
   `PN1_SHA`, `IB_ROWS`, `IB_SHA`, `IN_DIST="w2c isarc"`, `AUX_M6_IB=1`, `AUX_M6_IBX=0` (detached, log
   `m6/logs/chain-stage1.log`). Confirm PID and first log line.

## Next steps

1. Wait for IB1-r3; then the runbook above.
2. Integration merge; gist 06 entry (no Index values).

## Poll log (newest first)

- 07:12Z: chain tooling committed (`7d9e2ef59`, `482cb0ddb`: gates, build / arm / chain, node A relay and mlx watchers,
  `m5_verdicts` M6 mapping; 27 tests pass, shellcheck clean); mirrored to node A / B; node link up and checked;
  `m6_devgates.py` integration test on node B passed (scratch removed). IB1-r3: amendment 3 + build code committed,
  review pending. IX1 follow-up still on node C / D.
- 06:56Z: step 0 done: PN1 guard VALIDATED (L128 − A20r clean gold-no +.0188 [+.0095, +.0294]; hop level); 0.166
  GPU-h. Gates module `m6_devgates.py`, build / arm drivers written (not yet committed).
- 06:53Z: prereg committed (`90d38aba7`); tooling `20af2e4a1` (slices, PN1 / breadth scoring, M6 allocations, tail
  driver; 13 new tests + the existing 27B suites pass) mirrored to node B; PN1 dev fetched on node B (hash equal);
  step 0 launched on node B GPU0 / GPU1 (containers up, first log lines present).
- 06:35Z: worker 1 started; integration merged (fast-forward to `50bae2ddd`); inputs read (M5 results, COORDINATION
  to 14:25, IB1 r2 records, IX1 public records and its private report). mlx-diag diagnosis of M5-L128 (node A, CPU, from
  the scored files): its Noul loss is a PAWS-X yes-bias (gold-no yes-rate .200 → .294), the 9B failure mode.

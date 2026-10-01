# ~27B M6 state (resume file)

Updated: 2026-10-01 15:27 UTC+8 (07:27Z; M6 worker 1, started 06:17Z).
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

- **IB2: release-safe and merged into integration** (`7d9841c36`). Private dataset revision
  `c5dbdd0a88efe58059c6ece8ae2b181f9132619f`, `m6/ib2/`: TRAIN 24,518 rows `ee137efa…` (≈ 5.4M tokens; `argq`, `fc_rel`,
  `fc_ready`, `ytspam`, `hover`, `gsm2`; in-distribution `hover`, `gsm2`), DEV 1,272 `ab009fb1…` (`argq`, `gsm2`,
  `hover`, `ytspam`). Under 10M tokens, so stage 2 takes it whole.
- **IB1 round 3: review PASSED (3 / 216), status `release_safe: true` on the IB1 branch (`948811f66`)**; the r3 final
  files, upload revision and the integration merge are still pending. Stage 1 waits for them.
- IX1 follow-up (c0ce08eb): still fixing the long-input runtime bug (07:15Z); its M5-L128 Index diagnostic follows; holds
  node C / D.

## Plan change (to be recorded in the data-lock amendment, before any training GPU job)

IB2 landed together with IB1-r3, so stage 1 and stage 2 are ready at once, but only the four 27B leases are free. The
two arms the milestone goal depends on go first: **M6-IB** (`a20ib1`: node B GPU5 + GPU1) and **M6-IB2** (`a20ib12` =
A20 + IB1 + IB2, the stage-2 default IB1 block: node B GPU0 + node A GPU2). **M6-IBX** (the attribution ablation) runs
on the next free pair (node D once IX1 releases it, or two of these GPUs after training). One build
(`m6-build.sh` with the IB2 arguments) makes all four mixtures; one chain per arm (`m6-launch.sh`), IB DEV slice `ib`
for M6-IB and `ib12` (IB1 + IB2 DEV) for M6-IB2.

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

## Running now (launched 07:56Z from mirror `b74685ddb`, amendment 1)

| Seed | Node / GPU | Driver PID | Mixture | Cap | Container prefix |
| --- | --- | ---: | --- | ---: | --- |
| M6-IB-s1 | node B GPU5 | 2768538 | `a20ib1` (5,081 updates, save 636) | 16.0 | `d2-27b-M6-IB-s1-*` |
| M6-IB-s2 | node B GPU1 | 2768810 | `a20ib1` | 16.0 | `d2-27b-M6-IB-s2-*` |
| M6-IB2-s1 | node B GPU0 | 2769039 | `a20ib12` (6,614 updates, save 827) | 20.0 | `d2-27b-M6-IB2-s1-*` |
| M6-IB2-s2 | node A GPU2 | 147815 | `a20ib12` | 20.0 | `d2-27b-M6-IB2-s2-*` |

- Drivers log to `/data/dev2/runs/27b/<seed>/driver.log` (stages admit → onestep → reload → full).
- Node B chains: `m6-chain.sh` for M6-IB (PID 2769370, log `m6/logs/chain-M6-IB.log`, aux GPU1, slice `ib`) and
  M6-IB2 (PID 2769422, log `m6/logs/chain-M6-IB2.log`, aux GPU0, slice `ib12`); first lines present.
- Node A: relay watcher `m6-relay.sh M6-IB2-s2 147815` (PID 148145), mlx watchers for M6-IB (148094) and M6-IB2
  (148270); logs in node A `/data/dev2/runs/27b/m6/logs/`.
- Reference slices done (node B): A20r / M5-L128 × `ib` (0.085 / 0.092 GPU-h) and × `ib12`.
- Liveness: by PID and container name (`docker ps | grep d2-27b-M6`), never `pgrep -f`.

## Leases

node B GPU0, GPU1, GPU5 and node A GPU2: track 27b, running the four seeds. Node D: IX1 follow-up (not ours yet; IX1
is still fixing the long-input runtime bug and staging its M5-L128 Index diagnostic).

## Infrastructure

- **M6 node link (node B → node A): UP since 07:07Z** (`m6/m6-link.sh setup`, then `check` passed). Key in node B
  `/data/dev2/tmp/27b-m6-xfer/` (mode 700; `peer`, `known_hosts` with node A's host key, verified against node A's own);
  node A `authorized_keys` line `dev2-27b-m6-xfer-temp` = `from=<node B source>`, `command="/usr/bin/rrsync
  /data/dev2/xfer/27b-m6"`, `restrict`; backup `authorized_keys.bak.27b-m6-20261001T070730Z`. **Remove at milestone end
  with `m6/m6-link.sh remove`.** Never touch the MoE worker's link (`27b-moe-xfer`, node A → node B).
- Drivers (`v2/27b/m6/`): `m6-build.sh` (node B), `m6-arm.sh` (either node), `m6-relay.sh` + `m6-mlx-watch.sh`
  (node A), `m6-chain.sh` (node B), `m6-tail.sh` / `m6-gates.sh` (node B stages), `m6_devgates.py` (integration-tested on
  node B with real files: reproduces M5's L128 readout values, fails L128 on G5).

## Launch runbook (when the IB1-r3 record is on the integration branch with its upload revision)

1. Read the IB1 record (`v2/data/records/ib1/status.json`, `final.json`, `hf-readback.json`): revision, `ib1.train.jsonl`
   / `ib1.dev.jsonl` SHA-256, `release_safe: true`; IB2's are above.
2. Node B (detached, log `m6/logs/build.log`; ≈ 30–40 min): `bash <mirror>/v2/27b/m6/m6-build.sh REV1 TRAIN1 DEV1
   c5dbdd0a88efe58059c6ece8ae2b181f9132619f ee137efa8bbf86e5c62574f3b8fbd6204063e095a8da514f32600ba3d51d1cfa
   ab009fb12f9563c3ef4a846f5455dd5233f2a2c5fafe6ac8a9c74349532eb923`.
3. Data-lock amendment from `/data/dev2/private/27b/m6-data/BUILD.json` (revisions, mixture SHA-256, tokens, updates,
   `SAVE_EVERY`, projections, caps, C1 source counts, the plan change above); commit, push, mirror to node A / B.
4. Workstation: `bash v2/27b/m6/m6-launch.sh <mirror> M6-IB:a20ib1:ib:b5,b1:1 M6-IB2:a20ib12:ib12:b0,a2:0`
   (node A mixture push + hash check, reference slices A20r / M5-L128 for `ib` and `ib12`, four seeds, node A relay and
   mlx watchers, one chain per arm; prints driver PIDs and first log lines). Record everything here.
5. M6-IBX later: `m6-launch.sh <mirror> M6-IBX:a20ib1x:ib:<s1>,<s2>:<aux>` on the next free pair.

## Next steps

1. Confirm the four preflights (onestep finite, reload parity 0 argmax changes) and the first full-run updates; record
   seconds per update and ETAs.
2. M6-IBX: needs a free GPU pair (node D after IX1, or a coordinator lend); then
   `m6-launch.sh <mirror> M6-IBX:a20ib1x:ib:<s1>,<s2>:<aux>` (a node D placement needs a launcher map amendment).
3. After the chains: results record, item-8 hand-off (custodian C1 content recheck first, IB1 + IB2 roots) and the
   private Index request for frozen finalists.

## Poll log (newest first)

- 08:12Z: all four preflights passed (onestep exit 0, 0.072–0.077 GPU-h; reload 0 argmax changes on 32 rows, max |Δp|
  3e-8 / 6e-8; 0.018–0.020 GPU-h); full runs since ≈ 08:03Z. After ≈ 26 updates: M6-IB-s1 9.57 s/upd (projection
  13.5 h of cap 16), M6-IB-s2 9.50 (13.4 h), M6-IB2-s1 9.36 (17.2 h of cap 20), M6-IB2-s2 9.70 (17.8 h). The cost is
  mostly per row (short IB rows cost ≈ 0.6 s each, like A20 rows), so amendment 1's projection was optimistic; the
  caps still hold with 11–16% margin. Watch M6-IB2-s2's margin at every poll. ETAs: M6-IB ≈ 21:35Z, M6-IB2 ≈ 01:35Z
  (Oct 2). Reference reading (CPU): M5-L128 − A20r on IB DEV is level (`ib` −.003 [−.016, +.009]; `ib12` +.004
  [−.006, +.012]); A20r's B_dev is .920 (`ib`) / .901 (`ib12`), headroom in `args`, `isarc`, `gsm2`, `hover`. Node D
  runs IX1's M5-L128 Index run until ≈ 11:00Z.
- 08:00Z: **launched** (amendment 1 `b74685ddb`; build 07:31–07:36Z, two builds identical, all checks pass). Four seeds in
  onestep preflight; chains and node A watchers alive with first log lines. Incident (harmless): the workstation
  launch was started twice (a tool-call retry); the second copy stopped at the `ib12` reference-slice step on a
  "fresh cache exists" refusal before launching anything (its error lines overwrote the start of the two `ib12`
  slice logs; the first copy's slices completed and wrote their receipts). Exactly one set of seeds, watchers and
  chains exists.
- 07:27Z: IB2 release-safe and on integration; IB1-r3 review passed (status release-safe on its branch; upload and
  merge pending). Stage-2 build support (`758285710`: `a20ib12`, IB1 + IB2 DEV slice `ib12`, slice-name plumbing) and
  the launch orchestrator (`84b3ff8cc`, placement table, one chain per arm) committed. Plan change: M6-IB + M6-IB2 first,
  M6-IBX on the next free pair (to be recorded in the data lock). Integration merged at `ddc64bb5b`; gist 06 entry
  added.
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

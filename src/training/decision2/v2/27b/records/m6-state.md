# ~27B M6 state (resume file)

Updated: 2026-10-01 23:28 UTC+8 (15:28Z; **worker 3 = the coordinator's continuation #2, 0d2e488f, on duty from
15:12Z**; worker 2 a56025bb ran 10:44–15:15Z and handed off; worker 1 11741ee2 ran 06:17–10:55Z).
Prereg `m6-prereg-2026-10-01.md` (`90d38aba7`) with amendments 1 (`b74685ddb`), 2 (`35492ac25`) and **3 (`4e4211aee`,
data lock `7fcbc824c`)**. Mirrors: M6-IB / M6-IB2 seeds, chains and watchers run from **`b74685ddb`**, M6-IBX's from
**`d8edcf4e1`**, M6-IB2PN's from **`7fcbc824c`** (on node A / B / D); step 0 ran from `20af2e4a1`. Hand-off record for
other tracks: `m6-handoff-2026-10-01.md`. Results tables: `python3 -m v2.27b.m6.m6_report` (from mirror `4ee6b920e` or
later on node B).
Assignment: COORDINATION 2026-10-01 14:25 (27B M6, worker 11741ee2). Branch `xunzhuo/decision-2-training-27b`
(worktree `/home/xunliu/code/vllm-sr-dev2-27b`; merge-only into `xunzhuo/decision-2-training`). Gist file
`06-decision-2-27b.md`. Budget 140 GPU-h. Index numbers are private: never in this file, commits or the gist.

## Hand-off (worker 2 → continuation, 15:10Z)

- **State at 15:03Z:** eight seeds, four chains, five node A watchers / relays and four node D relays alive. GPU-h
  **≈ 48.7 of 140** (closed 1.314 + running ≈ 47.4); projection ≈ 134 with the hedge (M6 funds no Index run).
  Development SELECT700 bests so far (family macro): M6-IB .9148 / .9115, M6-IB2 .8931 / .8944, M6-IBX .8865 /
  .8873, M6-IB2PN .8173 / .8631 (first checkpoint only).
- **Eight seeds train unattended; four chains carry them to verdicts** (one per arm; M6-IB2PN is amendment 3's G5
  hedge). Nothing needs a manual step until a chain ends, except polling (≤ 45 min, state commit each time; > 60 min
  without a commit while GPU jobs run counts as a silent stop; tool sleeps here overran by up to 65%, so sleep ≤ 20
  min and check the clock). Training ends (measured rates): **M6-IB ≈ 21:05Z / 21:45Z** (s1 / s2), **M6-IBX ≈
  22:35Z**, **M6-IB2 ≈ 01:40Z** (Oct 2), **M6-IB2PN ≈ 06:20Z** (Oct 2); each chain then needs ≈ 1–2 h (soup, readout,
  slices, gates; formal + mlx-diag + items 1–7 only for passers).
- **Program notes since 18:50 that touch M6:** merges must be signed (`git merge --signoff`; this worker's merge
  `d96250da6` at 11:42Z predates the reminder and is unsigned, not rewritten); release cards now carry an Index section
  built by the release track from private reports (user 22:10) — Index values still never go into commits, gists or
  COORDINATION / STATUS; node D GPU4–7 stay idle and unleased by M6 (the coordinator may reassign them); IB3 data
  (phishing, grounding, ESCI, MCQ, contracts) is being built for a later milestone.
- **Poll (workstation):** per node `docker ps | grep d2-27b-M6`; driver / chain / relay / watcher PIDs ("Running now");
  per seed `full/run/train-metrics.jsonl` (step, `seconds`) and `select-step-*-metrics.json` (family macro). Watch
  M6-IB2PN's projection against its 22 GPU-h cap (expected ≈ 18.6–19.2).
- **When a chain's gates land:** on node B, from mirror `4ee6b920e` or later:
  `python3 -m v2.27b.m6.m6_report devgates /data/dev2/runs/27b/m6/readouts/DEVGATES-<UTC>.json --root
  /data/dev2/runs/27b/m6 --pn1-rows <PN1 dev path>` (the `PN1=` path in `m6-launch.sh`) → G1–G6 with intervals and
  the reported-only es / fr view for the results record; after verdicts, `m6_report verdicts
  /data/dev2/runs/27b/m6/gates/VERDICTS-<UTC>.json`. Each chain writes its own DEVGATES / VERDICTS file (one arm
  each); the "≤ 2 finalists per stage" rule cannot bind (stage 1: M6-IB, M6-IBX; stage 2: M6-IB2, M6-IB2PN).
- **If a chain stops:** rerun only the failed stage with `m6-tail.sh <stage> <that chain's mirror> …` (or
  `m6-gates.sh`), then the chain's remaining steps in `m6-chain.sh`'s order; never rerun a seed.
- **Finalists:** formal, mlx-diag (scored on node A by the watcher), items 1–7 and beats-AutoJev run in the chain. Then
  `m6-handoff-2026-10-01.md`: §1 the custodian's C1 content recheck (**requested 11:50Z; can run now**), §2 item 8 for
  the chosen finalist (fill in the package path and spec), §3 the private Index request to IX1 (M6 funds none), §4 the
  release hand-off on top of the 27B forward-budget fix revision of `main` `3236518c` (check that it has landed).
- **Attribution after all chains:** `m6-gates.sh <mirror ≥ c0056edca> gates <all sealed finalists>` for the formal
  contrasts (now including M6-IB2PN vs M6-IB2); development contrasts between arms from the readouts / slices.
- **Milestone end** (after M6-IB2PN's chain, ≈ 09:00Z Oct 2; its mlx push / pull uses the link): `m6/m6-link.sh
  remove`; node D leases GPU0–3 released (owner files back to idle / released for the next user); node B GPU0 / 1 / 5
  and node A GPU2 back to `reserved-idle`; final results, gist 06, merge into integration.

## Hand-off (worker 1 → continuation, ≈ 10:55Z)

- **Six seeds train unattended; three chains carry them to verdicts.** Nothing needs a manual step until a chain ends,
  except polling (≤ 30 min, state commit each time; > 60 min without a commit while GPU jobs run counts as a silent
  stop). ETAs: M6-IB ≈ 21:15Z, M6-IBX ≈ 22:15Z, M6-IB2 ≈ 01:20Z (Oct 2); each chain then needs ≈ 1–2 h (soup, readout,
  slices, gates; formal + mlx-diag only for passers).
- **Poll command (workstation):** per node `docker ps | grep d2-27b-M6`, the driver PIDs and chain PIDs below, and
  `full/run/train-metrics.jsonl` (`seconds` per update) / `select-step-*-metrics.json` (family macro) per seed.
- **When a chain ends:** see "Hand-off templates" (results record, item 8 with the custodian C1 recheck first, private
  Index of frozen finalists, release hand-off). A failed seed means no candidate for that arm (no rerun).
- **Decisions and incidents this session (all recorded in the prereg amendments or below):** launch order (M6-IB +
  M6-IB2 first; amendment 1); M6-IBX on node D (amendment 2); projection recalibrated on L128's receipts and stage-2
  cap 20; `read_lease` parses IX1's one-line owner files (`d8edcf4e1`); a duplicated workstation launch stopped
  harmlessly at a reference-slice refusal (07:49Z).
- **Infrastructure to remove at milestone end:** the M6 node link (`m6/m6-link.sh remove`), node D leases (GPU0 / GPU1,
  owner `track=27b`), and optionally node D's staged inputs (≈ 53 GB).

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

| M6-IBX-s1 | **node D GPU0** | 3894132 | `a20ib1x` (4,997 updates, save 625) | 16.0 | `d2-27b-M6-IBX-s1-*` |
| M6-IBX-s2 | **node D GPU1** | 3894589 | `a20ib1x` | 16.0 | `d2-27b-M6-IBX-s2-*` |
| **M6-IB2PN-s1** | **node D GPU2** | 609345 | `a20ib12pn` (6,886 updates, save 861) | **22.0** | `d2-27b-M6-IB2PN-s1-*` |
| **M6-IB2PN-s2** | **node D GPU3** | 645978 | `a20ib12pn` | **22.0** | `d2-27b-M6-IB2PN-s2-*` |

- **M6-IB2PN (amendment 3, G5 hedge)** launched from mirror **`7fcbc824c`** (tooling `c0056edca` + data lock):
  `M6_BUILD=BUILD-pn.json STAGGER=1 m6-launch.sh 7fcbc824c… M6-IB2PN:a20ib12pn:ib12:d2,d3:1`. s1 started 11:32Z alone
  (onestep 0.072, reload 0.017 GPU-h, 0 argmax changes on 32 rows, max |Δp| 7e-8; full run from ≈ 11:38Z); s2 started
  11:38:28Z after s1's reload passed. Node D relays 646528 (s1) / 646813 (s2); node A mlx watcher 253360; node B chain
  **2818836** (aux **GPU1**, slice `ib12`, log `m6/logs/chain-M6-IB2PN.log`). Leases node D GPU2 / GPU3 taken from IX1's
  released owners (moved to `owner.prev-20261001T1131*`). ETA ≈ 06:40Z (s1) / 06:50Z (s2) on Oct 2, then ≈ 2 h chain.
  Data: `/data/dev2/private/27b/m6-data/BUILD-pn.json`, `mixtures-m6pn-1/`, `pn1h.train.jsonl`; scan receipts under
  node A / B `/data/dev2/private/27b/m6-pn/` (private).

- M6-IBX launched 08:33Z from mirror `d8edcf4e1` (amendment 2; node D staged at 08:22Z): node D leases GPU0 / GPU1 taken
  from IX1's released owners (moved to `owner.prev-20261001T0832*`); node D relay watchers 3895113 (s1) / 3895407 (s2)
  with `RELAY_NODE=d`; node A mlx watcher M6-IBX 174557; node B chain M6-IBX PID 2789875 (aux **GPU5**, slice `ib`,
  log `m6/logs/chain-M6-IBX.log`). First launch attempt (mirror `35492ac25`) stopped before any GPU job: `launch3` could
  not parse IX1's one-line owner file; fixed in `d8edcf4e1` (`read_lease` splits single-line `key=value` owner files).
- Drivers log to `/data/dev2/runs/27b/<seed>/driver.log` (stages admit → onestep → reload → full).
- Node B chains: `m6-chain.sh` for M6-IB (PID 2769370, log `m6/logs/chain-M6-IB.log`, aux GPU1, slice `ib`) and
  M6-IB2 (PID 2769422, log `m6/logs/chain-M6-IB2.log`, aux GPU0, slice `ib12`); first lines present.
- Node A: relay watcher `m6-relay.sh M6-IB2-s2 147815` (PID 148145), mlx watchers for M6-IB (148094) and M6-IB2
  (148270); logs in node A `/data/dev2/runs/27b/m6/logs/`.
- Reference slices done (node B): A20r / M5-L128 × `ib` (0.085 / 0.092 GPU-h) and × `ib12`.
- Liveness: by PID and container name (`docker ps | grep d2-27b-M6`), never `pgrep -f`.

## Leases

node B GPU0, GPU1, GPU5 and node A GPU2: track 27b, running the four M6-IB / M6-IB2 seeds. **Node D GPU0–GPU3: track
27b** (M6-IBX on GPU0 / 1 since 08:32Z, M6-IB2PN on GPU2 / 3 since 11:31Z; IX1's released owner files moved to
`owner.prev-*`). Node D GPU4–7: not leased by M6 (IX1's released owners; GPU5 has no owner file).

## Infrastructure

- **Contrast guard on node B (worker 3, since 15:59Z): PID 2868454**, mirror `b980dd144`, log
  `m6/logs/contrast-guard.log`. The four chains share `m6/gates/`, and each chain's `m6-gates.sh gates` ends with
  `m4_contrast --output gates/contrast.json`, which refuses an existing file, so every chain after the first one
  with a sealed finalist would have stopped there, before `overlap` and `verdicts` (found by reading the code; no
  chain has reached it yet). The guard moves each complete `contrast.json` to `contrast-<finalists>-<UTC>.json`
  within 5 s and exits when no chain PID (2769370, 2789875, 2769422, 2818836) is alive. **Fallback** if a chain still
  stops at its contrast step (the chain log ends in a `FileExistsError` after `m6 gates gates <ARM>: start`): from
  that chain's mirror, `m6-gates.sh <sha> overlap <ARM>` and then `verdicts <ARM>` on node B (CPU; the chain's
  formal, mlx and per-arm gate files are complete by then). Nothing reads `contrast.json` downstream.
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

## Hand-off templates (use when a chain's verdicts exist)

(Superseded in part by `m6-handoff-2026-10-01.md`: the C1 content recheck is already requested and covers every arm,
including M6-IB2PN's PN1 roots and `mixtures-m6pn-1`; the Index runs go to IX1, M6 funds none.)

- **Read a chain's outcome:** node B `m6/logs/chain-<ARM>.log` (last line `m6 chain complete…`), the newest
  `m6/readouts/DEVGATES-*.json` naming the arm (gates G1–G6, `finalists`), `m6/gates/VERDICTS-*.json` (items 1–7,
  beats-AutoJev, `choice`). Fill `m6-results-2026-10-01.md` (development table with CIs, formal table, verdicts,
  attribution from `m6/gates/contrast.json`, GPU-h from receipts on node A / B / D).
- **Item 8 (only for a finalist passing items 1–7):** ask the eval custodian (COORDINATION thread) for (1) the C1 content
  recheck with the IB1-r3 and IB2 rescan roots: the data track's node A private run directories
  (`/data/dev2/private/data/ib1/`, `/data/dev2/private/data/ib2/`), the published `m6/ib1` (`31b200a3`) and `m6/ib2`
  (`c5dbdd0a`) files and the M6 TRAIN file of the finalist (node B `/data/dev2/private/27b/m6-data/mixtures-m6-1/`);
  then (2) the C1 post-key successor run against A20r's C1 baseline, from a frozen package copied to node A plus a
  successor spec. The worker never opens C1.
- **Private Index (frozen finalists only; never selects):** ask IX1 (or run its harness under the 27B node D leases once
  free) to restage the finalist's frozen package (node B `m6/<ARM>/package`, LoRA rank 256, T = 1 or CAL698) as a
  diagnostic package answering as DEV2.0-27B at `4e89288d` with the current runtime, run the full 0.2.1 panel with the
  86-request parity gate and dual scoring, and write values only to node `/data/dev2/private/eval/index021/` and
  `decision2-program/private/`. Compare per benchmark with A20r's IX1 run and the 25.8B frontier entrant privately.
- **Release hand-off (only if items 1–8 pass):** adapter package, `runtime_source` = current runtime (BF16-resident
  `5dc962b00`, plus IX1's long-input fix once merged), the C1 spec, the disclosures of prereg "Disclosures" (IB1 / IB2
  families, licences and attributions, in-distribution families of the released arm), to the 27B release worker.
- **Milestone end:** `m6/m6-link.sh remove`; leases back to `reserved-idle`; node D leases released (owner files);
  staged node D inputs may stay for later 27B work (≈ 53 GB on node D's data disk).

## Next steps (worker 2's hand-off, 15:10Z)

1. Poll (≤ 45 min, commit each time) until M6-IB's chain starts processing (≈ 21:45Z, both seeds done).
2. Per chain: when `m6/logs/<ARM>.CANDIDATE` and a `DEVGATES-*.json` naming the arm exist, run `m6_report devgates`
   (see the hand-off above) and fill the development table (G1–G6 with CIs, G5 and G6 especially, plus the es / fr
   view) in `m6-results-2026-10-01.md`; commit. On `m6 chain complete`, run `m6_report verdicts` and fill the formal
   table; record the choice rule's outcome across chains (stage 1: M6-IB / M6-IBX; stage 2: M6-IB2 / M6-IB2PN).
3. **Superseded by worker 3 (15:50Z; `m6-handoff-2026-10-01.md` is now the runbook):** item 8 is this track's own
   step (§2: one attempt, `c1-postkey.sh` on node A GPU2, only for the choice-rule finalist and only with zero
   exposure in the custodian's §1 record), and so is the private Index (§3: `m6-index.sh` on node D GPU4–7, eval
   allowance ≤ 12 GPU-h = one 27B run; first finalist passing items 1–7). The release (§4) supersedes DEV2.0-27B's
   current `main` through the release pipeline, only if items 1–8 pass: **`main` moved to `e7b4a372` at 15:41Z** (the
   card-redesign worker's card-only revision on top of the fix revision `09280791`; spec `dev2-27b-card.json`).
4. After the last chain (M6-IB2PN ≈ 08:30Z Oct 2): attribution (`m6-gates.sh <mirror> contrast` and `verdicts` with
   every sealed finalist; never `gates` again for an arm already gated), final
   results, gist 06, milestone-end cleanup ("Hand-off" above), merge (signed) into integration. **Order:** the
   chosen finalist's `m6-stage-a.sh` (node A, item 8 / release) runs before `m6-link.sh remove`, which deletes node
   B's link directory.

## Poll log (newest first)

- 16:06Z (poll 4 at 16:03Z): all alive, plus the new contrast guard (node B PID 2868454; see "Infrastructure").
  M6-IB 3,107 / 2,948 (13.5 / 14.0 h; ETA 21:27Z / 21:59Z); M6-IB2 2,999 / 2,996 (17.8 / 17.8 h); M6-IBX 2,695 /
  2,659 (13.8 / 14.0 h); M6-IB2PN 1,621 / 1,595 (18.9 / 18.7 h). **GPU-h ≈ 56.9** (closed 1.314 + running full runs
  ≈ 55.5). Fix `b980dd144` (guard, `m6-gates.sh contrast` stage for the final attribution) mirrored to node B and
  node D.
- 15:47Z (poll 3 at 15:43Z): all alive. M6-IB 2,990 / 2,830 (13.5 / 13.9 h); M6-IB2 2,878 / 2,870 (17.8 / 17.8 h);
  M6-IBX 2,573 / 2,540 (13.9 / 14.1 h; **s2 at 2500 .9116, BEST = 2500**); M6-IB2PN 1,500 / 1,471 (18.8 / 18.8 h).
  Node D disk 597 GB. Committed with this entry: `m6/m6-stage-a.sh` (stages the chosen finalist's frozen files on
  node A for item 8 and the release; runbook §2 step 1), and `m6-index.sh stage` now addresses node D directly
  (it no longer reads the link directory's `peer-d`, which `m6-link.sh remove` deletes). Runbook
  `m6-handoff-2026-10-01.md` rewritten at `c8acde617` (item 8, the Index and the release are this track's steps).
- 15:36Z (poll 2 at 15:32Z): all alive. M6-IB 2,917 / 2,760 (13.4 / 13.9 h); M6-IB2 2,807 / 2,800 (17.8 / 17.8 h);
  M6-IBX 2,501 / 2,478 (13.9 / 14.1 h; **s1 at 2500 .8950, BEST = 2500**); M6-IB2PN 1,429 / 1,401 (18.9 / 18.7 h).
  **Private Index tooling for finalists** committed with this entry: `m6/m6-index.sh` (workstation: `stage` node B
  soup → node D + IX1 restage, `parity`, `run`, `status`, `score`, `release`) and `m6/m6-index-run.sh` (node D: the 8
  panel-8 shards on GPU4–7, two per GPU), set up exactly as IX1's M5-L128 diagnostic (restaged into DEV2.0-27B
  `4e89288d` with the forward-budget runtime `e876fbe`, T = 1, A20r's frozen autotune cache), plus the four M6
  `DIAGNOSTIC` entries in `v2/eval/ix1/launch.sh` (a data-only change to the shared IX1 launcher; tests pass).
- 15:28Z (worker 3, poll 1 at 15:16Z): all alive (8 containers; node B drivers and the four chains, node A driver,
  relay and four mlx watchers, node D four drivers and four relays). M6-IB 2,818 / 2,661 (13.2 / 13.9 h; ETA 21:08Z /
  21:53Z); M6-IB2 2,707 / 2,699 (17.7 / 17.8 h; ETA 01:38Z / 01:44Z); M6-IBX 2,411 / 2,382 (13.9 / 14.0 h; ETA 22:29Z /
  22:36Z); M6-IB2PN 1,330 / 1,301 (18.9 / 18.8 h of cap 22; ETA 06:28Z / 06:27Z). Node D disk 572 GB. Integration
  merged (fast-forward to `bc1720c14`). Inputs since worker 2: the 27B forward-budget fix revision **has landed**
  (DEV2.0-27B `main` = `09280791`, COORDINATION 21:25), so §4 of the hand-off builds on it; the custodian's C1
  content recheck (3c7679b0) is running for IB1-r3 + IB2 + PN1-r2 (no record yet); **the finalists' private Index
  runs are now this track's job** on node D GPU4–7 with a separate eval allowance of ≤ 12 GPU-h (outside M6's 140;
  IX1's harness; values private); successor items 1–8 have no exceptions (COORDINATION 23:15). Node D GPU5 holds the
  0.8B fast-track's IX1 parity gate since 15:19Z (their lease); GPU4 / 6 / 7 idle.

- 15:05Z (poll at 15:03Z; worker 2's last): all alive. M6-IB 2,737 / 2,582 (13.0 / 13.7 h; s2 at 2544 .9115, BEST =
  2544); M6-IB2 2,626 / 2,617 (17.6 / 17.6 h; at 2481 .8931 / .8944, BEST = 2481 both); M6-IBX 2,333 / 2,303 (13.7 /
  13.9 h); M6-IB2PN 1,250 / 1,220 (18.7 / 18.6 h). Node D disk 566 GB. GPU-h ≈ 48.7.
- 14:35Z (poll at 14:34Z): all alive. M6-IB 2,546 / 2,412 (13.0 / 13.7 h; s1 at 2544 .9013, BEST stays 1908); M6-IB2
  2,454 / 2,445 (17.5 / 17.6 h); M6-IBX 2,159 / 2,129 (13.7 / 13.9 h); M6-IB2PN 1,074 / 1,043 (18.7 / 18.6 h).
  GPU-h ≈ 44.9.
- 14:14Z (poll at 14:13Z): all alive. M6-IB 2,413 / 2,280 (13.0 / 13.7 h); M6-IB2 2,322 / 2,312 (17.5 / 17.6 h);
  M6-IBX 2,029 / 2,000 (13.7 / 13.9 h; s2 at 1875 .8702, BEST stays 1250); **M6-IB2PN 943 / 911 (18.8 / 18.6 h;
  first SELECT700 at 861: .8173 / .8631, BEST = 861)**. Node D disk 566 GB (first hedge checkpoints). GPU-h ≈ 42.1.
- 13:53Z (poll at 13:52Z): all alive. M6-IB 2,272 / 2,147 (13.0 / 13.7 h); M6-IB2 2,191 / 2,180 (17.5 / 17.6 h);
  M6-IBX 1,900 / 1,875 (13.7 / 13.9 h; s1 at 1875 .8865, BEST = 1875; s2's 1875 evaluation running); M6-IB2PN
  820 / 787 (18.6 / 18.5 h). Node D disk 535 GB. GPU-h ≈ 39.3.
- 13:32Z (poll at 13:31Z; a 38-min wait ran ≈ 63 min, so this commit is ≈ 64 min after the previous one; worker
  alive): all alive. M6-IB 2,129 / 2,014 (13.0 / 13.8 h; third SELECT700 at 1908: **.9148 / .8894**, BEST = 1908);
  M6-IB2 2,058 / 2,046 (17.5 / 17.6 h; s2 at 1654 .8748, BEST stays 827); M6-IBX 1,776 / 1,750 (13.7 / 13.9 h);
  **M6-IB2PN 687 / 653 (18.7 / 18.5 h, cap 22)**. GPU-h ≈ 34.6 (closed 1.314 + running ≈ 33.3).
- 12:30Z (poll at 12:27Z): all alive. M6-IB 1,707 / 1,621 (13.1 / 13.8 h); M6-IB2 1,657 / 1,654 (17.5 / 17.5 h; s1's
  second SELECT700 at 1654 .8695, BEST = 1654); M6-IBX 1,376 / 1,357 (13.8 / 13.9 h; at 1250 .8692 / .8873, BEST =
  1250); **M6-IB2PN 290 / 253 of 6,886 (projections 19.0 / 18.8 h, cap 22)**; M6-IB2PN-s2 preflights: onestep 0.071,
  reload 0.017, 0 argmax changes, max |Δp| 6e-8. Node D disk 524 GB. **GPU-h ≈ 28** (closed 1.314 + running ≈ 26.6).
- 11:55Z (poll at 11:49Z): all eight seeds in full runs (M6-IB2PN-s2 passed both preflights; full run since ≈ 11:44Z).
  M6-IB2PN 54 / 14 of 6,886 at 9.59 / 9.88 s per update (≈ 18.6–19.2 GPU-h at that rate; early wall-clock projections
  include the baseline SELECT700 pass). **Reporter `m6_report.py` (`4ee6b920e`, tests pass):** `devgates` and
  `verdicts` print the results tables; with `--root` / `--pn1-rows` it recomputes each arm's PN1 report from the stored
  probabilities, including the es / fr view. Real-data check on node B reproduces step 0 (M5-L128 − A20r clean
  gold-no +.0188 [+.0095, +.0294]); **the es / fr view sees L128's yes-bias too: +.0179 [+.0036, +.0349] (279 rows),
  hop .000 (18)**, so the view is sensitive in languages absent from PN1H TRAIN.
- 11:50Z (poll 2 of worker 2, at 11:43Z): all eight seeds and every chain / relay / watcher alive. M6-IB 1,409 / 1,346
  of 5,081 (13.2 / 13.8 h; second SELECT700 at 1272: .8740 / .8659, BEST = 1272), M6-IB2 1,387 / 1,382 of 6,614
  (17.4 / 17.5 h), M6-IBX 1,113 / 1,095 of 4,997 (13.7 / 13.9 h), M6-IB2PN-s1 15 of 6,886 (9.56 s per update),
  M6-IB2PN-s2 in its reload preflight. Node D data disk 501 GB. Integration merged at `d96250da6` (amendment 3 +
  tooling); gist 06 entry added; **hand-off record `m6-handoff-2026-10-01.md` written: section 1 requests the
  custodian's C1 content recheck now** (IB1 + IB2 + PN1 roots; `a20ib12pn` covers every arm's rows); interim results
  updated.
- 11:45Z: **M6-IB2PN launched** (see "Running now"): s1 full run on node D GPU2, s2 in its onestep preflight on GPU3;
  chain, relays and mlx watcher alive. Eight seeds in flight. GPU-h ≈ 22.0 (closed 1.136 + 0.089 hedge s1 preflights +
  running ≈ 20.8).
- 11:30Z: hedge tooling `c0056edca` (137 tests pass; shellcheck clean) mirrored to node A / B / D; node B build
  11:20–11:26Z: two builds identical, amendment 1's four mixtures reproduced byte for byte, `a20ib12pn` 110,173 rows,
  6,886 updates, `8f5425c5…`, projection 19.09 (cap 22), C1 source check clean. Data-lock addendum committed. Next:
  node D staging, then launch (s1 on GPU2, s2 on GPU3 after s1's reload preflight).
- 11:20Z: **amendment 3 committed (G5 hedge M6-IB2PN, before any hedge job).** Node A scans done: 0 hits on all 13
  gold-free panels and PN1 dev; 4 near hits on Index suite rows in 3 PN1-r2 TRAIN groups (3 rows), dropped → PN1H
  (4,361 rows). Next: tooling (`m6_data drop-groups`, `m6-build-pn.sh`, arm / launch / gates for M6-IB2PN, cap 22,
  es / fr report), node B build, data-lock addendum, node D staging, launch on node D GPU2 then GPU3.
- 11:03Z (worker 2, poll 1): M6-IB 1,147 / 1,105 of 5,081 (projection 13.2 / 13.7 h), M6-IB2 1,127 / 1,126 of 6,614
  (17.4 / 17.5 h), M6-IBX 863 / 851 of 4,997 (13.8 / 14.0 h). Six containers, three chains, four node A watchers and
  two node D relays alive; node D data disk 479 GB used. **GPU-h ≈ 17.8** (closed 1.136 + running ≈ 16.7).
  G5 hedge (M6-IB2PN) assessment started: PN1-r2 TRAIN (`c1cec06b…`, 4,364 rows, cleared for released models in
  `m4-dq-results-2026-09-30.md`) shares 0 ids, groups, input hashes, Tatoeba sentence ids or sentence texts with
  PN1 dev; G0 Index-row scan 0 groups; short-text scan vs IB1 + IB2 DEV, SELECT700, CAL698 0 hits (node A panel /
  Index scan running). Node D GPU2–7 released by IX1 and idle.
- 10:41Z (worker 1's last poll): M6-IB 999 / 966 of 5,081 (13.1 / 13.5 h; SELECT700 at 636 .8396 / .8331), M6-IB2 985 /
  978 of 6,614 (17.2 / 17.3 h; at 827 .8668 / .8856), M6-IBX 728 / 715 of 4,997 (13.5 / 13.7 h; at 625 .8113 / .8596).
  Six containers, three chains, four node A watchers and two node D relays alive. **GPU-h:** closed receipts 1.136
  (node B 0.864, node A 0.094, node D 0.178) plus running full runs ≈ 14.6 → ≈ 15.7 used; projection ≈ 94 before any
  private Index run.
- 10:22Z: M6-IB 868 / 846 (13.1 / 13.5 h); M6-IB2 861 / 850 (17.2 / 17.4 h), first SELECT700 at 827: s1 .8668, s2
  .8856; M6-IBX 616 / 605 (13.5 / 13.7 h). Six containers, three chains alive.
- 09:54Z: M6-IB 687 / 674 of 5,081 (13.2 / 13.5 h); first SELECT700 family macro at update 636: s1 .8396, s2 .8331
  (checkpoints written, BEST = 636). M6-IB2 694 / 679 of 6,614 (17.1 / 17.5 h); M6-IBX 446 / 440 of 4,997 (13.5 / 13.7
  h). Six containers alive.
- 09:28Z: steps M6-IB 522 / 522 of 5,081 (9.49–9.51 s/upd, 13.4 h), M6-IB2 527 / 519 of 6,614 (9.36 / 9.51, 17.2 /
  17.5 h), M6-IBX 287 / 282 of 4,997 (9.72 / 9.87, 13.5 / 13.7 h); six containers, three chains alive; no checkpoint yet
  (first at 625–827).
- 09:02Z: all six seeds in full runs (M6-IBX preflights passed on node D: reload 0 changes, |Δp| ≤ 6e-8). Steps /
  s per update / projected GPU-h: M6-IB-s1 359 / 9.38 / 13.2, M6-IB-s2 359 / 9.37 / 13.2, M6-IB2-s1 356 / 9.36 / 17.2,
  M6-IB2-s2 353 / 9.48 / 17.4 (node A sped up), M6-IBX-s1 124 / 9.68 / 13.4, M6-IBX-s2 121 / 9.93 / 13.8 (node D).
  Every cap holds with ≥ 2.6 h margin. ETAs: M6-IB ≈ 21:15Z, M6-IBX ≈ 22:15Z, M6-IB2 ≈ 01:20Z (Oct 2); chains then
  read out, gate and run formal + mlx-diag for passers (≈ 1–2 h each). Budget projection ≈ 94 GPU-h before any private
  Index run (≤ 3 × ≈ 9.5). IX1's private M5-L128 diagnostic was read (private folder only); no gate changes.
- 08:37Z: IX1 released node D at 08:20Z; **M6-IBX launched on node D GPU0 / GPU1** (onestep running; admission 79,945
  rows, 0 over limit). All six seeds now train; three chains and five watchers alive.
- 08:27Z: node D staged for M6-IBX (`m6-stage-d.sh`: base tree, T0 tree, data files, `a20ib1x` equal to node B; image
  present; 1 min copy) and amendment 2 committed (`35492ac25`, mirrored to node A / B / D); node B's `on_d` probe of
  node D's relay directory works. Integration merged at `8a7527079`; gist 06 launch entry added.
- 08:20Z: steps 69–75 of 5,081 / 6,614; s/upd 9.20–9.71; projections M6-IB 13.0–13.2 h, M6-IB2 17.1 (node B) / 17.8
  (node A) of cap 20. Interim results record written. Node D support committed (`e11a19b57`: launcher map `d`, launch3
  `m6-d` and free-text released owners, `m6-stage-d.sh`, `pull-d`, `d` seeds in chain / launch; tests pass). Receipts
  0.958 GPU-h before the full runs.
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

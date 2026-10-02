# ~27B M6 state (resume file)

Updated: 2026-10-02 22:15 UTC+8 (14:15Z; **continuation #6 from 13:40Z**: the M6-IBxIB2-m50 release and M8; see its
section first). Earlier: 2026-10-02 17:45 UTC+8 (09:45Z; continuation #5 from 08:44Z: the cross-arm soups and M7).

## Continuation #6 (from 13:40Z): release M6-IBxIB2-m50, M8 on the idle GPUs, M7 watch — read this first

- **Assignment (parent, 13:40Z; COORDINATION 21:20 UTC+8):** (1) release the cross-arm candidate with the larger
  lower bound on top of Vega's then-current `main` (`e60bd8e3`, the org card-only revision; derive from
  `specs/dev2-27b-org.json` and the org gate / decision); (2) new, diverse M8 arms on node D GPU5–7, node E GPU0–2 / 6,
  node B GPU0–1; (3) M7 soups in release form when they land, cross-arm soups with M6 / M8.
- **Choice: `M6-IBxIB2-m50`** (lower bound above m67's; uniform weights; m67 is the fallback). R3: choice / noul /
  score OK (`gates/M6-IBxIB2-m50/types.json`, xarm complete 13:42Z on node B). IF3: 0 item rows over a20ib1 + a20ib12,
  control 200 / 200. Parity of the restaged package 86 / 86, max |dp| 0.0. Values private.
- **Release ops** (`8eff6a15c`, spec / decision `b9297450d`): base spec `dev2-27b-org.json`; current = `e60bd8e3`
  (org gate `e9591a86…`, decision `08d19f0e…`, manifest `23a04016…`, weights M6-IB `e50fb4c1…`); vendor = the formal
  runs' mirror `8351e7c4d`. `make_27bx.py --check` passes on node A (mirror `22b3f0268`): spec `dfa26f1f…`, decision
  `5b2a3018…` (decided 14:03:02Z). Other tiers' mains unchanged since the card Index input (13:35Z).
- **Staged on node A** (`m6-stage-a.sh`, 13:52–14:02Z): 10,840 files equal to node B's; node A's extra
  `gates/contrast.log` (an earlier stage) made the script's strict list compare fail; nothing else differs.
  Card assets rendered (`card-assets.json` `bf3d6d64…`, default generator, banner A).
- **Prerelease running** on node A GPU0 (shared lease `owner.release-27b-27bx`) since 14:08Z, mirror `22b3f0268`;
  log `/data/dev2/logs/27bx-M6-IBxIB2-m50-prerelease-20261002T140804Z.log`. Then `--release` (TF518 digest as M6-IB's)
  once Vega `main` is still `e60bd8e3` and no other Vega release.sh runs. The runtime phase A worker holds its Vega
  runtime-only revision until this release lands (its record §5).
- **M8 launched** (prereg `m8-prereg-2026-10-02.md`, `22b3f0268`): M8-IB-s3 node B GPU0, M8-IB2-s3 node B GPU1 (14:06Z),
  M8-IB14ML-s3 node D GPU5 (14:07Z); M8-IB14-s4 / -s5 (node D GPU6 / 7) and M8-IB124-s4 / -s5, M8-IB-s4, M8-IB2-s4
  (node E GPU0 / 1 / 2 / 6) start once node D / node E staging ends (`m6-stage-d.sh`, `M6_STAGE_NODE=e`).
- **M7:** 5 / 5 seeds on node D GPU0–4, untouched (caps end ≈ 03:20Z IB14ML, ≈ 06:18Z IB124ML).
- **14:28Z:** all nine M8 seeds past admit / onestep / reload (M8-IB2-s4 last) and in `full`; node E staged
  (base, T0, data files, `a20ib124`, `a20ib1`, `a20ib12`; a second staging run first failed on a script edited while it
  ran, then passed from a copy). Relay watchers (`48e4d4807`, `m6-relay.sh` for M7 / M8 seeds) run on node D for the
  five M7 and three M8 seeds and on node E for its four; node B's two M8 seeds soup in place. Prerelease in scored-panel
  parity since 14:17Z. M8 prereg amendment 1: the M7 cross-arm candidates and their order, fixed before any M7 result.
- **14:53Z:** prerelease passed pre-upload examples, card example, four-panel scored parity (14:17–14:41Z) and remote
  code; AutoModel parity since 14:50Z. M7 at steps 2,147–2,193 (14:32Z, 9.5–9.9 s / update): IB14ML ends ≈ 00:50Z;
  IB124ML projects ≈ 06:10–06:20Z against caps 06:17–06:19Z (a capped seed keeps its BEST among saves ≤ 6,993).
  TF518 site digest `93df9002…` (unchanged since 10-01 07:52, the 4B / 9B / M6-IB releases' site).
- **15:18Z: prerelease PASSED** (14:08–15:17Z, node A GPU0): build, two example processes, repeat, card example,
  four-panel scored parity (typed-final 1,600, css15 6,547, public231 231, mlx-diag 2,275), remote code, AutoModel
  parity on every scored prompt, `verify_bundle` ok (30 files, identity `58469731…`); manifest `a0d3129f…` (no upload).
  **Release launched 15:17:45Z** (`release27bx.sh M6-IBxIB2-m50 --release --gpu 0`, mirror `aaf270df1`): Vega `main`
  was `e60bd8e3` right before; storage 58.17 GB of 100. The 9B M10 KIB4-a40 release (Lux-9B, another repository) runs
  on node A at the same time, which COORDINATION 15:35 allows (headroom ≥ 10 GB; both uploads fit).
  Log `/data/dev2/logs/27bx-M6-IBxIB2-m50-release-20261002T151745Z.log`.
- **15:54Z:** release passed pre-upload examples, card example and three-panel scored parity (15:28–15:48Z); remote code
  since 15:48Z, then AutoModel parity and the upload. M8 seeds at steps 577–635 (≈ 9.5 s / update). Tooling `1ffd480fb`:
  the Index / release-form scripts take the preregistered M7 / M8 candidate names (161 27B tests pass).
- **16:17Z: the upload FAILED on the HF private storage limit** ("Private repository storage limit reached"), after every
  pre-upload check passed (examples, card, three-panel parity 15:28–15:48Z, remote code, AutoModel parity 15:58–16:17Z).
  Nothing was committed: Vega `main` is still `e60bd8e3`. Storage went 58.17 → 95.56 GB of 100 between 15:17Z and
  16:25Z: new private `vllm-sr/Vela-2.0-9B` (17.96 GB) and `Vela-2.0-4B` (9.73 GB), created 15:21Z, and Nox-4B's
  unpurged superseded weights (≈ 9.7 GB; new `main` `d55528d1`). The package is 14.98 GB (rank-512 FP32 soup).
  COORDINATION note 2026-10-03 00:30 asks M17 to purge and the coordinator for the rest. **Retry watcher** on node A
  since 16:30Z (`retry27bx.sh`, `575a3d811`): every 5 min, once `hf_headroom.sh` shows ≥ 20 GB and `main` is still
  `e60bd8e3` with no other Vega release.sh, it runs `release27bx.sh M6-IBxIB2-m50 --release --gpu 0` once (log
  `/data/dev2/logs/27bx-M6-IBxIB2-m50-retry-20261002T163029Z.log`). Failed work dir:
  `dev2-27b-27bx-M6-IBxIB2-m50-release-20261002T151745Z` (1.0 GPU-h).
- **16:55Z:** storage still 95.56 GB (watcher waiting). M7 at steps 3,009–3,076; M8 at 924–1,015; no relay yet.
- **17:37Z:** private storage fell to 49.82 GB (USER 01:11 UTC+8: **Decision 2.0 goes public**; Kai, Eos, Sol, Nox and
  Lux are public, and public repos don't count). The watcher started the release at 17:20Z (`release27bx.sh --release`,
  in scored parity since 17:31Z). **After `post_checks=ok`: make Vega public, then the collection, and check an
  anonymous README 200** (COORDINATION 01:25). Before the next 27B upload the hub guard must accept the six public
  Decision-2.0 repos.
- **Node E left the pool at 17:22Z** (returned to the user). The four M8 seeds there (M8-IB124-s4 / -s5, M8-IB-s4,
  M8-IB2-s4) and their relays were stopped; the coordinator copied their run directories to node F (checkpoints at
  848 / 848 / 636 / 827, with optimizer state). Plan (COORDINATION 01:25: resume or restart on node F): **exact resume**
  from those checkpoints with `m8-arm.sh` `M8_RESUME=1` (attempt `full-r2`; `14d48a005` adds node F GPU2–7 to the
  launcher, `pull-f`, `RELAY_NODE=f`, `M6_STAGE_NODE=f`). Node F GPU2–5 leased for 27B (reserved-idle); staging
  node F since 17:35Z. The node E attempts cost ≈ 4 × 3.1 GPU-h before the stop.
- **18:01Z:** node F staged (17:36–17:41Z; base, T0 and data files equal to node B's, `a20ib124`, `a20ib1`, `a20ib12`).
  The four seeds resumed at 17:40:44Z on node F GPU2–5 (`full-r2`, exact resume from 848 / 848 / 636 / 827) and are
  past their checkpoints (958 / 956 / 645 / 937); relays run on node F (`RELAY_NODE=f`). Release: pre-upload checks
  passed through remote code; AutoModel parity since 18:00Z, then the upload.
- **18:27Z: UPLOADED** — Vega-27B revision **`5c85c127828f4b5dfe0ca95d933be033a8a4caa3`** (≈ 18:17–18:20Z); real Hub
  download and tree check passed, post-download examples passed; remaining post checks (remote code on the download,
  card from the Hub under 5.17 / 5.18, scored parity, readback, collection, gate seal, card HTTP, links, gate evaluate)
  and then the purge of M6-IB's weights are running. Earlier: 2026-10-02 10:00 UTC+8 (02:00Z; **worker 5 = continuation #4, 355ad916, from 01:32Z**; worker 4 =
continuation #3, 4a20f83f, 19:37–20:10Z, silently stopped after its 20:10Z poll; its entries below say "cba71646",
which is the coordinator's ID; worker 3 0d2e488f ran 15:12–19:40Z; worker 2 a56025bb 10:44–15:15Z; worker 1 11741ee2
06:17–10:55Z). **Amendment 6 (`6bbb2512d`): the MLX-DEV2 guard for every M6 finalist** (COORDINATION 04:15).

## Gap 20:10Z–01:32Z reconstructed from the chain logs (worker 5) — read this first

No M6 worker was active; the detached chains, relays, watchers and the guard kept running.

- **M6-IB (chain `b74685ddb`, node B):** seeds ended 21:30Z / 21:46Z, **BEST = 5081 for both** (final SELECT700 .9338 /
  .9392). Soup 21:46–22:15Z (`e50fb4c1…`, max rel. diff 9.0e-7), readout 22:15–22:36Z (P_dev 80.76, T_dev .9656,
  H_dev2 .5770), slices 22:36–22:44Z, **dev gates 22:44Z: G1–G6 all pass → finalist** (`DEVGATES-20261001T224432Z`).
  Formal 22:44–23:19Z: CAL698 not adopted (worsened `css_pilot_ece_15`, `typed_dev_brier`) → T = 1; smoke 8 / 8 per
  panel; typed-final 1,600, css15 6,547, public231 231; autotune added 0; seal `0f79a8af…`; 0.510 GPU-h. mlx-diag
  23:19–23:28Z (2,275 rows), pushed, scored and paired on node A, pulled 23:32Z: **card −.0020 [−.0118, +.0074]
  (pass)**. Gates 23:32–23:34Z, overlap, **verdicts 23:34Z** (`VERDICTS-20261001T233349Z`): items 2–7 pass, **item 1
  FAILS (v3 73.16, +0.80 [−1.59, +2.80] vs A20r)**, beats-AutoJev fails (+1.03 [−1.21, +4.92]); `m6 chain complete`
  23:33:49Z. Upper bound > 0 → **M6-IB is an Index-path candidate (amendment 5 (a))**; its Index run was due at
  23:34Z and did not start (no worker).
- **M6-IBX (chain `d8edcf4e1`, node B; seeds on node D):** pulled 22:36Z (BEST 4375 / 4997, SHA-256 lists equal),
  soup 22:36–23:03Z (`051eec70…`), readout 23:03–23:24Z (P_dev 78.71, T_dev .9013), slices 23:24–23:33Z, **dev gates
  23:33Z: G5 FAILS** (PN1 clean gold-no yes Δ **+.0012** [−.0092, +.0109] > 0; the rule is a point estimate ≤ 0);
  G1–G4 and G6 pass → no finalist, `M6-IBX.SKIP`, chain complete (no finalist) 23:32:59Z.
- **Contrast guard never acted (bug, found by worker 5):** it waited for a `finalists` key that `m4_contrast` writes
  only with `--gates`, so M6-IB's `gates/contrast.json` (23:33Z) stayed, and M6-IB2's chain would have stopped at its
  contrast step. Fix `764f03321` (reads `score_levels`, test added; 148 27B tests pass), mirrored to nodes A–D; **new
  guard node B PID 3068520** (chains 2769422, 2818836; log `m6/logs/contrast-guard-764f033.log`) moved the file to
  `contrast-M6-IB-20261002T014456Z.json` at 01:44:56Z; the old guard 2868454 was stopped at 01:45Z. `m6_report
  devgates` also failed on G6's `{candidate, reference}` `B_dev`; fixed in the same commit.
- Nothing failed; no stage needed a rerun. Node D GPU4–7 owner files were gone at 20:26–20:27Z (not M6's doing);
  node C GPU1–7 hold the M16 Index worker's `track=eval-ix1` owners (its runs ended 20:35–21:03Z; idle).

## Rule change at 01:55Z: Index-first (amendment 7, `7ad871bac`) — supersedes the hand-off lists below

- **COORDINATION 09:55 (user decision):** the release gate is the private Index delta vs the current release with a
  95% lower bound > 0, plus integrity checks:
  - exact parity and Hub checks;
  - the contamination audit;
  - no type collapsed on formal typed FINAL.

  Items 1–8, v3, human transfer, mlx-diag / MLX-DEV2 and public 231 become references. **C1 item 8 is not run.**
  Selection goes to the largest lower bound.
- **COORDINATION 10:00:** this track (355ad916) owns 27B's Index-first sweep.
- **COORDINATION 10:35:** card worker fb5dd490 publishes a banner-A card-only revision of Vega-27B now, because no
  27B release is imminent. **Re-read Vega's `main` right before any upload.**
- **27B pool (fixed in amendment 7):**
  - M6-IB (running);
  - M6-IB2;
  - M6-IB2PN;
  - M5-L128 (an existing IX1 run; only a CPU bootstrap is new);
  - M6-IBX last, if the ≤ 40 GPU-h allowance still covers it.
- **Tool:** `python3 -m v2.27b.m6.m6_index_first --boot NAME=<private>/paired-boot-vs-a20r.json ... --types
  NAME=<gates>/<NAME>/types.json ... --family-delta ... --public <record>.json --private
  ~/code/decision2-program/private/m6/index-first.json` (`eb072ecb1`). It prints pass / fail, the order and the
  choice only.

## Running now (worker 5)

- **M6-IB Index path (amendment 5 (a), amendment 6 order):** `m6-index.sh 764f03321… M6-IB stage` 01:46Z (node D
  package `models/ix1/m6/M6-IB-re876fbe`, manifest `68802e7c…`, identity `e50fb4c1…`, loaded 27,497,508,864, T = 1; 31 /
  31 listed digests checked by hand because the local script was killed with its tool call after the restage wrote the
  manifest), `stage-c` 01:54–01:56Z (node C package equal file for file), **parity** on node D GPU4 since 01:51Z,
  **MLX-DEV2** readout on node D GPU5 01:51–02:08Z (`private/eval/mlx-dev2/runs/M6-IB`, with the mlx-diag prompts and
  cache `f474e2e9…`; 7,490 answers, 961 s; predictions copied to node A, SHA-256 equal). **Parity PASS** (86 / 86
  `ok`, max |Δp| 0.0). **MLX-DEV2 guard PASS** (node A `mlx_dev2 compare` vs A20r's validation predictions, code
  unchanged since the validation): card-eligible **+.0029 [−.0021, +.0079]** (Choice +.0083 [+.0012, +.0157], Noul
  −.0025 [−.0096, +.0043]); the same container's mlx-diag reading −.0020 [−.0112, +.0077] reproduces the formal
  item-4 delta (−.0020).
  - **Index run started 02:11Z:**
    - node D: shards 0–3 on GPU4–7 (`m6-index-run` PID 848640; log `ix1/logs/m6-index-M6-IB-d.log`);
    - node C: shards 4–7 on GPU1–4, started by hand at 02:12:30Z (PID 906277), because `m6-index.sh run` launched
      only the first node of a split plan (fixed in `5343df87c`).
  - Shard 0's first 300 rows took the same model time as M5-L128's (493 s vs 494 s: long inputs come first). Shards
    take ≈ 65–73 min, staggered by 300 s, so both nodes end ≈ 03:35–03:40Z.
  - Then: `collect` (node C → node D), `score`, `release`.
  - Node C GPU5–7 are free (the 0.8B / 2B Index worker b49d1f36 released node C GPU1–7 at ≈ 02:10Z). M6-IB2's
    MLX-DEV2 readout takes one of them.
- **M6-IB2:** s1 complete 01:46Z (BEST 4135), s2 (node A) at 6,594 / 6,614 at 01:48Z; chain 2769422 waits for the
  relay. **M6-IB2PN:** 5,200 / 6,886 at 01:48Z, ≈ 10.6 s per update now → ETA ≈ 06:50Z (projection ≈ 19.3 GPU-h per
  seed, cap 22).
- **Incident (harmless):** at 01:51Z one tool call ran twice; the first copy started parity and MLX-DEV2, the second
  stopped at the "log / work dir exists" guards. One parity and one MLX-DEV2 container exist.
Prereg amendment 4 (`c1eafcc08`, the Index path) and **amendment 5 (COORDINATION 03:40: Index runs only for
Index-path candidates and the chosen successor, ≤ 40 GPU-h on node C GPU1–7 + node D GPU4–7; release target
`Decision-2.0-Vega-27B` with the "audited" footnote)** apply from now on.
Prereg `m6-prereg-2026-10-01.md` (`90d38aba7`) with amendments 1 (`b74685ddb`), 2 (`35492ac25`) and **3 (`4e4211aee`,
data lock `7fcbc824c`)**. Mirrors: M6-IB / M6-IB2 seeds, chains and watchers run from **`b74685ddb`**, M6-IBX's from
**`d8edcf4e1`**, M6-IB2PN's from **`7fcbc824c`** (on node A / B / D); step 0 ran from `20af2e4a1`. Hand-off record for
other tracks: `m6-handoff-2026-10-01.md`. Results tables: `python3 -m v2.27b.m6.m6_report` (from mirror `4ee6b920e` or
later on node B).
Assignment: COORDINATION 2026-10-01 14:25 (27B M6, worker 11741ee2). Branch `xunzhuo/decision-2-training-27b`
(worktree `/home/xunliu/code/vllm-sr-dev2-27b`; merge-only into `xunzhuo/decision-2-training`). Gist file
`06-decision-2-27b.md`. Budget 140 GPU-h. Index numbers are private: never in this file, commits or the gist.

## Hand-off (worker 3 → continuation, 19:40Z) — read this first

- **State (poll 12, 19:32Z):** eight seeds train unattended on schedule; four chains, four node D relays, the node A
  relay and four mlx watchers alive; the **contrast guard** (node B PID 2868454) protects the shared `m6/gates/` (see
  "Infrastructure"). No candidate, gate or formal result exists yet. GPU-h ≈ 85 of 140 (projection ≈ 134); eval
  allowance used 0.056 (the restage control). Worker 3's last poll commit landed at 19:34Z (`a6d148235`): the next
  poll commit is due by 20:19Z (45 min), and 20:34Z is the 60-min silent-stop line.
- **Rules that changed this session** (all recorded before any M6 result):
  - **C1 content recheck r1 PASS, exposure 0 for all four arms** (`v2/eval/records/c1-recheck-r1-2026-10-01.md`,
    verdict `0823a1a8…`): C1 content blocks item 8 for no arm.
  - **Amendment 4 (`c1eafcc08`): the Index path** (COORDINATION 2026-10-02 02:05). Item 1' = v3 not significantly
    below A20r (paired upper bound > 0) **and** a significantly positive paired Index delta vs A20r (`paired_boot`
    lower bound > 0); items 2–8 unchanged; **one Index run per frozen formal finalist, started as soon as its formal
    run is sealed**; choice = classic passers first (prereg order), then Index-path passers by the Index delta's
    lower bound; private transfer-only tracking.
  - **Index allowance:** COORDINATION 23:20 gave ≤ 12 GPU-h (one 27B run, ≈ 9.5 GPU-h); amendment 4 asks the
    coordinator for one run per finalist (up to ≈ 38). Until it answers, run in seal order, the first within 12.
  - **DEV2.0-27B `main` = `e7b4a372`** (card-only revision on `09280791`, 15:41Z) and the **repositories are being
    renamed** (COORDINATION 00:25): the release supersedes the then-current `main` under the new ID.
- **Done once for the Index path:** restage control PASS (86 / 86, max |Δp| 0.0); contamination audit (0 item
  duplicates in every arm's TRAIN). Node D GPU4–6 hold released owners in the format IX1's `launch.sh` accepts;
  GPU7 runs the 4B track's panel until ≈ 22:18Z (its lease).
- **Commits to use:** node B mirror `b980dd144` (m6_report, `m6-gates.sh` `contrast` stage) or later; **node D
  mirror `c1eafcc08`** (M6 `DIAGNOSTIC` entries, `m6-index-run.sh` with a GPU list; older node D mirrors ignore
  `M6_INDEX_GPUS`). Run `m6-index.sh` / `m6-stage-a.sh` / `m6_index_path` from the worktree at `HEAD`.
- **Tool sleeps overran here** (a 20-min sleep ran ≈ 40 min once): sleep ≤ 10–15 min and commit within 45 min. **A
  tool call once ran a command twice** (18:22Z incident): before re-running anything that moves files or starts a
  GPU job, check whether the first copy is still running.

## Next steps (worker 3's hand-off; supersede the older lists below)

1. **Poll ≤ 45 min, commit each time.** M6-IB's seeds end ≈ 21:30Z / 21:47Z, M6-IBX's ≈ 22:22Z / 22:32Z, M6-IB2's ≈
   01:40Z, M6-IB2PN's ≈ 06:25Z (Oct 2); each chain then needs ≈ 1–2 h.
2. **Per chain, at its dev gates** (`m6/logs/<ARM>.CANDIDATE` and a `readouts/DEVGATES-*.json` naming the arm): on
   node B, `cd /data/dev2/src/b980dd1448940dfed4658d15624fd7fd54964e44-src_training_decision2/src/training/decision2 &&
   PYTHONPATH=. python3 -m v2.27b.m6.m6_report devgates <DEVGATES file> --root /data/dev2/runs/27b/m6 --pn1-rows
   /data/dev2/hf-cache/datasets--llm-semantic-router--decision-2.0-training-data/snapshots/27b1d2f130292268b43a618584bebab5d4e4a6b5/m4/pn1/dev/pn1.dev.jsonl`
   → the development table (G1–G6 with intervals; G5 yes-bias and G6 breadth; the es / fr view) in the results
   record; commit.
3. **Per finalist, once `m6/<ARM>/formal/SEAL.json` exists:** start its Index run (runbook §3; ≈ 3 h):
   `SHA=c1eafcc087f488e9e935050d2cfe26d8865c4704; M6=src/training/decision2/v2/27b/m6; bash $M6/m6-index.sh $SHA
   <ARM> stage`, then `parity`, then `M6_INDEX_GPUS="4 5 6" bash $M6/m6-index.sh $SHA <ARM> run` (`"4 5 6 7"` once
   the 4B run has released GPU7), `status` until 8 shards end 0, `score`, `release`. One run at a time on GPU4–7.
4. **Per chain, at `m6 chain complete`:** `m6_report verdicts <VERDICTS file>` → the formal table (items 1–7,
   beats-AutoJev). If a chain log ends in a `FileExistsError` after its gates start, the guard missed: run that
   chain's `overlap` and `verdicts` stages by hand ("Infrastructure").
5. **After the last chain and every finalist's Index run:** on node B `m6-gates.sh <b980dd144 or later> verdicts
   <every sealed finalist>` and `contrast <every sealed finalist>`; locally `python3 -m v2.27b.m6.m6_index_path`
   (runbook §3) → item 1' and the choice. Then runbook §2 (stage on node A with `m6-stage-a.sh <ARM>`, draft spec
   and pre-release build, C1 spec, `c1-postkey.sh` verify / preflight / one run), §4 (release under the new repo ID,
   after re-reading COORDINATION 00:25), §5 (milestone end: `m6-link.sh remove` after the staging, leases, gist 06,
   signed merge).

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

## Next steps (worker 2's hand-off, 15:10Z; superseded by worker 3's list above)

1. Poll (≤ 45 min, commit each time) until M6-IB's chain starts processing (≈ 21:45Z, both seeds done).
2. Per chain: when `m6/logs/<ARM>.CANDIDATE` and a `DEVGATES-*.json` naming the arm exist, run `m6_report devgates`
   (see the hand-off above) and fill the development table (G1–G6 with CIs, G5 and G6 especially, plus the es / fr
   view) in `m6-results-2026-10-01.md`; commit. On `m6 chain complete`, run `m6_report verdicts` and fill the formal
   table; record the choice rule's outcome across chains (stage 1: M6-IB / M6-IBX; stage 2: M6-IB2 / M6-IB2PN).
3. **Superseded by worker 3 (15:50Z; `m6-handoff-2026-10-01.md` is now the runbook):** item 8 is this track's own
   step (§2: one attempt, `c1-postkey.sh` on node A GPU2, only for the choice-rule finalist and only with zero
   exposure in the custodian's §1 record: met for every arm), and so are the private Index runs (§3:
   `m6-index.sh` on node D GPU4–7). **Amendment 4 (Index path, COORDINATION 02:05): one Index run per frozen formal
   finalist, started once its formal run is sealed; item 1' (v3 not significantly below A20r and a significantly
   positive paired Index delta) can replace item 1; `m6_index_path` applies item 1' and the amended choice.** The
   23:20 allowance (≤ 12 GPU-h) covers one 27B run; more are requested from the coordinator. The release (§4)
   supersedes DEV2.0-27B's current `main` through the release pipeline, only if items 1–8 (or 1' and 2–8) pass:
   **`main` moved to `e7b4a372` at 15:41Z** (card-only revision on top of the fix revision `09280791`; spec
   `dev2-27b-card.json`), and the repositories are being renamed (COORDINATION 00:25: use the new ID).
4. After the last chain (M6-IB2PN ≈ 08:30Z Oct 2): attribution (`m6-gates.sh <mirror> contrast` and `verdicts` with
   every sealed finalist; never `gates` again for an arm already gated), final
   results, gist 06, milestone-end cleanup ("Hand-off" above), merge (signed) into integration. **Order:** the
   chosen finalist's `m6-stage-a.sh` (node A, item 8 / release) runs before `m6-link.sh remove`, which deletes node
   B's link directory.

## Continuation #5 (from 08:44Z): cross-arm soups of M6-IB and M6-IB2, M7 watch — read this first

- **Assignment (parent, 08:44Z):** (1) cross-arm average of the M6-IB and M6-IB2 seed adapters now, measured once in
  release form (≥ 8 shards on node E GPU0 / 1 / 6 / 7, node D GPU5–7, node B GPU0), bootstrap vs M6-IB, release to
  Vega-27B on top of `781b2b24` if the lower bound is > 0 (integrity: formal typed-FINAL no-collapse, audit coverage,
  package parity, Hub `trust_remote_code` smoke), plus 1–2 cheap weightings; ≤ 25 GPU-h on top of M7's 120. (2) Speed
  up M7 if the trainer supports multi-GPU data parallel. (3) M7 soups in release form when they land.
- **Candidates** (exact weighted rank-concatenation soups of the four seeds M6-IB s1 / s2 @ 5081 and M6-IB2 s1 @ 4135 /
  s2 relay @ 4135; rank 512, alpha 1024; `mNN` = M6-IB2's weight in percent):
  - `M6-IBxIB2-m50` (1/4 each = the mean of the two arm soups): `m6-tail.sh lsoup` from mirror `ceefc7863` (the
    uniform tool, unchanged), node B CPU, since 08:51Z.
  - `M6-IBxIB2-m67` (1/6, 1/6, 1/3, 1/3: 2:1 toward M6-IB2): `SOUP_WEIGHTS="1 1 2 2"` from `e5b63a396`, since 09:12Z.
  - The soup tool verifies the weighted mean update of every projection (≤ 1e-6 relative) and the weighted head.
    Every a20ib1 row is an a20ib12 row (81,294 ⊂ 105,812 distinct rows), so the soups' training rows are a20ib12's.
- **Tooling** (`e5b63a396`, `973587a17`, `7042f3d84`; 27B suite 160 tests on node B, one pre-existing failure
  `test_m1_tools test_rejects_foreign_gpu` that also fails on `ceefc7863`):
  - `lora_soup --weight`; `m6-tail lsoup SOUP_WEIGHTS / SOUP_CPUS`.
  - `m6-index.sh` / `m6-index-run.sh` take `M6-IBxIB2-mNN` and `M7-IB124ML` / `M7-IB14ML` (IX1 DIAGNOSTIC entries
    added), the loaded count from the soup's rank (rank 512: 29,365,153,792), `M6_PARITY_GPU` (dN / eN / bN; node D
    GPU4 is an M7 seed now), `M6_INDEX_BASE=M6-IB` in `score` (family delta and paired bootstrap vs the current
    release), `audit-arm` over both training files (and the m7-data mixtures), `hold` / `unhold` (a 27B reserved-idle
    owner on idle GPUs of the 27B allocation whose eval-ix1 run has ended all 8 shards; `run` / `parity` set holds aside).
  - `m6-xarm.sh MIRROR NAME GPU` (node B): readout, formal (T = 1 unless CAL698 is adopted, which the release refuses),
    mlx-diag with node A pairing (`m6-mlx-watch.sh` with `MLX_ALSO=M6-IB=…`), gates, and the references vs M6-IB
    (paired v3, public 231, mlx-diag).
  - Release ops `v2/release/records/dev2-27b-xarm-2026-10-02/ops/` (`make_27bx.py`, `inputs27bx.sh`, `render27bx.sh`,
    `release27bx.sh`; `index27b.sh` of the M6-IB record is reused): spec from `dev2-27b-27bif-M6-IB.json`, current =
    M6-IB `781b2b24` (gate `2851727b…`, decision `c7b9b224…`, manifest `4d1ca0f5…`), purge against M6-IB's verified
    download. `CHOICE` is unset until a candidate qualifies.
- **M7 (task 2):** the LoRA trainer refuses `WORLD_SIZE != 1` (`training/model/train.py` `validate_args`: "single-device"),
  so a seed cannot be spread over GPUs; nothing restarted. Measured 09:14Z (≈ 300 updates): 9.78–9.84 s per update →
  **M7-IB124ML ≈ 21.75 h of steps + ≈ 0.22 h overhead (M6 receipts) vs its effective cap 21.91 h**: the three seeds may
  be stopped by `launch.py`'s watchdog in their last minutes; M6's rates fell ≈ 1% after the first hours, which would
  leave ≈ 7–15 min of margin. If a seed is stopped, its `BEST.json` (written at every save) names the best checkpoint
  among saves ≤ 6993 and that is its soup member (no resume: one attempt per arm-seed and the 120 GPU-h budget);
  M7-IB14ML projects ≈ 16.3 h of cap 19 (safe). No room for extra seeds inside 120 GPU-h (≈ 98 seeds + ≈ 20 Index).
- **GPUs:** node D GPU5–7 held for 27B at 09:41Z (the 4B SDMLxS17-m50 run had ended); node E GPU0–3 / 6–7 run the 4B
  M17 pool's Index runs (eval fast lane, since 09:05–09:32Z), a workstation loop holds each as soon as its run has
  ended all 8 shards; node B GPU0 (27B reserved-idle) for shards, GPU1 / 5 (no owner) for the formal path.
- **Node A mlx watchers** for both names (PIDs 925204 / 925205, mirror `7042f3d84`, `MLX_ALSO=M6-IB=…`).

## Hand-off (worker 4 355ad916 → continuation, 08:45Z)

- **Done:** M6-IB released (`Decision-2.0-Vega-27B@781b2b24`, record
  `v2/release/records/dev2-27b-indexfirst-2026-10-02.md`); gist 06 and a COORDINATION note written. The current Vega is
  M6-IB: the next 27B release needs an Index lower bound above 0 **vs M6-IB** (not A20r).
- **Running (detached, node D, mirror `ceefc7863`):** GPU0–2 M7-IB124ML s1 / s2 / s3, GPU3–4 M7-IB14ML s1 / s2.
  Logs `/data/dev2/runs/27b/<ARM-SEED>/driver.log`. Expected ends ≈ 00:20Z (IB14ML) and ≈ 04:55Z (IB124ML) on
  10-03. No chain exists for M7 yet: once each seed's `receipts/full.json` exists, its lease goes `reserved-idle`.
- **Next steps:**
  1. **Measure the run-rate** after the first saves (≈ 1 h after launch): if total M7 use projects past 120 GPU-h, stop
     M7-IB124ML-s3.
  2. **Cross-arm average (coordinator finding of 15:55 UTC+8)**: M6-IB s1 / s2 + M6-IB2 s1 / s2 as one 4-member soup
     (rank 512, equal weights = the mean of the two soups). Needs: `lora_soup` with 4 members, a new loaded-parameter
     count in `m6-index.sh` stage, the ARM regex extended, R3 types, an audit-arm over both mixtures, then one Index
     run (≥ 8 shards). Shards fit on node D GPU5–7 + node E GPU0 / 1 / 6 / 7 (`stage-e`) + node B GPU0. Cheap
     (CPU merge, ≈ 9 GPU-h Index); counts against M7's 120.
  3. **M7 soups + Index:** soup each arm (3 members → rank 384, 2 members → rank 256; loaded counts change), restage,
     Index vs M6-IB in release form, release the best with lower bound > 0 (`release27b.sh` needs a new spec and
     decision; main must equal `781b2b24`, superseded node copy is M6-IB's).
  4. Shared Index pool (COORDINATION 15:50): node C GPU1–7 and node A GPU0 / 3–6 are also allowed for Index shards.
- **Leases:** node D GPU0–4 are 27B `running`; node D GPU5–7 eval-ix1 released (free); node B GPU0 27B reserved-idle;
  node E GPU0 / 1 / 6 / 7 free (no owner); never node E GPU4–5. Node A GPU0's shared release lease is removed.
- **Parked:** M6-IB2PN (soup exists; chain stopped at its readout), the M7 warm-start hedge.

## Poll log (newest first)

- 13:40Z (hand-off, continuation #5 stops on the coordinator's interrupt): both cross-arm Index runs are scored
  against M6-IB (values private). The candidate is `M6-IBxIB2-m50`, with m67 as the fallback. `m6-xarm.sh` for both is
  still running on node B GPU0 and GPU1, waiting for node A's mlx-diag pairing. The card Index input and the release
  inputs for m50 are on node A. Not run yet: stage-a, render, spec / decision and release. Vega's `main` is now
  `e60bd8e3` (org card-only revision), so the superseded revision in `make_27bx.py` and `release27bx.sh` must change.
  The exact next steps are in the private hand-off file `private/m6/HANDOFF-27B-2140.md`. M7: 5 / 5 seeds untouched.

- 10:30Z (poll 28, continuation #5):
  - **Soups:** the first m50 soup failed its own check (FP32 accumulation over four members > 1e-6 on 28 projections);
    `8351e7c4d` checks the factor layout bit for bit plus a float64 update check on 256 rows per projection. Both soups
    rebuilt from that mirror 09:59–10:04Z: m50 identity `58469731…`, m67 `3d120c67…`, rank 512, loaded 29,365,153,792.
  - **Audit-arm** (IF3): both 0 items over a20ib1 + a20ib12, planted control 200 / 200.
  - **Index:** restaged on node D, B and E; parity 86 / 86 ok, max |dp| 0.0 (node D GPU5 / 6). m50 runs 8 shards on node D
    GPU5–7 + node B GPU0–1 since 10:24Z; m67 8 shards on node E GPU0 / 1 / 2 / 6 since 10:27Z (two rounds each,
    ≈ 2.3 h). Node C and node E GPU3 / 7 were busy with the 4B and 9B Index runs; node D GPU2 is M7-IB124ML-s3.
  - **Org rename:** integration merged (`f72f4de36`, the vllm-sr pipeline); the cross-arm release ops name
    `vllm-sr/Decision-2.0-Vega-27B`, the spec derives from `layout.current_ids` of M6-IB's spec, `index27bx.sh` and an
    own `purge_superseded.py` on vllm-sr (`3cdccffde`).
  - **M7:** 5 / 5 alive, steps 728–752 at ≈ 9.82 s/update: IB124ML ends ≈ 06:10Z 10-03 (cap ≈ 06:19Z for s3, tight),
    IB14ML ≈ 00:30Z.
  - **GPU-h:** cross-arm parity 0.3, Index running (≈ 18.5 projected); M7 ≈ 11.

- 09:45Z (poll 27, continuation #5): both soups in their verification pass (node B, 16 CPUs each); M7 5 / 5 seeds alive
  (≈ 300–330 updates; rates above); node D GPU5–7 held; node E busy with the 4B pool; audit-arm of `M6-IBxIB2-m50`
  copying a20ib12 to node D. GPU-h: cross-arm 0 so far (CPU only); M7 ≈ 6.9 (5 seeds × ≈ 1.4 h) of 120.
- 08:25Z (poll 26):
  - **M6-IB release:** uploaded (07:59Z); post-download examples / parity on the real Hub download since 07:59:44Z,
    then post checks and the purge (≈ 09:05Z).
  - **M6-IB2 vs M6-IB:** paired bootstrap done (private); not a release candidate over M6-IB. IB2's families go on in
    M7-IB124ML.
  - **M7 build:** `ceefc7863` (ML records `ml-K-MIX.json` outside the compared dirs); build 08:05–08:11Z on node B,
    two builds identical, C1 clean for both mixtures. `a20ib124ml` 127,819 rows / 7,989 updates (projection 20.6 GPU-h),
    `a20ib14ml` 94,099 rows / 5,882 updates (16.0 GPU-h); ML share restored to a20's (.386). Staged to node D
    (`m6-stage-d.sh`, all hashes equal). Incident (harmless): a retried edit command reported a failed assertion after
    the first copy had applied the fix; the rebuild was stopped once and restarted from the same mirror.
  - **M7 launched** 08:16–08:20Z from mirror `ceefc7863`: node D GPU0–2 M7-IB124ML s1 / s2 / s3 (save every 999, cap
    22), GPU3–4 M7-IB14ML s1 / s2 (save every 736, cap 19); all five past admit. Node D GPU4 taken from eval-ix1's
    released owner (`owner.prev-ix1-*`). Node D GPU5–7 left released for the Index runs.
  - **Budget (120 GPU-h):** ≈ 94 GPU-h of seeds + ≈ 18 for two Index runs. Stop rule: if the measured rate projects
    past 120, M7-IB124ML-s3 stops and that arm soups two seeds.

- 07:56Z (poll 25; COORDINATOR UPDATE 15:50 UTC+8: 27B goal top 3 of 15–40B, node D GPU0–7 + node E GPU0–3 / 6–7 +
  node B GPU0 for 27B, M7 budget 120 GPU-h):
  - **M6-IB release:** pre-upload parity passed again; AutoModel parity since 07:39:51Z, then upload.
  - **M6-IB2 Index:** all 8 shards ended exit 0; collected (node B shards 2 / 6 / 7 → node D, SHA-256 equal); leases
    released (node B GPU0 back to its 27B owner; GPU1 / 5 are b49d1f36's again); `score` running (paired bootstrap).
  - **M6-IB2PN:** soup done (`edfcee97…`, rank 256); the chain stopped at its readout (node B GPU1 was eval-ix1's
    then, now the 2B formal's). Resume it with `AUX_M6_IB2PN=0` on node B GPU0 (restart-safe; PN1 / IB12 env from
    `m6-launch.sh`) or skip: it is a hedge.
  - **M7 started (data):** `m7/m7_data.py` (ML block), `m7/m7-build.sh` (`b3ca2ae38`: IB1 without `sentfin`, 17,517
    rows; IB4 phase 1 `6045b456…` 9,459 rows; `a20ib14` / `a20ib124` + ML copies → `a20ib14ml` / `a20ib124ml`), build
    running on node B since 07:53Z. `m7/m7-arm.sh` (node D GPU0–7, node B GPU0; seeds s1–s3; cap 22) and
    `m6-stage-d.sh M6_DATA=m7-data`. Node D GPU0–7 are idle for M7.

- 07:08Z (poll 24):
  - **M6-IB prerelease PASSED** (node A GPU0, 05:57–07:02Z; work `dev2-27b-27bif-M6-IB-prerelease-20261002T055720Z`):
    build, two example processes, card example, scored-panel parity on typed-final / css15 / public231 / mlx-diag,
    Transformers remote code, parity through AutoModel, `verify_bundle` ok (30 files, identity `e50fb4c1…`).
  - **Release launched 07:02:33Z** (`release27b.sh M6-IB --release --gpu 0`, mirror `69b52757d`): Vega-27B `main` was
    `b689ee66` right before; no other `release.sh` on node A; TF518 digest set. In pre-upload examples.
  - **M6-IB2 Index:** shards 0–4 ended exit 0 (3 / 4 at ≈ 06:26Z, 2 at ≈ 06:35Z); 5 (node D GPU0, since 06:26Z), 6 / 7
    (node B GPU5 / GPU1) running. ETA all 8 ≈ 08:45Z (shard 5).
  - **M6-IB2PN:** both seeds ended ≈ 06:30Z; chain `2818836` (node B) in `lsoup` since ≈ 06:47Z.
  - **M7 plan** committed (`2559343d9`, `records/m7-plan-2026-10-02.md`): arms M7-IB24 / M7-IB14 (+ IB4 phase 1 block,
    IB1 `sentfin` dropped), warm-start hedge pending approval; M7 needs a new GPU-hour allocation.

- 06:14Z (poll 23):
  - **M6-IB release (interrupt item 1) in prerelease.** Ops `records/dev2-27b-indexfirst-2026-10-02/ops/`
    (`5ac8896d4`, `17bb3ca16`, `69b52757d`): `inputs27b.sh` (node A pulls the private Index files from node D, SHA-256
    equal at both ends), `index27b.sh` (card Index chained from the newest released input matching every other tier's
    Hub main; it picked the Nox-4B Index-first input, Nox `main` now `b285e7a1`; only the 27B point changed),
    `render27b.sh` (node A, the render env of `render_env.sh`, banner A), `make_27bif.py`, `release27b.sh`,
    `purge_superseded.py` (Vega-27B, node copy = A20r's verified download, identity `2e074511`).
    - Audit-arm (IF3) ended exit 0: planted control 200 / 200, `a20ib1` 0 item rows, 2 duplicate rows.
    - Gate profile passes (IF1, R3, IF3, references); `--check` reproduces spec `8f5b3d7c` and decision `c7b9b224`.
    - First prerelease (GPU6, 05:49Z) stopped in build: the new runtime-equivalence and `_release` texts named a node;
      reworded (`17bb3ca16`), CPU build then passed (31 files, loaded 27,497,508,864). GPU6 went back to eval-ix1.
    - **Prerelease running** on node A GPU0 (shared-lease co-tenant per COORDINATION; `owner.release-27b-27bif`)
      since 05:57Z: build, examples and card example done; four-panel parity since 06:06Z. Then `--release` on top of
      `b689ee66` (TF518 site digest computed on node A).
  - **M6-IB2 Index re-split again** (`49433d69f`, `M6_INDEX_AFTER`, released node B owners set aside): old loops
    stopped (containers kept); shard 2 node B GPU0, shards 3 / 4 node D GPU0 / 1 (running since 05:13–05:18Z); **shard 7
    node B GPU1 (06:10Z), shard 6 node B GPU5 (06:12Z)**; shard 5 waits on node D GPU0 for shard 3. Rate ≈ 155
    records / min per shard (≈ 15k per shard): shards 2–4 end ≈ 06:50–07:00Z, 6 / 7 ≈ 07:50Z, 5 ≈ 08:40Z. **ETA all 8
    ≈ 08:40Z**, then collect + score (≈ 20 min).

- 05:16Z (poll 22; COORDINATOR INTERRUPT 12:40 UTC+8, COORDINATION 12:30 / 12:40 / 13:05):
  - **Integration merged** (`8ce3ec8d0`, signed; ≥ `cd565a588`, the collection-check fix).
  - **M6-IB2 Index re-split (≥ 8-way asked; 3 GPUs were free).**
    - Node D's static loops (shards 2–5 queued) were stopped at 04:33Z; shards 0 / 1 kept running and **ended exit 0
      at ≈ 05:12Z**.
    - Node E GPU0–3 / 6–7 were staged (`stage-e`, package + A20r cache + digest file SHA-256 equal) but the 2B M18
      runs took all of them at 04:48Z (two launches failed before a container started: the missing `cache-frozen.sha256`,
      fixed in `d3f228e85`; then GPU0 busy). Node E is the sweep's per COORDINATION 12:40.
    - Node B GPU1 / GPU5 went to the 2B formal collection at 05:03:44Z (COORDINATION 13:05); **node B GPU0 only**.
      `stage-b` (`a073d8be8`: image host2 pulled from node D, ID checked; base snapshot equal; kit, panel-8, package,
      cache SHA-256 equal) done 05:06Z; GPU0's 27B owner set aside as `owner.m6-set-aside-20261002T050651Z`.
    - Node D GPU4–7 went to the 4B worker's LHA10UP / LHA10SDML runs at 05:05–05:08Z.
    - **Now:** node B GPU0 shards 2 → 7 (05:13Z, mirror `6d1bfee10`); node D GPU0 / GPU1 shards 3 → 5 / 4 → 6
      (05:14Z). **ETA ≈ 07:40Z** for the last shard, then `collect` (node B → node D) and `score`.
  - **M6-IB release (Index-first, qualifies):** preparing the 27B Index-first ops from the 4B worker's
    (`dev2-4b-indexfirst-2026-10-02`) and A20r's (`dev2-27b-a20r-release-2026-09-30`) drivers; base = card4 spec,
    current = `b689ee66` (card4 gate and decision); all steps on nodes.
- 04:25Z (poll 21):
  - **M6-IB scored** (`m6-index.sh 14ec16de5 M6-IB score`): 120,226 / 120,226 ok, scorers pass, 0 flagged, 8.95
    GPU-h; private outputs in `private/m6/M6-IB/`.
  - **`m6_index_first`: M6-IB passes the Index gate, types OK → eligible; choice so far = M6-IB.** M5-L128 fails.
  - **Index path: M6-IB passes items 1' and 2–7.**
  - A20r's `merged-budget` had no ix1 receipt.
    - `v2.eval.ix1.merge` over the same files gives `merged-budget-r` (records equal; 2 rerun lines differ in
      serialization only), with a receipt (`2e074511…`, panel `6455d7be…`, 120,226 ok).
    - M6-IB's bootstrap is being recomputed against it (`runs/M6-IB/paired-boot-vs-a20r-r.json`, since 04:14Z), so
      that gate IF1 can bind both receipts. `score` uses it from `c44db3447` (mirrored to nodes B–D).
  - **`m6-stage-a.sh M6-IB`** staged it on node A (10,786 files, SHA-256 lists equal; 04:18–04:22Z) while the link
    is up.
  - **Vega-27B `main` is `b689ee66`** (03:19Z, the card worker's banner-A card-only revision).
  - M6-IB2 Index: shards 0 / 1 running on node D GPU0 / GPU1.
- 04:00Z (poll 20; the state commit came 60 min after poll 19, during a GPU-contention fix; code commits in between):
  - **Manual formals sealed 03:29Z** (references; T = 1; `m6-gates.sh 199daf794 gates / overlap / verdicts M6-IB2
    M6-IBX`, `VERDICTS` at 03:32Z; the guard moved `contrast.json` at 03:32:28Z):

    | Arm | v3 (T / H) | vs A20r [95%] | vs AutoJev-27B [95%] |
    | --- | --- | --- | --- |
    | M6-IB2 | 72.94 (.924 / .576) | +0.58 [−2.46, +3.24] | +0.81 [−1.93, +5.41] |
    | M6-IBX | 74.10 (.916 / .600) | +1.74 [−1.23, +3.17] | +1.97 [−0.25, +5.42] |

    **Formal typed FINAL: no type collapsed for either arm** (Choice, Noul and Score OK, including M6-IB2, whose
    typed-DEV Noul collapsed). Both stay in the amendment-7 pool.
  - **M6-IB2 MLX-DEV2 (reference; node D GPU0, 03:34–03:51Z):**
    - card **−.0124 [−.0179, −.0069]** (the guard would fail); Choice +.0087, Noul **−.0336**;
    - the same container's mlx-diag reading **−.0308 [−.0438, −.0177]** (Noul −.0514): a material multilingual
      regression to flag to the coordinator.
  - **M6-IB Index run done:** 8 / 8 shards exit 0, 8.95 GPU-h. `collect`: the workstation path to node D ran at ≈ 50
    KB/s, so it was stopped (partial shard 4 → `void/`) and relayed through node B (`14ec16de5`); SHA-256 lists
    equal. `score` running.
  - **GPU contention:** within minutes of M6-IB's shards ending (03:31–03:51Z), every node C GPU and node D GPU4–7 went
    to the 4B Index-first and Index-sweep runs, which use free pool GPUs too (COORDINATION 10:00).
    - M6-IB2's Index run therefore uses **M6's own node D GPU0 / GPU1** (M6-IBX's leases, idle since 22:35Z).
      Their M6 owners are set aside as `owner.m6-set-aside-<UTC>`, with eval-ix1 owners naming M6-IB2;
      `m6-index.sh release` restores them (`fb7935a14`).
    - Shards 0–5 started 03:56:30Z (`m6-index-run.sh 14ec16de5 M6-IB2 d 0,1,2,3,4,5 0 1`, log
      `ix1/logs/m6-index-M6-IB2-d-0to5.log`); three rounds, ≈ 07:35Z.
    - **Shards 6–7 still to launch** on node D GPU2 / GPU3 once M6-IB2PN's seeds have ended (≈ 06:20Z), after the
      same owner set-aside: `m6-index-run.sh <sha> M6-IB2 d 6,7 2 3`.
- 03:02Z (poll 19):
  - **Index-first interim:** M5-L128 fails the Index gate (95% lower bound vs A20r not > 0; types OK), so it is not
    eligible. The public file is in `/tmp` only; values are in `private/m6/interim/`.
  - Manual formals: packages frozen at T = 1 (M6-IB2 `55e0ffb0…`, M6-IBX `051eec70…`); collections running.
  - M6-IBX staged for the Index on node D at 02:59Z (manifest `1d636706…`); `stage-c` running.
  - Integration merged again (`591ec6c8c`): `v2.release.gate` now has the Index-first profile (`index_first:
    {bootstrap, receipt, base_receipt, audit}`; IF1, R3, IF3, references).
  - **Release blocker found:** IF1 binds the base run's ix1 receipt, and A20r's `runs/DEV2.0-27B/merged-budget` (the
    bootstrap base of amendments 4 / 7) has none. Only `merged`, the pre-fix run, has one (identity `2e074511…`,
    panel IDs `6455d7be…`). A receipt for `merged-budget` must be built before a 27B release can pass IF1.
- 02:57Z (poll 18):
  - **M6-IB2's chain** (`DEVGATES-20261002T025243Z`): soup 01:55–02:2xZ, readout, slices; **development gates
    02:52Z:**

    | Gate | Result |
    | --- | --- |
    | G1 collapse | **FAIL**: typed-DEV Noul modal share .9675 (387 False / 13 True) |
    | G4 Noul | **FAIL**: .5125 < .52 |
    | G5 | **FAIL**: clean gold-no yes Δ +.0435 [+.0294, +.0581] |
    | G2 HT-DEV v2 | TIE (+.009) |
    | G3 | pass (T_dev .8781) |
    | G6 | pass (B_dev .9611, +.0597 [+.0470, +.0729]) |

    No finalist; chain complete (no finalist) at 02:52:51Z.
  - **Under amendment 7 the type-collapse check is the formal one.** Formal runs were started by hand on node B at
    02:54:36Z for the integrity check and the references:
    - M6-IB2 on GPU0 (mirror `b74685ddb`, PID 3087635, log `m6/logs/formal-M6-IB2-manual.log`);
    - M6-IBX on GPU5 (mirror `d8edcf4e1`, PID 3087637).

    An Index run follows only for a candidate whose formal typed FINAL shows no collapsed type.
  - M5-L128's paired bootstrap finished (exit 0); private copies are in `private/m6/M5-L128/`.
  - `m6-index.sh` can stage a candidate without a formal package (`M6_INDEX_FROM_SOUP=1`; `199daf794`, mirrored to
    nodes B–D).
  - M6-IB Index: shards at 1.4k–6.4k of ≈ 15k rows (02:49Z).
- 02:40Z (poll 17):
  - Read COORDINATION 09:55 / 10:00 / 10:35 (the Index-first rule, the 27B sweep, the banner card-only revision) and
    merged integration (`256c804c7`).
  - Amendment 7 (`7ad871bac`) and `m6_index_first` (`eb072ecb1`).
  - M6-IB Index run: 8 / 8 shards writing; node C GPU5–7 run the Index sweep's 2B shards (no collision).
  - M5-L128 paired bootstrap vs A20r running on node D (CPU, since 02:30:46Z).
  - M6-IB2 readout running (development gates ≈ 02:57Z).
  - **Plan:**
    - M6-IB2's MLX-DEV2 readout on node D GPU0 (M6's idle lease; M6's owner set aside, then restored) once its
      package is frozen (≈ 03:01Z);
    - M6-IB collect / score ≈ 03:40Z;
    - M6-IB2's Index run on d4–d7 + c1–c4 after that.
- 02:25Z (poll 16):
  - M6-IB: MLX-DEV2 PASS and parity PASS (`d5d505fe8`); Index run on 8 GPUs since 02:11–02:12Z (shards 0 / 1 / 4 / 5
    writing).
  - M6-IB2 soup running since 01:55Z.
  - M6-IB2PN at ≈ 5,240 / 6,886.
  - Closed receipts 93.43 GPU-h (node A 18.01, B 47.44, D 27.98) + running ≈ 29.3 = **≈ 122.7**.
  - Fix `5343df87c` mirrored to nodes B–D. Results record updated (development, formal, MLX-DEV2, incidents).
- 02:00Z (poll 15, worker 5's first; 01:32–02:00Z): integration merged (fast-forward to `3dd2dc72c`); gap
  reconstructed (above); guard and reporter fixed (`764f03321`); amendment 6 (`6bbb2512d`); M6-IB's Index path staged on
  node D and node C, parity and MLX-DEV2 running. Containers: node B M6-IB2-s1 ended, node A M6-IB2-s2 and node D
  M6-IB2PN-s1 / s2 alive. **GPU-h ≈ 121** (89 at 20:10Z + ≈ 30.4 seed-hours since + ≈ 1.6 chain readouts, slices,
  formal and mlx-diag); projection ≈ 135 of 140 (M6-IB2PN to ≈ 06:50Z, two chains' tails, three MLX-DEV2 readouts).
  The chains' own budget check reads only node B receipts and node A relay budgets (22:44Z: "28.535"), so it cannot
  bind; this file's count is the one to watch. Eval allowance used 0.056 of 40.
- 20:10Z (poll 14 at 20:02Z): all alive, guard alive. M6-IB 4,554 / 4,427 (s1 at 4452 .9171, BEST stays 3816; ETA ≈
  21:30Z / 21:50Z); M6-IB2 4,502 / 4,508 (ETA ≈ 01:35Z / 01:55Z, node A's seed at 10.1 s per update); M6-IBX 4,157 /
  4,106 (ETA ≈ 22:20Z / 22:30Z); M6-IB2PN 3,094 / 3,080 (ETA ≈ 06:35Z / 06:30Z). Node D disk 737 GB. GPU-h ≈ 89.
  **Index tooling for amendment 5 (`650465151`, mirrored to node C and node D):** `m6-index.sh` takes node-prefixed
  GPUs (`M6_INDEX_GPUS="d4 d5 d6 d7 c1 c2 c3 c4"`: shard k on entry k mod n), `stage-c` (node B checkpoint → node C,
  restage there, node C's package must equal node D's file for file), `collect` (node C shard results → node D's run,
  SHA-256 lists equal; Triton caches stay), `plan` (dry run), two-node `status` / `release`; `m6-index-run.sh` runs
  one node's shards (`NODE SHARDS GPU...`; placeholder GPU 9 for the other node's shards, as IX1's node C runs did).
  147 27B tests pass, shellcheck clean. **The fix2 template is now on node C too** (relayed node D → node C through
  node B's transfer key in 6 s; 33 files, SHA-256 list equal to node D's). Node C and node D hold identical IX1
  inputs (image `f83b1d10`, kit `87d4650b`, base snapshot, A20r's frozen cache `ec364143…`, panel-8). The
  workstation's path to node D is slow (≈ 3 s per connection, two timeouts); node B → node D, the chains' path, is
  fast (6 / 6 at 0.2 s). `m6-index.sh` now retries connections (`ConnectionAttempts=4`).

- 19:52Z (poll 13 at 19:39Z; worker 4's first): all alive (8 containers; node B drivers, four chains and the
  contrast guard 2868454; node A driver, relay and four mlx watchers; node D four drivers and four relays). M6-IB
  4,426 / 4,282 (s1 BEST 3816 .9196, s2 BEST 3180 .9308); M6-IB2 4,360 / 4,362 (BEST 4135 both); M6-IBX 4,005 / 3,958
  (BEST 3125 / 3750); M6-IB2PN 2,944 / 2,925 (BEST 2583 both). Index GPUs: node C GPU1–7 and node D GPU4–7 all hold
  `track=eval-ix1` released owners and run nothing (node D GPU7's 4B panel ended 18:55Z; node C GPU0 is the K8s pod).
  Integration merged (fast-forward to `68e337ed6`, COORDINATION 03:40). **Amendment 5 committed** (the coordinator's
  answers: allowance ≤ 40 GPU-h; Index runs only for Index-path candidates, i.e. finalists failing item 1 with item
  1'(a) and items 2–7 passed, and for the chosen successor's card; node C + node D GPUs, shards may be split; choice
  order confirmed; release to `Decision-2.0-Vega-27B` after the card fix round, footnote "…Training data audited at
  row level against all Index test items."). GPU-h ≈ 86.

- 19:36Z (poll 12 at 19:32Z; worker 3's last): all alive, guard alive. M6-IB 4,388 / 4,243 (13.5 / 13.8 h; ETA
  21:29Z / 21:47Z); **M6-IB2 4,318 / 4,321 (17.7 / 17.6 h; at 4135 .9198 / .9244, BEST = 4135 both)**; M6-IBX
  3,966 / 3,919 (13.9 / 14.0 h; **s2 at 3750 .9163, BEST = 3750**); M6-IB2PN 2,904 / 2,887 (18.8 / 18.7 h). No IX1
  container on node D at 19:33Z. **GPU-h ≈ 85** (closed 1.314 + running ≈ 83.5). COORDINATION 03:15 (M15 closed;
  MLX-DEV2 commissioned as the future guard for new breadth arms) changes nothing for M6's preregistered gates;
  note that formal mlx-diag (item 4) is what sank the breadth arms at 0.8B–9B.
- 19:01Z (poll 11 at 18:59Z): all alive, guard alive. M6-IB 4,183 / 4,030 (13.6 / 13.8 h; ETA 21:30Z / 21:47Z);
  M6-IB2 4,115 / 4,115 (17.7 / 17.7 h); M6-IBX 3,763 / 3,724 (13.8 / 14.0 h; s1 at 3750 .8988, BEST stays 3125);
  **M6-IB2PN 2,695 / 2,678 (18.9 / 18.7 h; at 2583 .9149 / .9073, BEST = 2583 both)**. GPU-h ≈ 80.
- 18:45Z (poll 10 at 18:41Z): all alive, guard alive. M6-IB 4,070 / 3,913 (13.7 / 13.9 h; s2 at 3816 .9221, BEST
  stays 3180); M6-IB2 4,000 / 3,998 (17.8 / 17.6 h); M6-IBX 3,657 / 3,612 (13.8 / 14.0 h); **M6-IB2PN 2,583 / 2,573
  (18.9 / 18.7 h; s1 at 2583 .9149)**. GPU-h ≈ 77.
- 18:33Z: **Index-path preparation (amendment 4, `c1eafcc08`, on integration and mirrored to node D).**
  - **Restage control PASS** (once for M6): A20r's own package through the forward-budget runtime
    (`DEV2.0-27B-budget` = `4e89288d` + `e876fbe`), read by its entry point over the 86 compatibility requests on
    node D GPU4 with A20r's frozen cache, equals IX1's A20r kit answers: 86 / 86 `ok`, 419 questions, max |Δp| 0.0
    (`runs/DEV2.0-27B-budget-control/control.json`, private dir); 0.056 GPU-h (eval allowance), 18:22:52–18:26:22Z.
  - **Contamination audit PASS for the card line** (`v2.eval.ix1.contamination`, node D CPU, 18:30–18:32Z, exit 0;
    planted control 200 / 200): `a20ib12pn` (110,173 lines, every arm's rows) and `a20ib1x` (79,945) have **0 item
    duplicates** in the 120,226 panel rows. Familiar text only in BANKING77 (2 rows; 16 rows with an exact leaf, 21
    partial), plus partial hits in API-Bank (25) and CLINC150 (1): A20's base rows, which A20r trained on too.
    HoVer, When2Call, iSarcasmEval, GSM8K and BPoMP are clean. Private copy `private/m6/audit/audit.json`.
  - **Lease format fix:** IX1's `launch.sh` accepts only owner files with a line `track=eval-ix1`; node D GPU4 / GPU6
    held IX1's one-line released owners, which it refused. They were moved to `owner.prev-20261001T182238Z` (worker
    1's convention for GPU0–3) and replaced by released owners in the accepted format. GPU7 runs the 4B track's
    `LHA10SD` panel (its lease ends ≈ 22:18Z).
  - **Incident (no result affected):** a tool call ran one command twice. Its first copy moved the two owner files
    and started the control; the second copy moved the live control's run directory into `void/` and started a
    third attempt, which stopped at a container-name conflict before running anything. The running container kept
    writing into the moved directory (bind mounts follow the directory), so after it ended (exit 0) its directory was
    moved back, the conflict attempt went to `void/`, and the parity comparison was run by hand on the finished
    run's files. Nothing ran twice on a GPU.
- 18:12Z (poll 9 at 18:04Z): all alive, guard alive. M6-IB 3,853 / 3,689 (13.4 / 13.9 h; **s1 at 3816 .9196, BEST =
  3816**); M6-IB2 3,770 / 3,762 (17.6 / 17.7 h); M6-IBX 3,430 / 3,387 (13.8 / 14.0 h); M6-IB2PN 2,363 / 2,343 (18.9 /
  18.7 h). GPU-h ≈ 73. **COORDINATOR DECISION 2026-10-02 02:05 (18:05Z): the "Index path"** (item 1' = v3 not
  significantly below the reference AND a significantly positive paired private-Index delta; items 2–8 unchanged; one
  Index run per frozen finalist; "27B M6 finalists will be judged on both paths"). Amendment 4 records it for M6
  before any M6 result. Integration merged (signed; `paired_boot.py` from the 9B track now on it). Node D GPU7 runs
  the 4B track's `LHA10SD` Index shard (eval-ix1 lease); the 0.8B fast-track released GPU5 at 16:39Z.
- 17:44Z (poll 8 at 17:42Z): all alive, guard alive. M6-IB 3,727 / 3,551 (13.2 / 13.9 h; ETA 21:08Z / 21:50Z);
  M6-IB2 3,628 / 3,621 (17.5 / 17.7 h); M6-IBX 3,293 / 3,251 (13.8 / 14.0 h; **s2 at 3125 .9159, BEST = 3125**);
  M6-IB2PN 2,225 / 2,203 (18.9 / 18.7 h). Node D disk 688 GB. GPU-h ≈ 70.
- 17:20Z (poll 7 at 17:19Z; a 20-min tool sleep ran ≈ 40 min, so this commit is 42 min after the previous one):
  all alive, guard alive. M6-IB 3,574 / 3,409 (13.3 / 13.9 h; **s2 at 3180 .9308, BEST = 3180**; s1 at 3180 .9111,
  BEST stays 1908); **M6-IB2 3,480 / 3,477 (17.7 / 17.7 h; at 3308 .9012 / .8980, BEST = 3308 both)**; M6-IBX 3,152 /
  3,120 (13.9 / 14.1 h; **s1 at 3125 .9053, BEST = 3125**); M6-IB2PN 2,084 / 2,062 (18.9 / 18.6 h). GPU-h ≈ 64.
- 16:42Z (poll 6 at 16:36Z): all alive, guard alive. M6-IB 3,303 / 3,151 (13.7 / 14.1 h; s1 at 3180 .9111, BEST
  stays 1908); M6-IB2 3,214 / 3,214 (17.7 / 17.5 h); M6-IBX 2,900 / 2,861 (13.9 / 14.0 h); **M6-IB2PN 1,821 / 1,795
  (18.8 / 18.8 h; second SELECT700 at 1722: .8672 / .8686, BEST = 1722 both)**. Node D disk 664 GB. GPU-h ≈ 61.
  New COORDINATION notes: **00:25 — the six repos are being renamed** (`Decision-2.0-{Kai,Eos,Sol,Nox,Lux,Vega}-
  {size}`; round-2 cards follow, so `main` moves again; hand-offs use the new IDs); **00:35 — 9B K-a13IB failed
  item 1** (breadth ties on v3), and an **Index-path alternative to item 1** is proposed, pending a user decision.
  Runbook §3 / §4 updated for both.
- 16:20Z: **C1 content recheck r1 PASS** (custodian, record `v2/eval/records/c1-recheck-r1-2026-10-01.md`,
  `36f93b7cc`, verdict `0823a1a8…`): IB1-r3, IB2, PN1-r2 and `a20ib12` expose 0 scored C1 items; the registry lists
  M6-IB, M6-IBX, M6-IB2 and M6-IB2PN with exposure 0, so C1 content blocks item 8 for no M6 arm. Integration merged
  (signed). Runbook §1 / §2 updated. Poll 5 (16:10Z): all alive, guard alive; **DEV2.0-27B `main` moved to
  `e7b4a372` at 15:41Z** (card-only revision from `f85ea4e17` on top of `09280791`); the runbook's base revision and
  spec now name it.
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

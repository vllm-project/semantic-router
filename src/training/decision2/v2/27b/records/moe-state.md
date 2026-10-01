# 27B MoE milestone (MoE-1): state (resume file)

Updated: 2026-10-01 17:50 UTC+8 (09:50Z) by **continuation #2** (worked 06:40Z–). Branch
`xunzhuo/decision-2-training-27b-moe`, worktree `/home/xunliu/code/vllm-sr-dev2-27b-moe`, gist
`06c-decision-2-27b-moe.md`. Assignment: COORDINATION 2026-09-30 23:50 (+ the 2026-10-01 continuation brief: formal
verdicts, private Index run, hand-off if qualified). Prereg `moe-prereg-2026-10-01.md` (amendments 1–4); results
`moe-results-2026-10-01.md` (formal verdicts final); hand-off notes `moe-handoff-2026-10-01.md` (**PENDING**).
**Continuation workers: read "Next steps" first.**

**Status in one paragraph.** Stage B ran unattended and finished at 09:40Z. The Gemma soup passed the development
gates but is **not a successor**: post-key v3 68.75 vs A20r 72.36 (−3.61 [−6.54, −1.18]; item 1 fail), mlx-diag
Choice + Noul −.036 [−.052, −.020] (item 4 fail); items 2, 3, 5, 6, 7 pass; vs AutoJev-27B −3.39 [−5.60, +0.79]
(no win). The **private Decision Index run** of the frozen package (IX1 harness, parity gate passed bit-identical)
runs on node A GPU3 / 4 / 5 and ends ≈ 12:30Z, then 20 oversized requests run alone (≈ 12:50Z). What remains:
score it (private), decide the hand-off status against the 27B-class frontier bar (private report
`private/ix1/ix1-report-2026-10-01.md`), free the node A leases, final records, gist, merge.

## Target, budget, GPUs

- Targets (prereg): beat AutoJev-27B significantly; successor items 1–7 vs DEV2.0-27B = A20r (72.360). Both missed.
- 60 GPU-h cap. **Receipts at 09:45Z: 33.987 finished** (node A 26.151, node B 7.836) + private Index path checks
  0.127 + parity 0.065 + **three Index shards running since ≈ 09:04Z** (≈ 3 × 4.5 h) → ≈ 47 at the end.
- GPUs: **node A GPU3 / 4 / 5 hold the Index shards (leases `track=27b-moe`, status running, set by the launcher;
  each returns to reserved-idle when its job ends)**. Node B GPU7 released 09:45Z (`owner.released-moe-20261001T0945Z`);
  node B GPU6 was released by continuation #1. Never use node A GPU0–2 / GPU6–7 or other node B GPUs.
- Platform rule: poll ≤ 30 min with a state commit; hand off through this file near 5 h.

## Running now (detached)

| What | Where | Log / status | Notes |
| --- | --- | --- | --- |
| Index shard 0 / 1 / 2 (40,010 / 40,302 / 39,894 requests after the presplit) | node A GPU3 / GPU4 / GPU5, containers `d2-27b-moe-ix-MOE-Git-soup-s{0,1,2}-g{3,4,5}` (each driven by a `moe-index.sh _shard` host process) | `/data/dev2/private/eval/index021/ix1/runs/MOE-Git-soup/shard-K/{status.json,runner.log,launch.log}` | ≈ 3.3 requests/s each; after the shard ends, the same host process reruns its pre-split requests alone (`extra-sK-N`; 5 / 9 / 6) and records a final error for any that fails alone |

Mirror for every Index job: `0403fb5796ca00a5a217f8e7656c77fe6e2205db-src_training_decision2` (node A).

## Infrastructure

- **Temporary node A → node B link: removed 09:45Z** (key line `dev2-27b-moe-xfer-temp` deleted from node B
  `authorized_keys`, backup `authorized_keys.bak.27b-moe-cleanup-*`; `/data/dev2/tmp/27b-moe-xfer` deleted on node A;
  a probe rsync is refused). `/root/.ssh/d2_temp_cd` (other tracks) untouched. `moe-index.sh offer / stage / auto-*`
  no longer work, and are not needed (the package is staged on node A).
- **Private Index layout (node A, mode 700):** `/data/dev2/private/eval/index021/ix1/` — `panel-3/` (gold-free
  shards + `compat-86.gold-free.jsonl.gz`, `panel.json`), `scan/` (request sizes), `parity/MOE-Git-soup/`
  (`parity.json`, `cache-frozen`), `runs/MOE-Git-soup/` (`presplit.json`, `shard-K/`, later `extra-*`, `merged/`),
  `logs/`. Path checks (tests only, never results): `/data/dev2/private/eval/index021/moe-pathcheck/`.
- **Tools** (`v2/27b/moe/`): `index_engine.py` (kit engine: the formal run's native path), `index_ref.py` (parity
  reference from the formal collector), `index_scan.py`, `index_skip.py` (IX1's rule), `moe-index.sh` (modes offer,
  stage, panel, parity, run, resume, extra, scan, presplit, auto-offer, auto-run). Scoring: IX1's
  `v2/eval/ix1/score.sh` (merge → port + kit 87d4650b → `compare.py`).
- Stage B outputs (node B): soup `R/MOE-Git-soup/` (checkpoint, cal698, ADOPTION.json, package/, latency/), readout
  `R/readouts/MOE-Git-soup`, `R/readouts/DEVGATES.json`, formal `R/formal/MOE-Git-soup{,-smoke,-mlx}`, `R/gates/`
  (`VERDICTS-20261001T094035Z.json`); node A `R/mlx-diag/MOE-Git-soup`. `pathcheck/` (both nodes) = tests only.
- Known pre-existing test failure (not this track's): `v2.27b.tests.test_m1_tools.LaunchTest.test_rejects_foreign_gpu`.

## Next steps (in order)

1. Poll every ≤ 30 min (node A): each `shard-K/status.json` (`completed`, `counts`); `docker ps | grep d2-27b-moe-ix`;
   commit a state line. If a shard's host process exits with the shard incomplete (`exit_code` ≠ 0 and no further
   attempt): read `runner.log`; `python3 -m v2.27b.moe.index_skip skip --shard-dir … --rows panel-3/shard-K-of-3.jsonl.gz`
   handles a device-aborting request; then `bash /data/dev2/src/$M/src/training/decision2/v2/27b/moe/moe-index.sh
   resume $M MOE-Git-soup 3 K GPU` (M = the mirror above). Never tune or select on Index rows.
2. When all three shards **and** their `extra-sK-N` reruns have `end_epoch` (≈ 12:50Z): on node A
   `bash /data/dev2/src/$M/src/training/decision2/v2/eval/ix1/score.sh --src /data/dev2/src/$M --model MOE-Git-soup
   --size 27B --panel /data/dev2/private/eval/index021/ix1/panel-3 --allow-errors` (`--allow-errors` because the two
   largest ToolRet requests, 488K and 751K padded tokens, are expected to exceed the GPU alone). Check the scorer
   gate (`merged/compare.json` → `scorers.pass`) and `merged/receipt.json` (row accounting).
3. **Private only:** copy `merged/{compare.json,port.json,receipt.json,latency.json}` to
   `/home/xunliu/code/decision2-program/private/moe1/` (mode 700) as `*-MOE-Git-soup.json`; run
   `python3 private/moe1/compare_moe.py --moe … --a20r ../ix1/compare-27B.json --frontier
   ../index021-frontier-gap-2026-10-01.json --out private/moe1/table-MOE-Git-soup.md`; write
   `private/moe1/moe1-index-report-2026-10-01.md` (label "independent provisional 0.2.1 reproduction": headline
   port / kit, vs DEV2.0-27B (IX1) and vs the 27B-class frontier peer and served size, areas, largest gains / losses,
   row accounting, latency). Never put an Index value in commits, the gist, cards, COORDINATION or STATUS.
4. **Hand-off status:** if the headline clears the 27B-class frontier bar in the IX1 report, mark
   `moe-handoff-2026-10-01.md` **ISSUED** (no value in the file: "the private Index criterion was met"); otherwise
   mark it **NOT ISSUED** (keep it as package notes). Add a public receipt (counts, hashes, GPU-h only; IX1's
   `ix1-public-receipt/1` shape) under `v2/27b/records/moe-index/`.
5. Free node A GPU3 / 4 / 5 (rename each `owner` to `owner.released-moe-<UTC>`, as for node B GPU7) once no
   `d2-27b-moe-*` container runs; results record + gist 06c (no Index values); merge into
   `xunzhuo/decision-2-training` (merge-only); report.

## Poll log (newest first)

- 09:52Z: Index shards 10,190 / 8,810 / 9,260 requests, all `ok` so far; 3 containers running. Integration
  fast-forwarded to `6e8d8e9b9` (merge of this branch); gist 06c updated. Results record: typed FINAL loss by family
  (`exception_stack` −.225 carries most of it).
- 09:50Z: **Stage B done** (node B 09:40:35Z, node A 09:38:05Z). Formal v3 **68.75** (T .812, H .582); vs A20r
  −3.61 [−6.54, −1.18]; mlx-diag −.036 [−.052, −.020] (R4 fail); verdicts: items 1–7 **false**, beats AutoJev
  **false** (`VERDICTS-20261001T094035Z.json`). Latency (BF16, M1 roster) 114.6 / 120.5 ms vs A20r 83.6 / 88.8.
  Link removed; node B GPU7 released. Index shards at ≈ 8,000 of ≈ 40,000 each (≈ 3.3 requests/s), end ≈ 12:30Z.
  Records: results (formal final), hand-off notes (PENDING on the private Index criterion).
- 09:08Z: **T = 1 readout** P_dev 75.26 (T_dev .927, H_pilot .611), H_dev2 .5728 (+.007 vs A20r, TIE).
  **CAL698** fit (.465 / .381 / .481) rejected (CSS-pilot ECE .043 → .128) → T = 1. **Development gates pass**
  (collapse, HT-DEV v2, proxy 3.73 below; typed guard reported, passes). **Package frozen** 08:57:35Z
  (`PACKAGE.json` `ffb11e1c…`, 25,310,379,550 loaded / 3,899,768,350 active). **Formal** started 08:57:35Z on
  node B GPU7 (smoke passed 09:02Z; collection running). **Private Index** (values private): package offered
  08:58:24Z, staged on node A 08:59:24Z; **86-request parity gate PASS** (86 / 86 ok, 419 questions, max |Δp| 0.0;
  0.065 GPU-h); presplit 20 requests ≥ 196,608 padded tokens (5 / 9 / 6 per shard); shards 0 / 1 / 2 running on
  node A GPU3 / 4 / 5 from 09:03:45Z (≈ 3–4.3 requests/s each; ≈ 2.5–3.3 h, then the 20 alone).
- 08:42Z: **MOE-Git-s2 finished** 08:28:00Z (10.373 GPU-h, exit 0; BEST `checkpoint-0002676`, SELECT .8219; last
  update 3,561 .8101). Node A relayed both BEST checkpoints at 08:32:53Z. **Soup** (node B, 08:37:45–08:39:20Z):
  `MOE-Git-soup` model `9165bed7…`, 2 members, rank 64 / α 128, 205 projections, max relative error 4.5e-7.
  Chain budget check: node A 26.151 + node B 6.862 = 33.01 GPU-h. T = 1 readout running on node B GPU7 from
  08:39:20Z.
- 07:45Z: MOE-Git-s2 update 3,296; SELECT at 3,122 .811 (BEST stays 2,676, .822); ends ≈ 08:27Z. Chains and
  watchers alive. **Formal smoke path check** (the chain's only GPU stage never run on an MoE checkpoint):
  `moe-formal.sh b 7 … smoke` from the chain's mirror `eacb6b85c` on PC-Git-s1-c892 with its T = 1 package
  calibration: typed FINAL / CSS15 / public 231 × 8 items, exit 0, 0.069 GPU-h; moved to
  `pathcheck/formal/PC-Git-s1-c892-smoke` (tests only). Node B GPU6's lease was released by continuation #1
  (`owner.released-moe-20261001T0645Z`); this track does not use it. Path checks 0.196 GPU-h in total.
- 07:32Z: MOE-Git-s2 update 3,235; ends ≈ 08:27Z. Both chains alive. **Memory finding (path check, synthetic rows,
  rank-32 package):** the FP32-resident native path fits 16 questions × 16K tokens (257K padded tokens, 27 s) but
  not 20 × 16K or 32 × 16K on one MI325X, and the kit runner halts a shard on any device error. Request-size scan of
  the panel under the package's encoder (inputs only): 149.0M tokens, 0 questions over 32,768, 20 requests ≥ 196,608
  padded tokens (ToolRet / BRIGHT 32-question requests; the largest two 488K and 751K). Tooling `0403fb579`:
  `index_scan.py`, `index_skip.py` (IX1's rule: such requests are pre-split out of their shard and rerun alone; a
  device-aborting request is skipped and rerun alone; one that fails alone is a final error). **Detached
  watchers:** node B `moe-index.sh auto-offer` (offers the package as soon as the chain freezes it;
  `stageb/index-offer.log`), node A `auto-run` with `MOE_INDEX_MIN_PADDED=196608` (stage → scan + 86-request parity
  on GPU3 → presplit → 3 shards on GPU3 / 4 / 5; `/data/dev2/private/eval/index021/ix1/logs/auto-run-MOE-Git-soup.log`).
  Path checks so far 0.127 GPU-h.
- 07:05Z (continuation #2): MOE-Git-s2 update 3,093 (BEST 2,676); ends ≈ 08:24Z. Both chains alive. Private Index
  tooling `a717a4315` (mirrored on both nodes): `v2/27b/moe/index_engine.py` (kit engine over the frozen package
  through the formal run's native path), `index_ref.py`, `moe-index.sh` (offer / stage / panel / parity / run /
  resume / extra). Node A panel built (`panel-3`: 120,226 rows, run-ID digest = IX1's). Path check (tests only, never
  results; synthetic non-Index rows): offer / stage of the PC-Git-s1-c892 package passed; parity path check on node A
  GPU3 (formal collector vs kit runner + engine, 7 synthetic requests: 5 ok, 2 refused, max |Δp| 0.0, **pass**;
  0.050 GPU-h) under `/data/dev2/private/eval/index021/moe-pathcheck/`; the path-check copies were removed.
- 06:40Z: hand-off. MOE-Git-s2 update 2,919 (BEST 2,676, SELECT .8219); ends ≈ 08:25Z. Both chains alive; relay / mlx
  hand-over dirs hold only MOE-Git-s1-best (pre-relayed) and the screen's files. Receipts ≈ 31.1 GPU-h.
- 06:07Z: MOE-Git-s2 update 2,759; BEST now 2,676 (SELECT 0.8219 / 0.1212); ends ≈ 08:25Z. Both chains alive.
- 05:45Z: MOE-Git-s2 update 2,642 (BEST 1,784); ends ≈ 08:25Z. Both chains alive.
- 05:08Z: MOE-Git-s2 update 2,421 (BEST 1,784; SELECT at 2,230 .8201); ends ≈ 08:25Z. Both chains alive.
- 04:40Z: MOE-Git-s2 update 2,250 (BEST still 1,784; ends ≈ 08:25Z). Both chains alive.
- 04:12Z: MOE-Git-s2 update 2,099 (BEST so far 1,784); ends ≈ 08:20Z. Both chains alive and waiting.
- 03:48Z: more path checks under `pathcheck/` (tests only; scratch removed from the relay / mlx hand-over dirs):
  `soup` on node B from two relayed checkpoints (205 projections, 4.3e-7, rank 64 / α 128, 1.5 min); `gates` and
  `verdicts` on A20r's own sealed run (self Δ 0 [0, 0]; vs AutoJev-27B +0.23 [−1.60, +4.74]; types / public 231 OK;
  overlap 0 groups; item 4 PENDING without a pairing, as designed); node A `mlx-score` + node B `mlx-pull` on a copy of
  M4-A20-soup's gold-free mlx-diag collection (pull, score, pairing vs A20r, push back). Untested on the soup only:
  the GPU collections (`readout` = the screen's proven driver; `formal` / `mlx` = `moe-formal.sh`, smoke first).
- 03:42Z: MOE-Git-s2 update 1,919 (10.1 s per update; BEST so far 1,784, SELECT .8214); ends ≈ 08:25Z. Both chains
  alive and waiting. MOE-Git-s1's BEST already relayed to node B by the relay stage (path check of the relay; the
  chain relays both seeds again when s2 ends). Integration fast-forwarded to `63582e120`.
- 03:15Z: MOE-Git-s1 finished (10.240 GPU-h; BEST 2,676). Amendment 4 `eacb6b85c`; Stage B chains launched on both
  nodes from it. Path checks passed (cal698, adoption, gates, package, latency; A20r latency reference). Receipts
  22.64 finished + s2 running.
- 02:55Z: G-it s1 at update 3,433, s2 at 1,616 (healthy). Shared fix `9273b0bc0` (CAL698 fitter uses the checkpoint's
  prompt version). Stage B drivers + chains `ead10fffa` (mirrored on both nodes): exact soup dry run on the two
  seeds' checkpoint 892 passed (205 projections, max relative error 4.3e-7; scratch deleted); `moe-formal.sh` peer
  paths fixed (they pointed at missing directories). Path check of cal698 / adopt / devgates / package / latency
  running on node B GPU7 under `/data/dev2/runs/27b-moe/pathcheck/` (MOE-Git-s1 checkpoint 892).
- 02:45Z: continuation #1 took over; screen reconstructed (Gemma won; Qwen cells stopped by rule 3); receipts 12.30
  finished + ≈ 14.0 running; both Gemma seeds healthy (no nonfinite loss).
- 17:05Z: Gemma cell training (≈ 9.9 s per update); both Qwen one-steps passed, reloads running; dense reference
  readout running; screen chains launched. Receipts ≈ 0.75 GPU-h.
- 16:50Z: amendment 2 (the A20r trainer is the repository trainer; wrapper); cells relaunched with `-r2` preflights.
- 16:35Z: gate done, prereg written; P0 probes → amendment 1 (grouped_mm, caps, three cells).

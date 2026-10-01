# 27B MoE milestone (MoE-1): state (resume file)

**Continuation #2 (from 06:40Z):** follows the Stage B chains, writes the formal verdicts, runs the private Decision
Index of the frozen package (IX1 harness; values private) and writes the hand-off if the package qualifies.

Updated: 2026-10-01 14:40 UTC+8 (06:40Z). **Continuation #1 handed off here** (worked 02:21–06:45Z). Branch
`xunzhuo/decision-2-training-27b-moe`, worktree `/home/xunliu/code/vllm-sr-dev2-27b-moe`, gist
`06c-decision-2-27b-moe.md`. Assignment: COORDINATION 2026-09-30 23:50. **Continuation workers: read "Next steps"
first.** Prereg `moe-prereg-2026-10-01.md` (amendments 1–4; amendment 4 = `eacb6b85c`, 03:06Z); gate
`moe-gate-2026-10-01.md`; results (interim) `moe-results-2026-10-01.md`.

**Hand-off in one paragraph.** The screen is done (Gemma-4-26B-A4B-it won; Qwen3.5-35B-A3B stopped by the proxy
rule). Seed 1 finished; seed 2 ends ≈ 08:25Z (16:25 UTC+8). From then on the two Stage B chains (PIDs below) run the
whole preregistered tail unattended: relay → soup → T = 1 readout → CAL698 + 23:15 adoption → development gates →
(pass) frozen package → formal → gates → latency → mlx-diag (node A) → `R/gates/VERDICTS-*.json` on node B, ≈ 2–2.5 h
after seed 2. Every stage was path-checked except the soup's GPU collections. The next worker watches the chain logs,
records each result, runs the verdicts if the chain stops early, and writes the release hand-off if there is a winner.

## Target, budget, GPUs

- Beat AutoJev-27B significantly (post-key v3 > 72.133, paired lower bound > 0, H not below); successor items 1–7 vs
  DEV2.0-27B = A20r (72.360, node B `/data/dev2/runs/27b/M4-A20r-soup/formal`); item 8 via the eval custodian.
- 60 GPU-h cap. **Receipts at 06:35Z: 22.64 GPU-h finished** (12.301 before; MOE-Git-s1 full 10.240; path checks
  0.069; A20r latency reference 0.030) **+ MOE-Git-s2 running ≈ 8.5 → ≈ 31.1.** Projection ≈ 33 at the end of
  training (≈ 08:25Z), ≈ 35 after Stage B (A20r's formal run was 0.41 GPU-h; readout ≈ 0.3, CAL698 ≈ 0.05,
  mlx-diag ≈ 0.15, latency ≈ 0.03).
- GPUs (leases `track=27b-moe`): node A GPU4 (MOE-Git-s2, ends ≈ 08:25Z); node B GPU7 (Stage B readouts / formal).
  **Idle for the rest of the milestone: node A GPU3, GPU5 and node B GPU6** (reserved-idle; the coordinator may
  reassign them). Never use node A GPU2 / node B GPU0, 1, 5 (27B M5), node A GPU6–7, node A GPU0–1 / node B GPU2,
  node B GPU3–4.
- Platform rule: poll ≤ 30 min with a state commit; hand off through this file near 5 h.

## Screen outcome (automatic, 2026-09-30 21:58Z; node B `/data/dev2/runs/27b-moe/screen/SCREEN.json`)

- **Gemma-4-26B-A4B-it won**; MOE-Git-s1 continued and MOE-Git-s2 started on node A GPU4 at 21:59:37Z.
- **Both Qwen3.5-35B-A3B cells were stopped by screen rule 3** (P_dev ≥ 8 below the dense reference's 78.22): Q-it
  69.69, Q-pt 69.75. No collapse, every HT-DEV v2 verdict TIE. G1 (reconstructed) passed.

## Running now (detached)

| What | Where | Log | Notes |
| --- | --- | --- | --- |
| MOE-Git-s2 full (seed 20260928) | node A GPU4, `d2-27b-moe-MOE-Git-s2-full` | node A `/data/dev2/runs/27b-moe/MOE-Git-s2/driver.log` | from `2139aac8d`; 10.2 s per update; update 2,919 at 06:34Z; BEST so far `checkpoint-0002676` (SELECT .8219); ends ≈ 08:25Z |
| Stage B chain, node A (waits for both `COMPLETE.json`, relays BEST checkpoints + receipt total; later scores mlx-diag) | node A host, PID 4105125 | node A `/data/dev2/runs/27b-moe/stageb/nodeA.log` | mirror `eacb6b85c` |
| Stage B chain, node B (soup → T = 1 readout → CAL698 → adoption → gates → package → formal → gates → latency → mlx → verdicts) | node B host, PID 2728109 | node B `/data/dev2/runs/27b-moe/stageb/nodeB.log` | mirror `eacb6b85c` |

**Finished:** MOE-Git-s1 (03:10:21Z, 10.240 GPU-h, exit 0; `COMPLETE.json`; BEST `checkpoint-0002676`, SELECT
family-macro .8281 / Brier .1188; last update 3,561: .8209). Seeds' `decision_config.json` are identical.

## Infrastructure

- **Bases** on both nodes at `/data/dev2/models/moe/<name>`, hash-verified (`logs/verify-bases-node{A,B}.json`).
- **Temporary node A → node B link** (rsync only): key on node A `/data/dev2/tmp/27b-moe-xfer/` (mode 700; `peer`
  holds node B's private address), authorized on node B with `from=<node A private address>,restrict,
  command="/usr/bin/rrsync /data/dev2/xfer/27b-moe"` (comment `dev2-27b-moe-xfer-temp`; node B's previous
  `authorized_keys` backed up as `authorized_keys.bak.27b-moe-<UTC>`). **At milestone end remove that key line on
  node B and `/data/dev2/tmp/27b-moe-xfer` on node A.**
- Stage B driver `v2/27b/moe/moe-tail.sh STAGE MIRROR ...` (relay, soup, readout, cal698, adopt, devgates, package,
  formal, gates, mlx, mlx-score, mlx-pull, latency, latency-ref, verdicts); chains `moe-stageb-node{A,B}.sh`; modules
  `moe_devgates.py`, `moe_params.py`, `moe_verdicts.py`, `latency.py`. Outputs (node B): soup `R/MOE-Git-soup/`
  (checkpoint, cal698, ADOPTION.json, package/PACKAGE.json, latency/), readout `R/readouts/MOE-Git-soup`, gates
  `R/readouts/DEVGATES.json`, formal `R/formal/MOE-Git-soup{,-smoke,-mlx}`, `R/gates/` (paired, types, public231,
  overlap, mlx, VERDICTS-*.json); node A `R/mlx-diag/`. Relay / mlx hand-over in `/data/dev2/xfer/27b-moe/{relay,mlx}`.
- If a chain stage fails: read the chain log, fix in the repo, commit, mirror
  (`v2/common/mirror_to_node.sh --path src/training/decision2 node-b <sha>`), and rerun **that stage only**, e.g.
  `M=<sha>-src_training_decision2; bash /data/dev2/src/$M/src/training/decision2/v2/27b/moe/moe-tail.sh formal $M
  MOE-Git-soup` (stages refuse to overwrite their outputs; continue the remaining stages by hand in the chain's
  order, and write `/data/dev2/xfer/27b-moe/mlx/MOE-Git-soup.PUSHED` only through the `mlx` stage). Never rerun a
  training arm.
- `pathcheck/` (both nodes) holds the path-check outputs only: never cite them as results.
- Latency references: A20r p50 / p95 83.6 / 88.8 ms (FLA kernels, BF16, 52.2 GB resident); path check on
  MOE-Git-s1 checkpoint 892 (rank 32) 119.7 / 126.6 ms (50.7 GB).
- Known pre-existing test failure (not this track's): `v2.27b.tests.test_m1_tools.LaunchTest.test_rejects_foreign_gpu`
  (the test still encodes M1's GPU allocation; fails identically on `2139aac8d`).

## Budget (sum receipts on both nodes)

```bash
for n in root@<node A> root@<node B>; do ssh $n 'python3 - <<EOF
import glob, json
r = "/data/dev2/runs/27b-moe"
t = sum(json.load(open(p)).get("gpu_hours", 0) for p in glob.glob(f"{r}/*/receipts/*.json") + glob.glob(f"{r}/pathcheck/*/receipts/*.json"))
g = sum(json.load(open(p)).get("gpu_hours", 0) for p in glob.glob(f"{r}/readouts/*/GPU-TIME.json") + glob.glob(f"{r}/formal/*/GPU-TIME.json"))
print(round(t, 3), round(g, 3))
EOF'; done
```

Running full attempts have no receipt until they end: add (now − start) for each.

## Next steps (in order)

1. Poll every ≤ 30 min: `docker ps`, MOE-Git-s2's step count, both chain logs (PIDs above); commit a state line.
2. ≈ 08:25Z MOE-Git-s2 ends → node A relays → node B soups and reads out (≈ 30 min) → gates → (pass) package →
   formal (≈ 1–1.5 h) → gates, latency, mlx (≈ 20 min) → node A pairing (≤ 10 min) → `R/gates/VERDICTS-*.json`.
   Check each stage's output as it lands (`stageb/nodeB.log`); record soup, adoption, gates, formal v3 / T / H, paired
   intervals, types, public 231, overlap, mlx R4, latency and parameters in the results record and gist 06c.
3. Verdicts: "beats AutoJev" (v3 > 72.133, lower bound > 0 vs AutoJev-27B, H not below) and successor items 1–7 vs
   A20r; report loaded / active parameters and latency (soup vs A20r, same tool).
4. If the soup passes items 1–7 (a successor): write the release hand-off — the frozen package (`PACKAGE.json` +
   checkpoint + base pin + `grouped_mm` + BOS prompt) and a C1 post-key spec draft for the eval custodian (template
   `v2/eval/sealed/c1-postkey/dev2-27b-a20r.json`; adapter spec `v2/27b/moe/adapters/moe-lora-cal.json`; stage the
   package, the base and the formal run's stored predictions on node A). Naming (DEV2.0-26B-A4B) and placement
   (replace the 27B tier or add a family member) are the coordinator's call. **Release gap to flag:** the release
   builder (`v2/release/build.py`) has only a `qwen-adapter` LoRA profile; a Gemma MoE package needs a new profile
   and a native runtime that loads Gemma 4 MoE with `grouped_mm` and the BOS prompt (A20r's C1 spec used a release
   pre-build, so item 8 waits for that build).
   Matched latency for the report: A20r's formal per-prompt `latency_ms` (FP32-resident + BF16 autocast, the formal
   path) p50 / p95 typed FINAL 118.8 / 145.1, CSS15 103.8 / 232.6, public 231 106.2 / 418.2 ms; compare the soup's
   formal predictions the same way. A20r's whole formal run was 0.407 GPU-h.
5. Results record final, gist 06c, merge into `xunzhuo/decision-2-training` (merge-only), remove the temporary link
   (above), release the leases, report.

## Poll log (newest first)

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

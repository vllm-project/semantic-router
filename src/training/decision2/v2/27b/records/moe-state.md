# 27B MoE milestone (MoE-1): state (resume file)

Updated: 2026-10-01 11:15 UTC+8 (03:15Z). Worker: 27B MoE continuation #1 (started 02:21Z; hand-off due ≈ 07:15Z),
branch `xunzhuo/decision-2-training-27b-moe`, worktree `/home/xunliu/code/vllm-sr-dev2-27b-moe`, gist
`06c-decision-2-27b-moe.md`. Assignment: COORDINATION 2026-09-30 23:50. **Continuation workers: read "Next steps"
first.** Prereg `moe-prereg-2026-10-01.md` (amendments 1–4; amendment 4 = `eacb6b85c`, 03:06Z); gate
`moe-gate-2026-10-01.md`; results (interim) `moe-results-2026-10-01.md`.

## Target, budget, GPUs

- Beat AutoJev-27B significantly (post-key v3 > 72.133, paired lower bound > 0, H not below); successor items 1–7 vs
  DEV2.0-27B = A20r (72.360, node B `/data/dev2/runs/27b/M4-A20r-soup/formal`); item 8 via the eval custodian.
- 60 GPU-h cap. **Receipts at 03:15Z: 22.64 GPU-h finished** (12.301 before; MOE-Git-s1 full 10.240; path checks
  0.069; A20r latency reference 0.030) **+ MOE-Git-s2 running ≈ 5.2 → ≈ 27.8.** Projection ≈ 33 at the end of
  training (≈ 08:25Z), ≈ 36 after Stage B.
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
| MOE-Git-s2 full (seed 20260928) | node A GPU4, `d2-27b-moe-MOE-Git-s2-full` | node A `/data/dev2/runs/27b-moe/MOE-Git-s2/driver.log` | from `2139aac8d`; 10.3 s per update; ends ≈ 08:25Z |
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
- If a chain stage fails: read the chain log, fix in the repo, commit, mirror, and rerun **that stage only** with
  `moe-tail.sh` (stages refuse to overwrite their outputs); never rerun a training arm.
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

# 27B MoE milestone (MoE-1): state (resume file)

Updated: 2026-10-01 10:45 UTC+8 (02:45Z). Worker: 27B MoE continuation #1 (started 02:21Z; hand-off due ≈ 07:15Z),
branch `xunzhuo/decision-2-training-27b-moe`, worktree `/home/xunliu/code/vllm-sr-dev2-27b-moe`, gist
`06c-decision-2-27b-moe.md`. Assignment: COORDINATION 2026-09-30 23:50. **Continuation workers: read "Next steps"
first.** Prereg `moe-prereg-2026-10-01.md` (amendments 1–3); gate `moe-gate-2026-10-01.md`; results (interim)
`moe-results-2026-10-01.md`. The previous worker stopped silently after its 17:20Z integration push; its screen chains
ran the preregistered screen without it.

## Target, budget, GPUs

- Beat AutoJev-27B significantly (post-key v3 > 72.133, paired lower bound > 0, H not below); successor items 1–7 vs
  DEV2.0-27B = A20r (72.360, node B `/data/dev2/runs/27b/M4-A20r-soup/formal`); item 8 via the eval custodian.
- 60 GPU-h cap. **Receipts at 02:30Z: 12.30 GPU-h finished** (node A 5.538, node B 5.148 + readouts 1.615) **+ the two
  running Gemma seeds ≈ 14.0 → ≈ 26.3.** Projection: ≈ 33 at the end of training (≈ 08:25Z), ≈ 36 after Stage B.
- GPUs (leases `track=27b-moe`): node A GPU3 (G-it s1, ends ≈ 03:15Z), GPU4 (G-it s2, ends ≈ 08:25Z), GPU5 (idle,
  reserved); node B GPU6 (idle, reserved), GPU7 (readouts / formal). Never use node A GPU2 / node B GPU0, 1, 5 (27B M5),
  node A GPU6–7, node A GPU0–1 / node B GPU2, node B GPU3–4.
- Platform rule: poll ≤ 30 min with a state commit; hand off through this file near 5 h.

## Screen outcome (automatic, 2026-09-30 21:58Z; node B `/data/dev2/runs/27b-moe/screen/SCREEN.json`)

- **Gemma-4-26B-A4B-it won.** MOE-Git-s1 continues; its seed 2 (MOE-Git-s2) started on node A GPU4 at 21:59:37Z.
- **Both Qwen3.5-35B-A3B cells were stopped by screen rule 3** (proxy gap ≥ 8 below the best of the cells and the dense
  matched reference, 78.22): Q-it 69.69 (gap 8.53), Q-pt 69.75 (gap 8.46). No cell tripped collapse or HT-DEV v2.
  Q-it stopped at update 1,136 (4.993 GPU-h), Q-pt at 1,058 (4.980 GPU-h); both drivers wrote `STOP` and their receipts.
- G1 (never recorded live): reconstructed from the logged speeds, a pass (details in the results record).

## Running now (detached; liveness by `docker ps` name or the driver's flock `RUN/.driver.lock`)

| What | Where | Log | Notes |
| --- | --- | --- | --- |
| MOE-Git-s1 full (seed 20260926) | node A GPU3, `d2-27b-moe-MOE-Git-s1-full` | node A `/data/dev2/runs/27b-moe/MOE-Git-s1/driver.log` | from `e5bbaba56`; 9.8 s per update; update 3,331 at 02:29Z; BEST so far `checkpoint-0002676` (SELECT .8281) |
| MOE-Git-s2 full (seed 20260928) | node A GPU4, `d2-27b-moe-MOE-Git-s2-full` | node A `.../MOE-Git-s2/driver.log` | from `2139aac8d`; 10.3 s per update; update 1,516 at 02:29Z; preflights passed |

`decision_config.json` of the two seeds is identical (the soup's member check needs that).

## Infrastructure

- **Bases** on both nodes at `/data/dev2/models/moe/<name>`, hash-verified (`logs/verify-bases-node{A,B}.json`).
- **Temporary node A → node B link** (rsync only): key on node A `/data/dev2/tmp/27b-moe-xfer/` (mode 700; `peer`
  holds node B's private address), authorized on node B with `from=<node A private address>,restrict,
  command="/usr/bin/rrsync /data/dev2/xfer/27b-moe"` (comment `dev2-27b-moe-xfer-temp`; node B's previous
  `authorized_keys` backed up as `authorized_keys.bak.27b-moe-<UTC>`). **At milestone end remove that key line on
  node B and `/data/dev2/tmp/27b-moe-xfer` on node A.**
- Drivers: `v2/27b/moe/moe-arm.sh`, `moe-readout.sh` (T = 1 readout), `moe-formal.sh` (smoke / collect / score / mlx),
  `moe-screen-node{A,B}.sh` (finished), `screen_rules.py`, `train_moe.py`, `collect.py` + `adapters/moe-lora{,-cal}.json`.
- **Known gap (being fixed):** `v2.release.calibrate_frozen` encodes CAL rows with the plain prompt, not the
  checkpoint's prompt version, so a Gemma CAL698 fit would miss the BOS token used in training and serving.

## Budget (sum receipts on both nodes)

```bash
for n in root@<node A> root@<node B>; do ssh $n 'python3 - <<EOF
import glob, json
t = sum(json.load(open(p)).get("gpu_hours", 0) for p in glob.glob("/data/dev2/runs/27b-moe/*/receipts/*.json"))
g = sum(json.load(open(p)).get("gpu_hours", 0) for p in glob.glob("/data/dev2/runs/27b-moe/readouts/*/GPU-TIME.json"))
print(round(t, 3), round(g, 3))
EOF'; done
```

Running full attempts have no receipt until they end: add (now − start) for each.

## Next steps (in order)

1. Poll every ≤ 30 min: `docker ps`, driver logs, step counts; commit a state line.
2. Fix the CAL698 fitter's prompt encoder (shared module, separate commit with a test); test the soup tool on the two
   seeds' `checkpoint-0000892` (CPU, scratch output, deleted afterwards); prepare the Stage B driver.
3. Stage B when MOE-Git-s2 ends (≈ 08:25Z): relay both BEST checkpoints to node B; exact soup (`v2.27b.lora_soup`,
   rank 64 / α 128); T = 1 soup readout on node B GPU7 (`moe-readout.sh`, HT-DEV v2 reference = M5's A20r-ref
   predictions); CAL698 fit; 23:15 adoption (`v2.release.dev_calibration`); development gates (collapse, HT-DEV v2 not
   FLAG vs A20r .5655, P_dev not ≥ 8 below 78.99); frozen package; formal (`moe-formal.sh`, node B GPU7); `gates types`,
   `gates public231`, mlx-diag (score on node A), overlap exposure on `a20`, latency; verdicts.
4. Results record, gist, merge into `xunzhuo/decision-2-training`; a successor gets a release hand-off (the frozen
   package and a C1 spec for the eval custodian).

## Poll log (newest first)

- 02:45Z: continuation #1 took over; screen reconstructed (Gemma won; Qwen cells stopped by rule 3); receipts 12.30
  finished + ≈ 14.0 running; both Gemma seeds healthy (no nonfinite loss).
- 17:05Z: Gemma cell training (≈ 9.9 s per update); both Qwen one-steps passed, reloads running; dense reference
  readout running; screen chains launched. Receipts ≈ 0.75 GPU-h.
- 16:50Z: amendment 2 (the A20r trainer is the repository trainer; wrapper); cells relaunched with `-r2` preflights.
- 16:35Z: gate done, prereg written; P0 probes → amendment 1 (grouped_mm, caps, three cells).

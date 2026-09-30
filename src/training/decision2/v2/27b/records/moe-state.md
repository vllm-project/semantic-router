# 27B MoE milestone (MoE-1): state (resume file)

Updated: 2026-10-01 01:05 UTC+8 (17:05Z). Worker: 27B MoE (branch `xunzhuo/decision-2-training-27b-moe`, worktree
`/home/xunliu/code/vllm-sr-dev2-27b-moe`, gist `06c-decision-2-27b-moe.md`). Assignment: COORDINATION 2026-09-30
23:50. **Continuation workers: read "Next steps" first.** Prereg `moe-prereg-2026-10-01.md` (amendments 1–3); gate
`moe-gate-2026-10-01.md`. Latest mirror on both nodes: **`2139aac8d`** (cells run from `e5bbaba56`).

## Target, budget, GPUs

- Beat AutoJev-27B significantly (post-key v3 > 72.133, paired lower bound > 0, H not below); successor items 1–7 vs
  DEV2.0-27B = A20r (72.360, node B `/data/dev2/runs/27b/M4-A20r-soup/formal`); item 8 via the eval custodian.
- 60 GPU-h cap. Receipts at 17:05Z ≈ 0.75 (probes 0.15, checks 0.05, failed launches 0.001, Gemma preflights 0.09,
  Qwen preflights running, dense reference readout running). Sum: see "Budget" below.
- GPUs (leases `track=27b-moe`): node A GPU3 (G-it), GPU4 (held for seed 2), GPU5 (Q-it); node B GPU6 (Q-pt), GPU7
  (readouts / formal). Never use node A GPU2 / node B GPU0, 1, 5 (27B M5 L128), node A GPU6–7 (9B M7), node A
  GPU0–1 / node B GPU2 (serving / eval), node B GPU3–4 (HR2).
- Platform rule: poll ≤ 30 min with one-line state updates; hand off through this file near 5 h.

## Running now (detached; liveness by `docker ps` name or the driver's flock `RUN/.driver.lock`)

| What | Where | Log | Notes |
| --- | --- | --- | --- |
| MOE-Git-s1 full (Gemma-4-26B-A4B-it, s1, grouped_mm, 4,736 limit) | node A GPU3, `d2-27b-moe-MOE-Git-s1-full` | node A `/data/dev2/runs/27b-moe/MOE-Git-s1/driver.log` | preflights passed (reload 0/32, 6e-8); started ≈ 16:56Z; ≈ 9.9 s per update; cap 15 GPU-h |
| MOE-Qit-s1 (Qwen3.5-35B-A3B, s1) onestep-r2 → reload-r2 → full | node A GPU5 | node A `.../MOE-Qit-s1/driver.log` | cap 20 GPU-h |
| MOE-Qpt-s1 (Qwen3.5-35B-A3B-Base, s1) onestep-r2 → reload-r2 → full | node B GPU6 | node B `.../MOE-Qpt-s1/driver.log` | cap 20 GPU-h |
| Dense matched reference readout A20r-s1 checkpoint 892 (T = 1, 32K, typed-dev + css-pilot + ht-dev2) | node B GPU7 | node B `/data/dev2/runs/27b-moe/logs/readout-A20r-s1-c892.log` | output `readouts/A20r-s1-c892` |
| Screen chain node A (relay checkpoint 892 → node B; then apply SCREEN.json: stops + seed 2 on GPU4) | node A host | node A `/data/dev2/runs/27b-moe/screen/nodeA.log` | mirror `2139aac8d` |
| Screen chain node B (waits for the dense reference; readouts of cells on GPU7; `screen_rules.py`; SCREEN.json) | node B host | node B `/data/dev2/runs/27b-moe/screen/nodeB.log` | mirror `2139aac8d` |

## Infrastructure

- **Bases** on both nodes at `/data/dev2/models/moe/<name>`, hash-verified (`logs/verify-bases-node{A,B}.json`).
- **Temporary node A → node B link** (rsync only): key on node A `/data/dev2/tmp/27b-moe-xfer/` (mode 700; `peer`
  holds node B's private address), authorized on node B with `from=<node A private address>,restrict,
  command="/usr/bin/rrsync /data/dev2/xfer/27b-moe"` (comment `dev2-27b-moe-xfer-temp`; node B's previous
  `authorized_keys` backed up as `authorized_keys.bak.27b-moe-<UTC>`). **At milestone end remove that key line on
  node B and `/data/dev2/tmp/27b-moe-xfer` on node A.**
- Drivers: `v2/27b/moe/moe-arm.sh` (arm-seed: probe, check, admit, onestep, reload, full; `PREFLIGHT_TAG`),
  `moe-readout.sh` (T = 1 readout), `moe-screen-node{A,B}.sh`, `screen_rules.py`, `train_moe.py` (wrapper around the
  byte-frozen trainer), `collect.py` + `adapters/moe-lora.json` (same-panel collector).
- Speeds (probe): Gemma ≈ 0.66 s per row (≈ 13–14 s per update), Qwen-MoE ≈ 1.07 s per row (≈ 17–21 s per update).

## Budget (sum receipts on both nodes)

```bash
for n in root@<node A> root@<node B>; do ssh $n 'python3 - <<EOF
import glob, json
t = sum(json.load(open(p)).get("gpu_hours", 0) for p in glob.glob("/data/dev2/runs/27b-moe/*/receipts/*.json"))
g = sum(json.load(open(p)).get("gpu_hours", 0) for p in glob.glob("/data/dev2/runs/27b-moe/readouts/*/GPU-TIME.json"))
print(round(t, 3), round(g, 3))
EOF'; done
```

## Next steps (in order)

1. Poll every ≤ 30 min: `docker ps`, driver logs, `train-metrics.jsonl` step counts. If a Qwen preflight fails, the
   cell stops (prereg); its driver ends and the screen chains mark it absent automatically.
2. **G1 at ≈ 300 updates of every cell** (≈ 17:45Z Gemma, ≈ 18:40Z Qwen): project each full attempt from
   `train-metrics.jsonl` timestamps; if the worst-case plan (amendment 1) exceeds 60 GPU-h, stop Q-pt first (write
   `RUN/STOP`, `docker stop d2-27b-moe-MOE-Qpt-s1-full`, and write `ABSENT`-equivalent: the node B chain treats an
   ended driver as absent). Record in the prereg as the G1 outcome.
3. Screen (automatic): checkpoint 892 arrives ≈ 19:25Z (Gemma) / ≈ 21:00–22:00Z (Qwen). Check `screen/nodeB.log`,
   `SCREEN.json`, and that node A applied it (`screen/nodeA.log`; seed 2 on node A GPU4 as `MOE-<cell>-s2`).
   Record the screen in `moe-results-2026-10-01.md` and gist 06c.
4. Stage B: when the continuing cell(s) and seed 2 finish: exact soup (`v2/27b/lora_soup.py`, as `m5-tail.sh lsoup`),
   soup readout + CAL698 fit on node B GPU7, development gates (prereg), formal (`v2/eval/run_same_panel.sh` with
   `adapters/moe-lora.json`, full panels, 32K, node B GPU7), seal / report / compare vs A20r, AutoJev, Eikos,
   Jebadiah; `gates types`, `gates public231`, mlx-diag (score on node A), overlap exposure on `a20`; latency.
5. Verdicts, results record, gist, merge into `xunzhuo/decision-2-training`, report; a successor gets a release
   hand-off (frozen package + C1 spec for the eval custodian).

## Poll log (newest first)

- 17:05Z: Gemma cell training (≈ 9.9 s per update); both Qwen one-steps passed, reloads running; dense reference
  readout running; screen chains launched. Receipts ≈ 0.75 GPU-h.
- 16:50Z: amendment 2 (the A20r trainer is the repository trainer; wrapper); cells relaunched with `-r2` preflights.
- 16:35Z: gate done, prereg written; P0 probes → amendment 1 (grouped_mm, caps, three cells).

# Decoder Milestone 10 — state (keep current; newest first)

Assignment: COORDINATION 2026-10-01 11:20 (4B readout × initialisation probe on node E GPU0–3 + node F GPU2–7, 120
GPU-h). Preregistration [`dec-m10-prereg-2026-10-01.md`](dec-m10-prereg-2026-10-01.md) (`2dea44d6f`); data lock
[`dec-m10-datalock-2026-10-01.md`](dec-m10-datalock-2026-10-01.md) (`6a8092977`); amendment
[1](dec-m10-amendment-1-2026-10-01.md) (`04088322b`). Branch `xunzhuo/decision-2-training-dec-m10`, worktree
`/home/xunliu/code/vllm-sr-dev2-dec-m10`.

## Now

- 2026-10-01 ≈12:10 UTC+8 (04:10Z) — **Nine seeds training.**
  - Node F (mirror `2dea44d6f…`): LH s1–s3 on GPU2–4, FB s1–s3 on GPU5–7; all six preflights PASS (cross-process
    drift ≤ 1e-7 on the seeded cache); 787 updates per seed at ≈ 6–8 s → ETA ≈ 05:30Z.
  - Node E (mirror `04088322b…`): LT2 s1–s3 on GPU0–2 (wave `w2`), in preflight; NT2 follows on the same GPUs if all
    three LT2 preflights pass.
  - LT (8,192 tokens) stopped at LT-s1's zero-step (amendment 1).
- Next: retention probes (candidates → TRAIN / suite overlap → finalize), C0 references on node E GPU3, merge / line /
  scoring scripts, the release-runtime `label_token` path with its parity test.

## Plan

| Step | Where | Status |
| --- | --- | --- |
| Prereg, tooling, tests | workstation | done `06206fef9`, `2dea44d6f` |
| Inputs node B → E / F, data build, lock | nodes | done `6a8092977` (identical on both nodes) |
| Chains: LH / FB (node F), LT2 (+ NT2) (node E) | nodes E / F | running |
| Retention probes (build, overlap checks, finalize) | node E CPU, node C CPU | next |
| References: C0 (both nodes), base ceiling (LT2-s1 zero-step) | node E GPU3, node F after chains | next |
| Merge LoRA BESTs, soups, lines, scoring, gates | nodes E / F, node A CPU | after training |
| Formal (≤ 2 finalists), successor items, IX1 hand-off | nodes E / F, node A | after gates |

## Operations

- Liveness: `cat /data/dev2/runs/dec/m10/chains/chain-*.pid` + `ps -p`; `docker ps | grep m10-`. Never `pgrep -f`.
- Logs: `m10/OPERATIONS.log`, `m10/logs/chain-*.log`, `m10/arms/OPERATIONS.log`; markers `m10/status/`.
- GPU-hours: `python3 <mirror>/…/v2/dec/ops/m10/m10_gpuh.py table` on each node.

## GPU-hours (cap 120)

| Item | GPU-h |
| --- | ---: |
| LT-s1 zero-step (failed) | 0.01 |
| Running | — |

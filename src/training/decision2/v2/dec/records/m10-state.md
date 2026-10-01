# Decoder Milestone 10 — state (keep current; newest first)

Assignment: COORDINATION 2026-10-01 11:20 (4B readout × initialisation probe on node E GPU0–3 + node F GPU2–7, 120
GPU-h). Preregistration [`dec-m10-prereg-2026-10-01.md`](dec-m10-prereg-2026-10-01.md) (`2dea44d6f`); data lock
[`dec-m10-datalock-2026-10-01.md`](dec-m10-datalock-2026-10-01.md) (`6a8092977`); amendment
[1](dec-m10-amendment-1-2026-10-01.md) (`04088322b`). Branch `xunzhuo/decision-2-training-dec-m10`, worktree
`/home/xunliu/code/vllm-sr-dev2-dec-m10`.

## Now

- 2026-10-01 ≈12:42 UTC+8 (04:42Z) — **Training ≈ 75–98%; post chains armed; formal path ready.**
  - FB 747–769 / 787, LH 671–718, LT2 558–587 (correction: the 04:50Z header below was 04:20Z; ETAs ≈ 04:45Z
    node F, ≈ 04:58Z node E).
  - **C0 readout parity:** `4b-C0-e` on the M10 path equals node B's stored `4b-I` readouts on typed DEV, CSS pilot,
    HT-DEV v2, Score5-typed-DEV and `hs1-dev`: 0 differing decisions, maximum drift 0.0. The base ceiling
    (`4b-BASE-e`) is reading.
  - Post chains (`m10-post.sh`, mirror `b55d8828…`): LH → node F GPU2, FB → GPU5, LT2 → node E GPU3 (merge LoRA
    BESTs, soup, read all panels); `4b-C0-f` (cross-node parity) on node F GPU6 after FB-s2.
  - Formal path for nodes E / F committed: `run_same_panel --isolate` (`51db2416c`, shared eval change), M6 formal
    library `M6_4B_NODE=E|F` (`4427e618c`), `m10_formal_select.py`, `m10-formal.sh` (`9a6b5aa87`); node B's frozen
    masters, the gold-free formal panels, CAL698 and the typed-DEV / CSS-pilot gold (the 23:15 rule) copied to E / F.

- 2026-10-01 ≈12:20 UTC+8 (04:20Z) — **Training ≈ 60%; C0 references read; probes built; runtime path committed.**
  - Node F: LH s1–s3 at 452–495 / 787, FB s1–s3 at 509–521 / 787 (ETA ≈ 05:20Z). SELECT700 family macro so far:
    LH .83–.88, FB .87–.90 (N4XF's seeds from Nox were ≈ .88–.90).
  - Node E: LT2 s1–s3 at 344–390 / 787 (ETA ≈ 05:45Z); SELECT700 so far .72–.81 (label-token readout).
  - Retention probes final (`a324f1e2…`, gold `5c674e35…` on nodes A / E / F, private): MMLU 1,265, ARC-Challenge
    254, ARC-Easy 570, GSM8K 1,000. Excluded before any readout: 823 candidates with a 13-gram in TRAIN (821 GSM8K
    train, 2 MMLU) and 1,729 with one in the Index suite (checked on node C; only hit counts left node C).
  - C0 (`4b-C0-e`) read on node E GPU3: typed DEV, CSS pilot, HT-DEV v2, Score5-typed-DEV, `hs1-dev`, PN1 dev; probes
    reading; then the base ceiling (`4b-BASE-e` = LT2-s1's zero-step checkpoint, label-token readout).
  - Tooling: readout / merge / soup / node-A scoring / rules `083604187`; release-runtime `label_token` path
    (`de275f9fb`: `qwen.py` dispatch, builder vendoring + identity; tests). Mirrors on nodes A / E / F.

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

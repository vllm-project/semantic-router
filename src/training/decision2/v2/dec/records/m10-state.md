# Decoder Milestone 10 — state (keep current; newest first)

Assignment: COORDINATION 2026-10-01 11:20 (4B readout × initialisation probe on node E GPU0–3 + node F GPU2–7, 120
GPU-h). Preregistration [`dec-m10-prereg-2026-10-01.md`](dec-m10-prereg-2026-10-01.md) (`2dea44d6f`); data lock
[`dec-m10-datalock-2026-10-01.md`](dec-m10-datalock-2026-10-01.md) (`6a8092977`); amendment
[1](dec-m10-amendment-1-2026-10-01.md) (`04088322b`). Branch `xunzhuo/decision-2-training-dec-m10`, worktree
`/home/xunliu/code/vllm-sr-dev2-dec-m10`.

## Now

- 2026-10-01 ≈14:35 UTC+8 (06:35Z) — **M10 development and formal work done; waiting on hand-offs and on IB1-r2.**
  - Final results: [`dec-m10-results-2026-10-01.md`](dec-m10-results-2026-10-01.md). Two finalists pass items 1–7:
    **`m10-4b-LH`** (C1 candidate; v3 67.34, +4.19 [+0.10, +9.88]) and `m10-4b-NT2` (v3 65.25, +2.10 [+0.49,
    +5.82]). 13.24 of 120 GPU-h.
  - Amendment 3 (`1a5bec9d9`): NT2's first formal step was refused by a finished runner's lease entry; NT2's
    formal then ran once.
  - All M10 GPUs are idle with leases `status=idle` (node E GPU0–3, node F GPU2–7); no M10 container runs.
  - **Next (continuation or coordinator):**
    1. Release engineering builds a release-format package of `m10-4b-LH` (node A
       `formal/m10/pkg/m10-4b-LH`, local only).
    2. The eval custodian runs C1 item 8 (spec draft in the hand-off record; node A holds the package, the formal
       predictions and the frozen cache `formal/m10/m10-4b-LH-cache-frozen`).
    3. IX1 runs the frozen candidate (private).
    4. When IB1-r2 is release-safe, an amendment adds W+IB1 / W+C on the LH recipe (three seeds each; node F
       GPU2–7, node E GPU0–3).
    5. Optional, the coordinator decides: a formal test of LT2 (label-token from the base), and an informational
       FB formal run.

- 2026-10-01 ≈13:45 UTC+8 (05:45Z) — **`m10-4b-LH` passes successor items 1–7; hand-offs written; NT2 finishing.**
  - Wave-1 gates (`m10/select/4b-finalists-w1.json`, node A): finalist `4b-LH`; FB fails the typed Score floor
    (342 < 350); LT2 fails HT-DEV v2 (FLAG −.024).
  - Formal (node F GPU3): v3 67.34 vs 63.15, +4.19 [+0.10, +9.88]; vs Decider 4B +5.46 [+0.28, +8.11], vs Jet v6.2
    +6.97 [+0.98, +11.13]; mlx-diag card-eligible +.0377 [+.0256, +.0498]; items 1–7 PASS
    (`formal/m10/successor/4b-m10-4b-LH.md` on node A). Item 8 pending (needs a release-format package).
  - Records: interim results `0064f6c62`; hand-offs [`dec-m10-handoff-2026-10-01.md`](dec-m10-handoff-2026-10-01.md)
    (release package → C1 item 8 → IX1); integration `39096d7b5`; gist 04 updated.
  - Label-token runtime parity re-run on the merged BF16-resident runtime: PASS (1,600 / 1,600, drift 0.0).
  - NT2 at 623–665 / 787 (ETA ≈ 05:52Z); post chain `post-NT2` on node E GPU3 (soup on CPU, then readouts); then
    node-A scoring and the wave-2 rules (`4b-finalists-w2.json`).

- 2026-10-01 ≈13:05 UTC+8 (05:05Z) — **First-wave seeds done; FB and LH read and scored; LT2 reading; NT2 training.**
  - Per-seed BEST SELECT700: LH .895 / .890 / .905, FB .897 / .906 / .913 (N4XF's seeds from Nox ≈ .88–.90); LT2
    BESTs at the final updates; LoRA merges agree with the adapters on 128 / 128 SELECT rows (drift ≈ .011).
  - Amendment 2 (`88aacb9fb`, before any arm readout): two gate waves; shared reference `4b-C0-e`.
    **Cross-node parity:** `4b-C0-f` = `4b-C0-e` on all seven panels (0 differing decisions, drift 0.0).
  - **Retention ceiling** (probe macro of MMLU / ARC / GSM8K; development diagnostic): base (label-token zero-shot)
    .762 vs C0 .710, +.052 [+.035, +.067] (MMLU +.048, GSM8K +.122, ARC −.015 n.s.).
  - **FB:** HT-DEV v2 −.011 [−.027, +.006] TIE; Score5t clean; retention .787, +.077 [+.063, +.089] vs C0 (+.025
    above the base); typed DEV T .869 vs .704 but Score 342 < floor 350 (362 − 12): fails gate 1 as preregistered.
  - **LH:** HT-DEV v2 −.014 [−.031, +.003] TIE; Score5t clean; retention .770, +.060 [+.047, +.072] vs C0 (level
    with the base). FB − LH: HT-DEV v2 +.003 TIE, retention +.017 [+.008, +.027].
  - **Formal path parity (node F, isolated runner):** C0 collected in 5.8 min, 0 differing answers on typed FINAL,
    CSS15 and public 231 vs the stored bar run → the bar run stands for M10 finalists.

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
| LH / FB / LT2 / NT2 training (incl. preflights); LT zero-step | 2.984 / 2.654 / 3.074 / 2.739; 0.008 |
| Readouts, merges, runtime parity | 1.330 |
| Formal (C0 parity, LH, NT2 incl. mlx-diag) | 0.450 |
| **Total** | **13.24** |

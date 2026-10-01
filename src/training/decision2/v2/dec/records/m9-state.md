# Decoder Milestone 9 — state (keep current; newest first)

Assignment: COORDINATION 2026-10-01 10:30 (4B HR2 efficacy pilot, NOT releasable: full seeds from Nox + N4XF + an HR2
block vs a matched control, screened with HT-DEV v2). Budget 16 GPU-h. GPUs: node A GPU6–7 only (9B-owned, lent).
Preregistration [`dec-m9-prereg-2026-10-01.md`](dec-m9-prereg-2026-10-01.md) (`f22a5797c`, committed 10:37 UTC+8;
its "written ≈11:10" line is a typo for ≈10:35); data lock [`dec-m9-datalock-2026-10-01.md`](dec-m9-datalock-2026-10-01.md)
(`2f20c2223`, PASS); [amendment 1](dec-m9-amendment-1-2026-10-01.md) (`e7f8e9858`, committed 11:10:50 UTC+8 =
03:10:50Z, before any arm readout; its "written ≈11:35" line should read ≈11:10). Gist 04 entry.

## Now

- 2026-10-01 12:23 UTC+8 (04:23Z) — poll: H9-s1 step 1,042 / 1,128, H9-s2 986 / 1,128; g6b waiting on the GPU6
  lock (expected); post chain waiting. No failures since amendment 2.

- 2026-10-01 ≈12:00 UTC+8 (03:58Z) — **Amendment 2 applied: H9 arm cap 5.8 GPU-h; GPU6 chain replaced by `g6b`.**
  - [Amendment 2](dec-m9-amendment-2-2026-10-01.md) (`29057a4de`, before any H9 readout; its "≈12:05 / 04:05Z" means
    ≈11:57 / 03:57Z): H9 seeds cost ≈ 1.75–1.8 GPU-h, so the 4.8 cap would have refused H9-s3; C9's unused budget
    funds it; totals unchanged.
  - 03:57Z: old g6 chain bash + watchdog stopped (its flock exited with it); H9-s1 `drive_arm.sh` pid 4082191 and
    container `dec-m9-H9-s1-full` kept running (step ≈ 665 / 1,128). `g6b` (mirror `29057a4de…`, `M9_CAP=5.8`, flock
    pid 4163090) waits on `gpu6.flock` until H9-s1's process tree exits (≈ 04:45Z), then adopts it via
    `chains/adopt-m9-H9-s1.pid`, runs the early read, STOPs C9-s2 and starts H9-s3 (≈ 04:50Z → ≈ 06:25Z).
  - g7 unchanged: H9-s2 (≈ 04:52Z) → C9-s3 STOPPED → C9 no soup → ends. Post chain armed on GPU7.

- 2026-10-01 ≈11:55 UTC+8 (03:55Z) — **Control line done; formal path exact; waiting for H9.**
  - **L-N7C (control, M7's N7C on the M9 path; identical to M7's readings):** α 1 HT-DEV v2 vs `4b-I` −.0081
    [−.0217, +.0057] TIE, typed DEV T .648 (Choice 440 < 477, Score 338 < 350: fails the typed floors), Score5t no
    flag (top .41), HR2 DEV .693 (+.005); α ½ +.0010 [−.0093, +.0115] TIE, T .703 (passes), Score5t no flag, HR2 DEV
    .689.
  - **Formal-path parity EXACT** (`m9/formal/m9-ref-N4XF/PARITY.json`): 0 category changes on typed FINAL / CSS15 /
    public 231 vs `dev2-4b-t1-derived`; v3 63.151; the stored bar stays the paired reference. Its mlx-diag collected.
  - **H9:** s1 / s2 at step ≈ 470 of 1,128 (even8 checkpoints every 141 updates); s1 ≈ 04:40Z, s2 ≈ 04:47Z, s3
    (GPU6) ≈ 06:25Z, soup ≈ 06:35Z. No co-tenant jobs on GPU6 during s3. Post chain (GPU7) armed.

- 2026-10-01 ≈11:25 UTC+8 (03:25Z) — **H9 training (3 seeds); control = M7's N7C (amendment 1); post chain armed.**
  - **C9 stopped:** m9-C9-s1 preflight FAIL 03:04Z on `zero_trainer_cross_process` (698 / 700, drift 0.0187; every
    other gate passed; cold shared autotune cache filled by both chains at once). No rerun (rules). E1 does not apply.
  - **H9:** s1 (GPU6) and s2 (GPU7) passed preflight (03:04Z, 03:09Z) and are in their full runs (≈ 04:30Z); GPU6
    then runs H9-s3 (≈ 06:00Z); the soup follows (`m9/status/H9.DONE`).
  - **Readout path parity (refs, done 03:12Z):** `4b-I` on the M9 node-A path = the stored node-B readouts, 0 answer
    differences and drift 0.0 on typed DEV, CSS pilot, HT-DEV v2, `hs1-dev`, Score5-typed-DEV; HT-DEV v2 = the eval
    reference (Δ 0.0). `4b-I`: Score5-typed-DEV check no flag (top share .39); HR2 DEV family macro .688.
  - **Control line L-N7C** (GPU7 co-tenant, since 03:14Z): N7C soup copied (list `9bcc0d10…` verified); α ½ rebuilt =
    M7's finalist weights (only `decision_config.json` paths differ). Readouts running.
  - **Formal parity run** (GPU6 co-tenant, since 03:14Z): `m9-ref-N4XF` (DEV2.0-4B weights, T = 1) on the node-A
    formal path (dbe5f32b, copy of `cache-frozen` `f6d0f920…`); typed FINAL collected, CSS15 running; then PARITY.json
    vs `dev2-4b-t1-derived` and its mlx-diag.
  - **Post chain** (`m9-post.sh`, mirror `3adaa9cc1…`, pid 4131371, GPU7): waits for H9.DONE → `line H9` → `score` →
    `rules` → (pick only) formal select / smoke / finalist / score / mlx / readout. Markers `m9/post/*.DONE|FAILED`,
    log `m9/logs/post.log`.

## Plan

| Step | Where | Status |
| --- | --- | --- |
| Prereg, tooling, data lock | | done |
| Chains (H9 s1–s3; C9 stopped) | node A GPU6–7 | running; H9 soup ≈ 06:10Z |
| References + readout parity | node A GPU6 | done (exact) |
| Control line L-N7C (amendment 1) | node A GPU7 | running |
| Formal-path parity run | node A GPU6 | running |
| H9 line, scoring, rules, formal | node A GPU7 (post chain) | armed |
| Results record, gist 04, merge | workstation | after the post chain |

## Operations (node A, from the newest mirror `S=/data/dev2/src/<mirror>/src/training/decision2`)

- Liveness: `m9/chains/chain-g{6,7}.pid`, `m9/post/post.pid` + `ps -p`; `docker ps | grep -E 'dec-m9|dev2-dec-gpu'`.
  Never `pgrep -f`.
- Logs: `m9/OPERATIONS.log`, `m9/logs/{chain-g6,chain-g7,post,line-N7C,formal-parity}.log`, `m9/arms/OPERATIONS.log`,
  `m9/lines/4b/OPERATIONS.log`, `m9/formal/OPERATIONS.log`.
- If the post chain stops on a failed step: record it; do not rerun that step.

## GPU-hours (so far, approximate)

| Item | GPU-h |
| --- | ---: |
| C9-s1 (zero-step, one-step, gate) | ≈ 0.10 |
| H9 s1 / s2 (running) | — |
| References (6 panels) | 0.21 |
| **Total (cap 16)** | running |

# 0.6B Milestone 8 — working state (worker-maintained; newest first)

Goal: a releasable successor of DEV2.0-0.6B (`m4-t-a7-soup`, revision `99c4e799…`, post-key v3 43.54, T = 1, 8K)
from `m6-mxcx-soup` + per-level Score offsets fitted on the score5t-dev fit half (COORDINATION 16:40 note).
Budget ≤ 6 GPU-h on node A GPU0–1. Nothing to HF during the milestone.

## Log

- 2026-09-29 17:25 UTC+8 — **development done (CPU only, node A):** tools `95cf53e96`, records `c76da0cac`
  (`records/m8/dev/`). Eligible: `s5-b05` (check-half acc .360, top share .625, CHK_5 Δ −.011 [−.028, +.007]) and
  `s5h-b05` (.310, .820, CHK_5 +.006 [−.010, +.022]) → **finalists in that order**. λ = 1 fits fail D3a (CHK_5
  significantly worse); λ = 0 fits stay COLLAPSE on the check half (.95/.96). 264 A7q FIT rows removed as
  Score5-DEV panel rows. Runtime support `c6e993f48` (qwen.py + build.py vendoring fix + 13 tests) merged
  (`2713abf38`); integration merged (`ebb2b8bff`, pushed to both branches). JevBench decision (17:15 note): R7 =
  `v2.eval.gates public231 --left <successor> --right <current revision>` must not return REGRESSION; nobody selects
  on public 231. Next: package, D4 replay, formal runs, gates.
- 2026-09-29 16:55 UTC+8 — preregistration `m8-prereg-2026-09-29.md` written before any fit or GPU job: six
  L = 5 offset fits of `m6-mxcx-soup` (λ ∈ {1, ½, 0} × human weight {0, 1}), gates D1–D3, ≤ 3 finalists by
  check-half accuracy, D4 replay, formal, successor rule R1–R7. Code maps done (offsets tool, formal pipeline,
  score5t readout, gates, runtime/build). Found: the release builder never vendors `score_bias.py` although the
  vendored `infer.py` imports it since `8730d9413`; CSS15 has no Score items, so offsets cannot move axis H.
- 2026-09-29 16:36 UTC+8 — worker started. Read COORDINATION (mandate; notes 16:40, 16:05, 16:00, 14:05;
  Score5-typed-DEV v1 runner rules), the M7 record and prereg, the score5t-dev record and hand-off, gist 03.
  Merged `origin/xunzhuo/decision-2-training` (fast-forward to `aa65f0844`); tree clean.
  Next: code map of the offsets tool, formal scripts, readout formats, gates and runtime; then the preregistration.

## Status

| Step | State |
| --- | --- |
| 1. Preregistration | `e6a5f4515` (08:55Z, before any fit or GPU job) |
| 2. Runtime offsets (`v2/release/runtime/qwen.py`) | `c6e993f48` (+ builder vendoring fix, 13 tests), merged `2713abf38` |
| 3. Development fit/check (CPU) | done: finalists `s5-b05`, `s5h-b05` (`c76da0cac`) |
| 4. Formal runs + mlx-diag | pending |
| 5. Successor rule | pending |
| 6. Release hand-off | pending |
| 7. Records, gist 03, merge | pending |

## GPU-hours (node A)

| Job | GPU | Start–end (UTC) | GPU-h |
| --- | --- | --- | ---: |

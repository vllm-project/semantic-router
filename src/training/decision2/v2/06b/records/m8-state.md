# 0.6B Milestone 8 — working state (worker-maintained; newest first)

Goal: a releasable successor of DEV2.0-0.6B (`m4-t-a7-soup`, revision `99c4e799…`, post-key v3 43.54, T = 1, 8K)
from `m6-mxcx-soup` + per-level Score offsets fitted on the score5t-dev fit half (COORDINATION 16:40 note).
Budget ≤ 6 GPU-h on node A GPU0–1. Nothing to HF during the milestone.

**Outcome: SUCCESSOR `m8-s5-b05`** (v3 48.64, +5.09 [+2.65, +9.30] vs released; R1–R7 pass). Results
`m8-results-2026-09-29.md`; release hand-off `m8-handoff-2026-09-29.md`. M8 GPU total 0.191 GPU-h.

## Log

- 2026-09-29 18:20 UTC+8 — results, hand-off, state and gist 03 written; merged into the integration branch.
  GPU0–1 idle (owner files set to idle).
- 2026-09-29 18:12 UTC+8 — **staging dry run passed** (no HF; `5e5e3afdf`, `76840256c`): the builder packages the
  path-free offsets bound by value to the scored run; parity at tolerance 0 on typed FINAL 1,600, public 231 231,
  CSS15 400, mlx-diag 300 (0 changes); negative control shows offsets active. 0.036 GPU-h.
- 2026-09-29 17:59 UTC+8 — **release blocker found and fixed:** the scored `score_bias.json` has node paths in its
  `fit` block, which the package screen refuses, and the builder bound offsets by file hash. Fix `c5b499812`
  (builder binds by value; shared module) + `m8_scorebias publish` and the path-free file `725c16f9…`
  (`2a252a1b7`); merged (`d2159697c`, integration).
- 2026-09-29 17:48 UTC+8 — **formal done** (`49e632bfc` tools, `63c555d33` records): D4 passes for both finalists;
  both pass R1–R7; release choice `m8-s5-b05` (the alternate is not significantly better: −0.18 [−0.55, +0.19]).
  0.155 GPU-h.
- 2026-09-29 17:25 UTC+8 — **development done (CPU only, node A):** tools `95cf53e96`, records `c76da0cac`
  (`records/m8/dev/`). Eligible: `s5-b05` (check-half acc .360, top share .625, CHK_5 Δ −.011 [−.028, +.007]) and
  `s5h-b05` (.310, .820, CHK_5 +.006 [−.010, +.022]) → **finalists in that order**. λ = 1 fits fail D3a (CHK_5
  significantly worse); λ = 0 fits stay COLLAPSE on the check half (.95/.96). 264 A7q FIT rows removed as
  Score5-DEV panel rows. Runtime support `c6e993f48` (qwen.py + build.py vendoring fix + 13 tests) merged
  (`2713abf38`); integration merged (`ebb2b8bff`, pushed to both branches). JevBench decision (17:15 note): R7 =
  `v2.eval.gates public231 --left <successor> --right <current revision>` must not return REGRESSION; nobody selects
  on public 231.
- 2026-09-29 16:55 UTC+8 — preregistration `m8-prereg-2026-09-29.md` written before any fit or GPU job: six
  L = 5 offset fits of `m6-mxcx-soup` (λ ∈ {1, ½, 0} × human weight {0, 1}), gates D1–D3, ≤ 3 finalists by
  check-half accuracy, D4 replay, formal, successor rule R1–R7. Code maps done (offsets tool, formal pipeline,
  score5t readout, gates, runtime/build). Found: the release builder never vendors `score_bias.py` although the
  vendored `infer.py` imports it since `8730d9413`; CSS15 has no Score items, so offsets cannot move axis H.
- 2026-09-29 16:36 UTC+8 — worker started. Read COORDINATION (mandate; notes 16:40, 16:05, 16:00, 14:05;
  Score5-typed-DEV v1 runner rules), the M7 record and prereg, the score5t-dev record and hand-off, gist 03.
  Merged `origin/xunzhuo/decision-2-training` (fast-forward to `aa65f0844`); tree clean.

## Status

| Step | State |
| --- | --- |
| 1. Preregistration | `e6a5f4515` (08:55Z, before any fit or GPU job) |
| 2. Runtime offsets (`v2/release/runtime/qwen.py`) | `c6e993f48` (+ builder vendoring fix, 13 tests), value binding `c5b499812` |
| 3. Development fit/check (CPU) | done: finalists `s5-b05`, `s5h-b05` (`c76da0cac`) |
| 4. Formal runs + mlx-diag | done (`63c555d33`) |
| 5. Successor rule | **`m8-s5-b05` passes R1–R7** (alternate `m8-s5h-b05` passes too) |
| 6. Release hand-off | `m8-handoff-2026-09-29.md`; staging dry run passed (`76840256c`) |
| 7. Records, gist 03, merge | done |

## GPU-hours (node A)

| Job | GPU | Wall | GPU-h |
| --- | --- | ---: | ---: |
| D4 `s5-b05` / `s5h-b05` | 0 / 1 | 32 s / 32 s | 0.018 |
| Formal `s5-b05` / `s5h-b05` | 0 / 1 | 191 s / 187 s | 0.105 |
| mlx-diag `s5-b05` / `s5h-b05` | 0 / 1 | 57 s / 58 s | 0.032 |
| Staging dry-run parity | 0 | 131 s | 0.036 |
| **Total** | | | **0.191** |

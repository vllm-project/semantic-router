# Milestone 2 deferred peers and Jev reference status

Eval & peers track, 2026-09-28, amendments A4/A4.1. **Post-key same-panel** evidence;
reports in [`m2-reports/`](m2-reports/). Paired intervals: joint v3 bootstrap, 5,000
draws, seed 20260927, against the tier's own 1.0 model on the same node.

| Model @ pinned revision | Tier / node | Loaded params | v3 | T | H | Choice / Noul / Score | Public 231 (E/S/H) | Invalid CSS / pub | Δ v3 vs own 1.0 [95% CI] | mlx-diag type-macro (non-EN) | Dev P |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: |
| Intern-Decision-0.8B `85a0cc5a` | 0.8B / A | 852,985,920 | 43.535 | .4956 | .3824 | 495 / 431 / 88 | 164 (47/57/60) | 115 / 0 | +0.989 [−2.328, +6.084] vs Eos1 | 58.3 (57.4) | 35.73 |
| Kev-0.8B `9a45d25e` | 0.8B / A | 752,917,824 | 43.217 | .4794 | .3896 | 489 / 407 / 82 | 147 (48/58/41) | 13 / 0 | +0.670 [−3.027, +6.828] vs Eos1 | 64.8 (63.9) | 33.50 |

Both are within noise of Eos1 (42.547); the 0.8B tier now has three peers
(Intern 43.535, Kev 43.217, JPT-0.8B 40.085). Intern's 115 CSS invalids are native
8,192-token rejections; Kev's 13 are its strict state limit. As an out-of-sample proxy
check, v3 ≈ 19.12 + 0.629·P predicts 41.6 (Intern) and 40.2 (Kev), errors +1.9 and +3.0,
inside the calibrated ±3.

Q2 (Intern) first stopped at engine load (14.1 s, no prediction) because the release's
dataclasses need their module in `sys.modules`; the adapter was fixed with a regression
test and Q2b ran once. Shared-adapter changes (separate commits with tests): Kev size
table, Eikos size table, new `inference/intern_decision.py`; existing defaults unchanged.

Eikos-27B (BF16 sibling `103a5647` of the board's FP8 artifact) runs on node B; its row is
added when the run finishes. Still deferred: this-that 1.2, Jet v6.2, Nimble v2,
Jebadiah 27B, Hopper (G); Rune needs a parity-checked ROCm path.

## Jev reference (approved 2026-09-28 13:40, private)

The official API run on v3 and public 231 completed (model pinned `jev-1.13.0`, which the
`jev-latest` alias resolved to; about 0.15 s per request; zero failed requests). Its
predictions, seal and report stay in private storage on node A and are not published in
this branch, the gist or any card, per the coordinator's rule. It is labelled separately
from open-weight ranks and is never a training target.

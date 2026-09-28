# Development-proxy calibration against post-key v3 (16 models)

Eval & peers track, Milestone 2 item a; preregistered as amendment A3. Inputs: fresh
typed DEV (1,600) and CSS pilot (1,430) readouts with each model's formal adapter,
package and runtime (DEV2.0-0.6B reuses its identity-matched earlier readout;
AutoJev's runs on node B), and each model's published v3 aggregate. No v3 item,
label or per-item output entered any proxy. Raw output:
[`m2-reports/proxy-calibration-v1.json`](m2-reports/proxy-calibration-v1.json).
Tool: `python3 -m v2.eval.proxy_calibration`.

## Result

| Proxy | Spearman | Kendall τ-b | LOO v3 error MAE / RMSE / max | Pairs in v3 order | Same-tier pairs |
| --- | ---: | ---: | --- | ---: | ---: |
| **P = 100·√(T_dev·H_pilot)** | 0.941 | 0.817 | 3.13 / 3.67 / 7.35 | 109/120 | 14/18 |
| mean(T_dev, H_pilot) | 0.956 | 0.850 | 3.30 / 3.73 / 7.23 | 111/120 | 14/18 |
| P_type = 100·√(mean typed-DEV type acc · H_pilot) | 0.926 | 0.800 | 2.98 / 3.55 / 6.67 | 108/120 | 14/18 |
| T_dev alone | 0.938 | 0.817 | 4.28 / 4.77 / 7.63 | 109/120 | 11/18 |
| H_pilot alone | 0.897 | 0.750 | 3.63 / 4.25 / 8.44 | 105/120 | 14/18 |
| CSS pilot micro accuracy | 0.893 | 0.762 | 3.86 / – / 8.90 | 105/120 | 11/18 |
| typed-DEV Choice / Score / Noul accuracy | 0.894 / 0.844 / 0.542 | 0.750 / 0.650 / 0.477 | 5.30 / 6.44 / 8.79 (MAE) | 105 / 97 / 88 | 12 / 8 / 12 |

P, mean(T,H) and P_type are indistinguishable at n = 16; single components are
clearly worse (typed-DEV Noul and Score especially). Linear map across models:
**v3 ≈ 19.12 + 0.629·P**.

**Noise.** Resampling the development panels (typed groups within family, pilot items
within task; 1,000 draws) gives a per-model P standard deviation of 1.31 (0.95–1.48),
so a difference of two checkpoints has SD ≈ 1.9 P points before seed variance (the
0.6B track saw a single seed change move SELECT by 38/700). Order agreement by gap:
|ΔP| ≥ 2 SD of the difference (≈ 3.7 P points): **102/109 = 94%**; 1–2 SD: 3/4;
below 1 SD: 4/7.

**Where P fails (all four same-tier misses).** Kai1 vs Lex (P favours Lex by 4.0, v3
favours Kai1 by 4.9: typed DEV rewards Lex's typed gains, T_dev 0.384 vs 0.266, yet both
have the same FINAL T 0.3619 and Lex loses CSS transfer); and three JPT-vs-Decision/
Decider pairs (JPT's pilot H is high relative to its CSS15 H). Typed DEV families differ
from FINAL families, and three pilot tasks do not track 15-task transfer for every
lineage.

## Recommended rule for training tracks

1. Rank checkpoints by **P** (unchanged definition, so earlier readouts stay
   comparable); expect a v3 prediction error of about ±3 points (worst seen 7.4).
2. Treat |ΔP| < 4 as a tie. Within a tie, send **both** checkpoints (or both seeds) to
   the formal runner; do not break ties on SELECT, T_dev or any single typed type.
3. Never select on T_dev alone or typed-DEV Noul/Score, and flag any checkpoint whose
   T_dev rises while H_pilot falls (the Kai→Lex pattern) for a formal run before
   discarding its parent.
4. Across architectures or lineages, use P only for shortlisting; the final
   comparison is the formal paired v3 interval.

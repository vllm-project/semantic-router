# Development proxy v2 — calibration on 51 models (eval & peers, 2026-09-29)

Preregistered in [`m5-proxy-v2-prereg-2026-09-29.md`](m5-proxy-v2-prereg-2026-09-29.md) (commit
`79482d863`, pushed before any candidate proxy was computed). Tool:
`python3 -m v2.eval.proxy_calibration extract|analyze`. Raw outputs, aggregates only:
[`m5-proxy-v2/features.json`](m5-proxy-v2/features.json) and
[`m5-proxy-v2/analysis.json`](m5-proxy-v2/analysis.json); model list:
[`m5-proxy-v2/spec.json`](m5-proxy-v2/spec.json). No v3 item, label or per-item output and nothing
from JevArena-C1 entered any proxy; post-key v3 composites are only the target.

## Result

- **No candidate proxy beats P, so proxy v2 is P with an unchanged definition, recalibrated.**
  P = 100·√(T_dev·H_pilot), where H_pilot is the CSS-pilot median task macro-F1. The new linear
  map on 51 models is **v3 ≈ 22.39 + 0.576·P**, with leave-one-out error MAE 2.6, RMSE 3.1 and
  worst case 6.6.
  - The best alternative has only a 0.38 bootstrap probability of ordering more of the
    preregistered decision pairs correctly (same tier, at least one 2.0 candidate,
    |Δv3| ≥ 2).
  - Under the preregistered selection rule, nothing qualifies.
- **Tie band: |ΔP| < 8** within a tier, replacing |ΔP| < 4.
  - At |ΔP| ≥ 8, P's order matched v3 in 39 of 43 within-tier decision pairs (91%), and only one
    pair (2%) was reversed by ≥ 2 v3 points.
  - Between 4 and 8, which the old rule treated as decided, P matched in 76% of pairs and 10%
    were reversed by ≥ 2 points.
  - Below 4, P matched in 63% and 20% were reversed.
- **No proxy separates close within-tier candidates reliably.**
  - 67 of the 70 finalist pairs (same track, both 2.0 candidates) lie inside |ΔP| < 8. There P
    matches v3 in 69% of pairs and reverses 16% by ≥ 2 points.
  - The three-task mean, the per-type terms and the fitted combination do no better.
  - **The rule stays: ties go to the formal runner.**

## Model set

51 distinct weight sets:

- 24 comparators: the v1 matrix and the 8 out-of-sample peers. The node-B 27B peers are their
  kernel-image `dbe5f32b` re-collections.
- 27 candidates, all with a formal post-key v3 run and a typed-DEV + CSS-pilot readout of the
  same weights. The readout and the formal run share the model hash.
  - 0.6B soups: T, V2, X and Z.
  - 0.8B: E8F s1–s3 and soup, B8F s1–s2, and the E8V soup.
  - 2B: the S2T soup.
  - 4B: X2, X4R s1–s2, X4K s1–s2, and the N4T, N4J, N4L and N4LKr soups.
  - 9B: L2 and M3 B-s1.
  - 27B: C0 (BEST368), M2-C1, M2-S1 and M2-K1.

By tier: 0.6B 9, 0.8B 11, 2B 5, 4B 14, 9B 5, 27B 7. Pairs: 223 same-tier, 185 decision (129 with
|Δv3| ≥ 2, of which 60 close at 2–5 and 69 clear at ≥ 5; 56 ties below 2), and 70 finalist (42
with |Δv3| ≥ 2).

Checks:

- Every feature was recomputed on node A from the prediction files with one scorer (frozen gold,
  hash-verified). Each candidate's recomputed P equals the value its track recorded, to the
  recorded precision.
- The v1 models reproduce exactly: features and noise within 1e-14, and every v1 number in S4.
- The two recorded Lux1 T_dev values (.868 and .876) come from the prediction files themselves
  (native vs shared renderer), not the scorer: typed-DEV Noul is graded by probability > 0.5.
- The 11 node-B inputs were streamed to node A with matching sha256.

## Comparison (main set, 51 models)

"Decision" means same tier with at least one candidate; "finalist" means same track with both
candidates. Noise is the median per-model panel-bootstrap SD converted to v3 units. The bands
are in each proxy's own units (L2 is in v3 points): "empirical" is the preregistered rule, and
"model" is 1.28·σ/β at a 10% reversal risk. P(>P) is the paired cluster-bootstrap probability of
ordering more primary pairs correctly than P.

| Proxy | Spearman | Within-tier r | LOO v3 error MAE / RMSE / max | All pairs | Same tier | **Decision, \|Δv3\| ≥ 2 (primary)** | Close 2–5 | Finalist, \|Δv3\| ≥ 2 | Noise | Band empirical / model | P(>P) |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **P** (v1) | 0.939 | 0.63 | 2.61 / 3.13 / 6.55 | 1146/1275 | 165/223 | **106/129** | 44/60 | 31/42 | 0.75 | **8** / 10.8 | — |
| P_mean3 | 0.946 | 0.64 | 2.53 / 3.03 / 6.85 | 1153/1275 | 159/223 | 105/129 | 41/60 | 31/42 | 0.57 | 9 / 9.8 | 0.37 |
| P_type | 0.932 | 0.67 | 2.58 / 3.08 / 7.05 | 1141/1275 | 162/223 | 104/129 | 43/60 | 31/42 | 0.81 | 6 / 8.6 | 0.12 |
| P_type_mean3 | 0.942 | 0.67 | 2.64 / 3.05 / 6.98 | 1145/1275 | 157/223 | 102/129 | 38/60 | 29/42 | 0.61 | 6 / 8.4 | 0.22 |
| P_CS_mean3 (no typed Noul) | 0.934 | 0.64 | 2.64 / 3.17 / 8.20 | 1138/1275 | 153/223 | 98/129 | 37/60 | 24/42 | 0.54 | 11 / 11.0 | 0.16 |
| A_med = (T+H_med)/2 | 0.945 | 0.62 | 2.55 / 3.08 / 6.78 | 1153/1275 | 164/223 | 105/129 | 44/60 | 29/42 | 0.67 | 11 / 11.5 | 0.33 |
| A_mean3 = (T+H_mean3)/2 | 0.946 | 0.64 | 2.49 / 3.03 / 7.88 | 1148/1275 | 158/223 | 105/129 | 40/60 | 30/42 | 0.53 | none / 10.5 | 0.38 |
| L2 = a + b·T + c·H_mean3 (LOO) | 0.941 | 0.60 | 2.55 / 3.07 / 7.53 | 1148/1275 | 159/223 | 105/129 | 41/60 | 31/42 | 0.57 | none / 7.3 | 0.37 |
| *T_dev alone* | 0.917 | 0.51 | 3.03 / 3.89 / 11.55 | 1125/1275 | 144/223 | 89/129 | 34/60 | 18/42 | 0.55 | — | — |
| *H_pilot (median) alone* | 0.884 | 0.51 | 3.47 / 4.13 / 8.59 | 1083/1275 | 146/223 | 90/129 | 37/60 | 28/42 | 1.16 | — | — |
| *H_mean3 alone* | 0.911 | 0.58 | 3.30 / 3.92 / 7.92 | 1114/1275 | 152/223 | 96/129 | 38/60 | 35/42 | 0.92 | — | — |
| *typed-DEV Choice / Noul / Score* | 0.82 / 0.54 / 0.72 | 0.37 / 0.13 / 0.38 | RMSE 5.7 / 8.2 / 6.2 | — | 149 / 122 / 120 | 95 / 69 / 74 | — | 26 / 25 / 17 | — | — | — |

Italic rows are references and were never eligible.

- **Primary-metric intervals.** P's 95% bootstrap interval is [0.68, 0.94]. Every candidate's
  paired difference to P spans 0: P_mean3 [−0.09, +0.09], P_type [−0.07, +0.03].
- **Close pairs (2–5 v3 points):** 73% for P, versus 62–73% for the alternatives.
- **Finalist pairs:** P's 95% interval is [0.46, 0.97].
- **H_mean3 alone on finalist pairs (exploratory, not preregistered):** 35/42, but the paired
  difference to P spans 0 ([−0.19, +0.38]) and it is worse on the primary metric (96/129).

## Tie band for P

Grid over the 185 within-tier decision pairs. "Reversed ≥ 2" means the v3 order is opposite to
P's by at least 2 points.

| \|ΔP\| ≥ | Pairs | P order = v3 order | Reversed ≥ 2 |
| ---: | ---: | ---: | ---: |
| 1 | 169 | 76% | 11% |
| 2 | 144 | 80% | 8% |
| 4 | 110 | 82% | 7% |
| 6 | 77 | 79% | 6% |
| 7 | 56 | 86% | 5% |
| **8** | **43** | **91%** | **2%** |
| 9 | 32 | 94% | 0% |
| 10 | 26 | 92% | 0% |

- **Old v1 claim.** v1 reported 94% agreement at |ΔP| ≥ 4, but that figure was over all pairs,
  which are mostly cross-tier. Within tiers the same cut gives 82%.
- **Model-based check.** Within tiers, one P point is worth only 0.49 v3 points (0.58 across
  tiers). The residual SD of pair differences is 4.2 v3 points, so a normal model reaches a 10%
  reversal risk only at |ΔP| ≈ 10.8. Treat 8 as the minimum band.
- **v3's own noise.** Part of this residual is the target's own sampling noise: paired v3 CIs of
  close candidates span about ±3. A development proxy cannot resolve pairs closer than that, and
  the formal paired interval has to decide them.

Costly reversals at |ΔP| ≥ 4 (8 of 110 pairs):

- **Involving JPT:** JPT-0.8B over B8F-s2 (+6.0 / −3.9), JPT-4B over N4L (+8.1 / −4.3) and over
  N4LKr (+7.9 / −5.0).
- **Decider 2B over the S2T soup:** +5.1 / −3.9.
- **Nox1 under the N4J soup:** −7.7 / +2.3.
- **GLiNER2.5 under the 0.6B V2 soup:** −4.1 / +3.4.
- **Finalist pairs:** the 0.6B T soup under the V2 soup (−6.9 / +4.4), and E8F-s1 under B8F-s1
  (−4.4 / +6.7).

Leave-one-out residuals show lineage offsets that no scalar proxy removes:

| Group | Mean v3 − predicted |
| --- | ---: |
| JPT peers | −4.5 |
| 27B LoRA arms | −4.2 |
| Decoder candidates | +1.5 |
| 9B candidates | +1.6 |
| Other comparators | +0.2 |

## The three-task pilot mean (the 02:40 suggestion)

- **0.6B:** it orders the decision pairs better, 19/20 vs 16/20 for P. It also brings the V2-soup
  vs T-soup case inside the band (ΔP_mean3 +3.3 vs ΔP +6.9; v3 −4.4), though still with the wrong
  sign.
- **Other tiers:** it does not help and sometimes hurts: 9B 2/4 vs 4/4, 27B 16/17 vs 17/17, 4B
  38/49 vs 39/49.
- **Without the four 0.6B soups that motivated it (S1):** it is worse, 86/109 vs 90/109.
- **Real gains:** lower panel noise (0.57 vs 0.75 v3 SD) and a slightly better LOO RMSE (3.03 vs
  3.13). Neither improves within-tier ordering. Not adopted. The per-task pilot F1 stay in every
  `dev_readout` output for inspection.

## Per-type typed-DEV terms

- **Equal-type mean (P_type):** the highest within-tier r (0.67) and the smallest empirical band
  (6), but it orders fewer primary pairs than P (104 vs 106; P(>P) 0.12).
- **Dropping typed-DEV Noul (P_CS_mean3):** the worst candidate (98/129; finalist 24/42).
- **Single typed types remain unusable for selection:** Noul 69/129 and Score 74/129.

## Named cases

The last column says whether the pair falls inside the new band (|ΔP| < 8).

| Pair | ΔP | ΔP_mean3 | ΔP_type | Δv3 | Inside band |
| --- | ---: | ---: | ---: | ---: | --- |
| Kai1 vs Lex (12:30) | −4.0 | −5.1 | −3.9 | +4.9 | yes |
| 4B X2 vs Nox1 (16:05) | +3.7 | +3.3 | +3.4 | −0.5 | yes |
| 0.6B V2 soup vs T soup (21:30) | +6.9 | +3.3 | +6.4 | −4.4 | yes (old band: no) |
| 0.6B X soup vs V2 soup | −2.4 | +0.9 | −2.5 | +6.2 | yes |
| 0.6B Z soup vs T soup | +6.2 | +6.6 | +6.0 | +1.9 | yes |
| 0.8B E8F soup vs Eos1 | +10.2 | +14.2 | +10.0 | +7.7 | no (correct) |
| 0.8B B8F-s1 vs B8F-s2 | +7.5 | +8.3 | +5.5 | −0.9 | yes |
| 0.8B E8F-s2 vs E8F-s1 | +1.7 | −2.9 | +1.2 | −2.4 | yes |
| 2B S2T soup vs Sol1 | +4.0 | +4.5 | +3.1 | +7.9 | yes |
| 4B N4J vs N4L soup | +1.8 | +1.3 | +1.5 | −4.7 | yes |
| 4B N4LKr vs N4T soup | +0.0 | −0.5 | −0.2 | +3.0 | yes |
| 9B L2 vs Lux1 | +1.6 | +1.8 | +1.7 | −0.5 | yes |
| 27B M2-S1 vs M2-C1 | +3.7 | +1.9 | +4.7 | +2.6 | yes |
| 27B M2-K1 vs M2-S1 | −0.6 | +1.1 | −0.4 | −2.4 | yes |

## Sensitivity (not used for selection)

| Set | Models (candidates) | P primary | P finalist | P band empirical / model | Best alternative on primary |
| --- | --- | ---: | ---: | --- | --- |
| Main | 51 (27) | 106/129 | 31/42 | 8 / 10.8 | P_mean3, A_med, A_mean3, L2 105/129 |
| S1: no 0.6B soups | 47 (23) | 90/109 | 31/39 | 9 / 11.2 | A_med 90/109 (tie), P_type 89/109, P_mean3 86/109 |
| S2: same-renderer 1.0 controls | 51 (27) | 106/125 | 31/42 | 8 / 10.7 | 105/125 |
| S3: same runtime only | 39 (15) | 46/60 | 5/11 | 7 / 11.6 | A_mean3 50/60 (mean variants 49–50/60) |
| S4: the v1 spec | 16 (0) | 109/120 all pairs, 14/18 same tier (= v1) | — | — | — |

S3 keeps only readouts at the formal limit and on a kernel image. It is the one set where the
mean-based variants look better, but with 60 primary and 11 finalist pairs this is not evidence
for a switch. It does suggest reading candidates at their formal package limit.

## Rule for training tracks (supersedes the v1 tie band)

1. **Rank checkpoints by P** (unchanged definition, so every earlier readout stays comparable).
   Predicted v3 ≈ 22.4 + 0.58·P, with an error of about ±3.
2. **Within a tier, |ΔP| < 8 is a tie.** Send every candidate within 8 P points of the best to
   the formal runner and decide on the paired v3 interval. Only a lead of ≥ 8 lets P drop a
   candidate without a formal run.
3. **A track's own seeds, soups and sibling arms are almost always ties** (67 of 70 pairs). P
   cannot choose among them; use it only to discard clear losers.
4. **Never select on T_dev, H_pilot or a single typed-DEV type alone.** The three-task mean and
   the per-type terms do not improve within-tier ordering.
5. **Across lineages, P is a shortlist only.** It over-predicts JPT peers and the 27B LoRA arms
   by about 4–5 v3 points.
6. **Read development panels at the candidate's formal package limit and image** where feasible,
   since 12 of the 27 candidates were read at a lower limit or on the kernel-less image.

## Compute and reproduction

- **Compute:** CPU only. The extract took about 1 minute on node A (16 processes, mirror
  `1baaf67dc`), and the analysis takes under 1 s anywhere. 0 GPU-hours.
- **Node-B inputs:** streamed to node A under `/data/dev2/runs/eval/m5/proxy-v2/inputs/` with
  `nodeB-map.tsv` and `nodeB-source.sha256`: the six decoder M3 soup readouts, the X2 readout,
  and the four 27B readouts and reports.

```bash
# node A (frozen development gold; writes aggregates only)
python3 -m v2.eval.proxy_calibration extract --spec $S/v2/eval/records/m5-proxy-v2/spec.json \
  --workers 16 --output /data/dev2/runs/eval/m5/proxy-v2/features.json
# anywhere (from the committed features)
python3 -m v2.eval.proxy_calibration analyze --features v2/eval/records/m5-proxy-v2/features.json \
  --output analysis.json
```

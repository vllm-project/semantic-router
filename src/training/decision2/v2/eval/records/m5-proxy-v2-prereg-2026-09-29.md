# Development proxy v2 — preregistration (eval & peers, 2026-09-29)

Written and pushed before any of the candidate proxies below is computed on the enlarged
model set. Task: recalibrate the development proxy that every size track uses to choose
finalists, with special attention to close within-tier pairs. Previous calibration:
[`m2-proxy-calibration-2026-09-28.md`](m2-proxy-calibration-2026-09-28.md)
(P = 100·√(T_dev·H_pilot), 16 models, tie rule |ΔP| < 4).

## Inputs (unchanged rules from v1)

- Features come only from development readouts: typed DEV (1,600 items) and the CSS pilot
  (1,430 items, three tasks). The calibration target is each model's post-key same-panel
  v3 composite, read from its `REPORT.json`. No v3 item, label or per-item output and nothing
  from JevArena-C1 enters any proxy, fit or selection.
- All features are recomputed on node A (CPU only) from the prediction files with one scorer
  (`benchmark.score.evaluate_answer`, `transfer.score.evaluate`; hash-verified frozen gold),
  so values recorded by the tracks serve only as cross-checks. Only aggregates leave node A.
- Model set ([`m5-proxy-v2/spec.json`](m5-proxy-v2/spec.json)): every distinct weight set that
  has both a formal post-key v3 run and a typed-DEV + CSS-pilot readout of the same weights
  (identity by model hash / revision in the manifests). 51 models:
  - 24 comparators: the v1 matrix (16) and the 8 out-of-sample peers. The three node-B 27B
    peers use their kernel-image `dbe5f32b` re-collections (dev panels and formal from the
    same run), the image every 27B candidate ran on.
  - 27 candidates formally run by the size tracks: 0.6B soups T, V2, X and Z; 0.8B E8F s1–s3
    and soup, B8F s1–s2, E8V soup; 2B S2T soup; 4B X2, X4R s1–s2, X4K s1–s2, and the N4T,
    N4J, N4L and N4LKr soups; 9B L2 and M3 B-s1; 27B C0 (BEST368), M2-C1, M2-S1 and M2-K1.
  - One row per distinct weights. Renderer / limit controls of the same 1.0 weights and
    release re-scores with identical answers are not extra rows.

## Candidate proxies (fixed list; eligible for selection)

T_dev = typed-DEV family-macro accuracy; C, N, S = typed-DEV Choice / Noul / Score accuracy;
H_med = median CSS-pilot task macro-F1 (v1); H_mean3 = mean of the three pilot tasks'
macro-F1 (the 02:40 suggestion).

| Name | Definition | Fitted parameters |
| --- | --- | ---: |
| P (v1) | 100·√(T_dev·H_med) | 2 (linear map) |
| P_mean3 | 100·√(T_dev·H_mean3) | 2 |
| P_type (v1 candidate) | 100·√(mean(C,N,S)·H_med) | 2 |
| P_type_mean3 | 100·√(mean(C,N,S)·H_mean3) | 2 |
| P_CS_mean3 | 100·√(mean(C,S)·H_mean3) (drops typed-DEV Noul) | 2 |
| A_med (v1 `mean_T_H`) | 100·(T_dev + H_med)/2 | 2 |
| A_mean3 | 100·(T_dev + H_mean3)/2 | 2 |
| L2 | v3 ≈ a + b·T_dev + c·H_mean3, fitted leave-one-out | 3 |

References only (never eligible): T_dev, H_med, H_mean3, C, N, S, each pilot task, pilot micro
accuracy.

## Metrics

- Across models: Spearman, Kendall τ-b, leave-one-out linear v3 error (MAE / RMSE / max),
  agreement over all pairs.
- Within tiers: tier-demeaned Pearson r; order agreement on same-tier pairs; on **decision
  pairs** (same tier, at least one candidate); and on **finalist pairs** (same track and tier,
  both candidates). Each is stratified by |Δv3|: < 2 (a tie at formal precision), 2–5 (close),
  ≥ 5 (clear).
- **Primary metric:** order agreement on decision pairs with |Δv3| ≥ 2.
- Uncertainty: 2,000 stratified cluster-bootstrap draws over models within each tier (seed
  20260929), paired between proxies. Per-model panel noise: 1,000 draws of typed-DEV groups
  within family and pilot items within task (seed 20260928), as in v1. L2 orders pairs by its
  leave-one-out predictions.

## Selection rule

Replace P by candidate X only if all of the following hold:

1. X beats P on the primary metric, with a paired-bootstrap probability of at least 0.90.
2. X's LOO RMSE is at most P's + 0.25.
3. X's all-pairs agreement is at most 1 point below P's.
4. X's median panel noise (in v3 units) is no larger than P's.

If several candidates qualify, take the fewest fitted parameters, then the highest primary
metric. Otherwise keep P (earlier readouts stay comparable) and update only the tie band.

## Tie band rule

For the recommended proxy R, B is the smallest integer gap on {1, …, 15} (R units) such that,
among decision pairs with |ΔR| ≥ B, sign agreement with v3 is ≥ 90% and reversals by ≥ 2 v3
points are ≤ 5%. The condition must hold at B and at every larger grid value that still has
≥ 20 pairs, and B itself must leave ≥ 20 pairs. Model-based check: β_w = least-squares slope of
Δv3 on ΔR through the origin over decision pairs, σ = RMS residual, B_model = 1.2816·σ/β_w
(10% reversal risk under a normal residual model). If no grid value qualifies, the conclusion is
**"no proxy separates close within-tier candidates reliably"** and the rule stays "ties go to the
formal runner".

## Sensitivity analyses (reported, never used for selection)

- **S1:** drop the four 0.6B soups, which motivated the three-task mean.
- **S2:** replace the adopted 1.0 rows by same-renderer / same-limit controls (Kai1 8K; Eos1,
  Sol1, Nox1 and Lux1 shared-renderer 8K).
- **S3:** drop the 12 candidates whose readout ran at a different limit or on the kernel-less
  image than their formal run.
- **S4:** the v1 16-model spec exactly (must reproduce the v1 numbers).

## Known before this preregistration (disclosed)

- The v1 results and the per-model v3 composites.
- The P / T_dev / H_pilot values and per-task pilot F1 the tracks recorded for their
  candidates.
- The mis-rankings in the cross-track notes: Kai1 vs Lex; 4B X2; the 0.6B V2 vs A7 soups. The
  three-task mean was proposed after seeing the four formal 0.6B soups; hence S1.

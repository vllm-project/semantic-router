# 0.8B candidate meets the first-release threshold (seed 1): M2 E8F-s1

> **Erratum (2026-09-29, per coordinator 05:25 note).** The staging revision cited below (`5421582d`) is no longer in the
> `dev2-dec-staging` history, because LFS cleanups rewrote it. The E8F-s1 weights were deleted from staging at 20:51Z.
> Identity = the `m2/E8F-s1/` lines of hash list node B `m2/hf-staging/batch1.sha256` (`74c04b5d…`), with weights
> `8b332483…` and head `bab6b40c…`. See [dec-staging-citation-errata-2026-09-29.md](dec-staging-citation-errata-2026-09-29.md).

Status: **seed 1 meets the coordinator's 0.8B threshold on post-key v3; the
preregistered seed-2 confirmation (E8F-s2, training) is pending.** Written
2026-09-28 ≈19:00 UTC+8 immediately after scoring. Lock:
[batch-1 formal lock](dec-m2-batch1-formal-lock-2026-09-28.md) (`6dbc18141`).

## Result (post-key same-panel, node A, frozen runner, 8,192 tokens)

| | E8F-s1 | Eos 1.0 adopted run | Eos 1.0 same-limit control |
| --- | ---: | ---: | ---: |
| **JevArena v3** | **49.777** | 42.547 | 42.361 |
| Paired Δ (95% CI) | — | **+7.23 [+1.83, +13.68]** | +7.42 [+1.89, +13.65] |
| T (typed FINAL family macro) | .5713 | .3925 | .3931 |
| H (CSS15 task-median macro-F1) | .4337 | .4612 | .4565 |
| Choice / Noul / Score (of 800/800/400) | 510 / 613 / 103 | 315 / 410 / 120 | 318 / 406 / 121 |
| Public JevBench 231 (easy/standard/hard) | 150 (48/61/41) | 142 (48/55/39) | 142 (48/55/39) |
| Typed Brier / ECE | .267 / .122 | — | .343 / .181 |
| Invalid typed / CSS15 / public | 0 / 18 / 0 | 0 / 4 / 0 | 0 / 18 / 0 |

Regressions to disclose: typed-FINAL Score −17 of 400 (resource ledger), CSS15
H −.028 (18 over-budget `tropes` inputs at the 8,192-token package limit vs 4
in the native 1.0 run). No decision type collapses. The official-Base start
B8F-s1 (development proxy 42.10 vs E8F 37.72) reaches only **43.084** (+0.54
[−3.53, +5.04]): the development ranking reversed on v3.

## Candidate identity and location

- Weights: private HF staging repo `llm-semantic-router/dev2-dec-staging`,
  commit `5421582dfed51dd3b19b5686128ed75e0f8525e3` [corrected 2026-09-29: revision no longer exists in the repo
  history; identity by hash list `batch1.sha256` `74c04b5d…`, see errata record], folder
  `m2/E8F-s1/checkpoint-0001776/` (full Decision 2.0 checkpoint: Qwen3.5 text
  backbone 752,393,024 + head 1,053,184 parameters, FP32) and
  `m2/E8F-s1/cal/calibration.json` (per-type CAL700 temperatures Choice
  1.00520, Noul 1.01111, Score 1.05724). `model_sha256`
  `5359b701c4a351f0f5130b34aa757494c4648f571092eaccf0912900a1a2281a`.
- Weight origin: own `llm-semantic-router/Decision-1.0-Eos-0.8B@363c4a5e`
  (itself from `Qwen/Qwen3.5-0.8B@2fc06364`, Apache-2.0), full fine-tuning on
  the M2 full mixture (`d1dc33fc…`, 162,777 rows / 138.4M tokens: A0s-r + A1,
  A2, A3, A4v2h, A5, A6g, A6h + all A7 minus the two A7h shortcut families),
  code `163d40dab`, image `dbe5f32b…`, node B GPU3, 1 epoch, 2,030 updates,
  SELECT-selected step 1,776 (592/700).
- Packaging profile: **`qwen-full`**.
- Scored run directory (node A): `/data/dev2/runs/dec/formal/m2/m2-E8F-s1-nodeA`
  (REPORT.json, SEAL.json, PAIRED files; persisted autotune cache in
  `m2-E8F-s1-nodeA-triton`).

## Remaining conditions (preregistered)

1. E8F-s2 (seed 20260927, identical configuration) formal v3 point estimate
   above 42.547.
2. Release gates (3)–(6) by the release pipeline (real HF download and
   package hash, System One native examples, cross-process repeatability,
   card); `mlx-diag` multilingual diagnostic reported with the candidate.

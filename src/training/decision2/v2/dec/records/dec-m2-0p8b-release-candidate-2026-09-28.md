# 0.8B release candidate meets the first-release threshold: E8F seed soup

Status: **qualifies under the coordinator's 0.8B threshold and the M2
release reading (amendment 3)** — reported for release engineering.
Written 2026-09-28 ≈20:30 UTC+8 right after scoring. Lock:
[batch-2 formal lock](dec-m2-batch2-formal-lock-2026-09-28.md) (`0a4de2b17`).
Supersedes the seed-1 note [`f02b73966`](dec-m2-0p8b-candidate-2026-09-28.md).

## Result (post-key same-panel, node A, frozen runner, 16,384-token package)

| | E8F soup | Eos 1.0 adopted run | Eos 1.0 same-limit (16K) |
| --- | ---: | ---: | ---: |
| **JevArena v3** | **50.236** | 42.547 | 42.361 |
| Paired Δ (95% CI) | — | **+7.69 [+3.65, +13.32]** | +7.88 [+3.69, +13.36] |
| T (typed FINAL) | .5734 | .3925 | .3931 |
| H (CSS15 median macro-F1) | .4401 | .4612 | .4565 |
| Choice / Noul / Score | 529 / 611 / 107 | 315 / 410 / 120 | 318 / 406 / 121 |
| Public JevBench 231 (easy/standard/hard) | **156** (48/63/45) | 142 (48/55/39) | 142 (48/55/39) |
| Typed Brier / ECE | .277 / .150 | — | .343 / .181 |
| Invalid typed / CSS15 / public | 0 / 4 / 0 | 0 / 4 / 0 | 0 / 4 / 0 |

Release reading (amendment 3): paired lower bound +3.65 > 0 ✓; E8F seed mean on
the development proxy 37.04 > Eos 1.0 30.58 ✓; no decision type collapses ✓.
**Disclose:** typed-FINAL Score −13/400 (resource ledger; Score is weak for
both), CSS15 H −.021 (the gain is typed reasoning: Choice +214, Noul +201).
Best measured 0.8B peers: Intern-Decision-0.8B 43.535, Kev-0.8B 43.217.

## Artifact for release engineering

- **Weights:** private `llm-semantic-router/dev2-dec-staging` commit
  `16c0929ac0df649d8223483e1adf419d78a647ed`, folder `m2/E8F-soup/checkpoint/`
  (full Decision 2.0 checkpoint, FP32: Qwen3.5 text backbone 752,393,024 +
  decision head 1,053,184 = 753,446,208 loaded parameters). `model_sha256`
  `60356482ceeb669c4a97eb14dcfae5144b1b181f6c8b7a628ea5d02c86a6dd8b`.
- **Calibration:** `m2/E8F-soup/cal698-16k/calibration.json` (`9f76867d…`;
  CAL698 `19cc1a8c…`, `selection_policy = frozen_checkpoint`, max length
  16,384; temperatures Choice 1.12262, Noul 1.07053, Score 0.39532).
- **Packaging profile:** `qwen-full`, `max_input_tokens` 16,384, adapter
  `v2/dec/adapter-spec-infer-dec.json` (`v2.dec.infer_dec`; the source path is
  recorded only — the full checkpoint carries every weight).
- **Scored run directory (node A):** `/data/dev2/runs/dec/formal/m2/m2-E8F-soup-nodeA`
  (REPORT, SEAL, PAIRED vs adopted and same-limit; persisted autotune cache
  `m2-E8F-soup-nodeA-triton`; image `f83b1d10…`, node A GPU5).
- **Weight origin:** uniform FP32 average of three seeds (20260926/27/28) of
  the same recipe from own `llm-semantic-router/Decision-1.0-Eos-0.8B@363c4a5e`
  (itself from `Qwen/Qwen3.5-0.8B@2fc06364`, Apache-2.0): full fine-tuning,
  backbone lr 1e-5, head lr 1e-4, one epoch (≈2,030 updates of ≥ 64 rows) on
  the M2 full mixture `d1dc33fc…` (162,777 rows / 138.4M tokens: A0s with
  rule-7d key renumbering, data-arms v1 A1–A6, all A7 own-1.0 corpora; the two
  A7h shortcut families excluded), SELECT700 checkpoint selection per seed
  (steps 1,776 / 1,520 / 2,041), code `163d40dab`, image `dbe5f32b…`, node B.
  All training sources: Apache-2.0 own weights; data from the private HF
  dataset revisions `39a120ca…` (A7) and `5c0255ed…` (v1), licences per their
  registries. The A0s component equals the data track's published
  positional-key `m3/pk1/A0s` (`d8eae3e4…`) minus the two excluded families,
  row for row; the eval track's C1 independence check (20:40) covered the
  registered training data.

## Seed evidence and multilingual diagnostic (added after scoring)

Per-seed post-key runs (8,192 tokens, CAL700): s1 49.777 (+7.23 [+1.83,
+13.68]), s2 47.360 (+4.81 [+0.72, +9.59]), s3 41.084 (−1.46 [−4.28, +4.99];
CSS15 H .3635). Seed mean 46.07; the soup (chosen on development data) is
above every seed. `mlx-diag` (development diagnostic, 16,384 tokens, CAL698):
type-macro 65.2 (Eos 1.0 66.5), English 67.1 (67.9), non-English Choice /
Noul / Score 68.7 / 54.2 / 71.9 (68.5 / 59.2 / 71.0), weakest ko 55.0 (56.0);
run dir `/data/dev2/runs/dec/formal/m2/m2-E8F-soup-nodeA-mlx`.

## Remaining release gates (release pipeline)

Real HF download + package hash + loaded-parameter verification; System One
native Choice/Noul/Score examples; cross-process repeatability (borrow one
decoder GPU on node A with image `f83b1d10…`); card (owl banner, same-panel
table, rank and model × task charts; disclose the Score, H and multilingual
Noul regressions and the seed dependence).
Note: release packaging must accept `frozen_checkpoint` calibration reports
(shared loader change `0a399c1d9`).

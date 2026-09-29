# HT-DEV v1 validation — does it track formal human transfer better than the CSS pilot? (eval & peers, 2026-09-29)

Preregistered in [`htdev-prereg-2026-09-29.md`](htdev-prereg-2026-09-29.md) §5–§6 (amendments
[1](htdev-prereg-amendment-1-2026-09-29.md) and [2](htdev-prereg-amendment-2-2026-09-29.md); panel build
[`htdev-build-2026-09-29.md`](htdev-build-2026-09-29.md), registration `679d3636c`). Tool:
`python3 -m v2.eval.htdev.validate extract|analyze` at `0b2da7c4b` (`701eb6fdd` plus the strict-JSON fix; node-A mirror
tree `ffca6639`). Aggregates only: [`htdev-validation/features.json`](htdev-validation/features.json),
[`htdev-validation/analysis.json`](htdev-validation/analysis.json), model list
[`htdev-validation/spec.json`](htdev-validation/spec.json), GPU ledger
[`htdev-validation/gpu-time.json`](htdev-validation/gpu-time.json). Formal H and v3 come from the stored post-key
`REPORT.json` files. No v3 item was re-read and nothing from JevArena-C1 was touched. HT-DEV numbers are development
readouts, never release scores.

## Result

- **HT-DEV v1 does not track formal human transfer better than the CSS pilot.** Both preregistered conditions fail, so
  nothing is recommended. **Proxy v2 stays** (P = 100·√(T_dev·H_pilot), tie band |ΔP| < 8), and no track switches.
  - (i) Primary agreement: within-tier sign agreement of ΔH_dev with ΔH_formal on the decidable pairs
    (|ΔH_formal| ≥ 0.02) is **48/89 = 0.539, exactly the pilot's 48/89**. The paired model bootstrap gives
    **P(HT-DEV better) = 0.51**, with Δ +0.003 [−0.233, +0.235]. The bar was ≥ 0.90.
  - (ii) Within-tier Pearson r with formal H is **0.396 vs 0.402** for the pilot. The bar was a higher point estimate;
    P(better) is 0.47.
- **The pilot's three-task mean does better than both, but not significantly.** It gets 54/89 = 0.607 with
  within-tier r 0.523. HT-DEV's P(better than the three-task mean) is 0.27 on agreement and 0.20 on r. It is
  reported because the prereg asks for it; it is not a preregistered replacement. Proxy v2 already found that
  P_mean3 does not order v3 better than P.
- **§6, computed for the record only.** On within-tier v3 decision pairs (|Δv3| ≥ 2):
  - P_HT = 100·√(T_dev·H_dev) orders **58/90 (0.644)** correctly and P_HT,mean orders 62/90.
  - Proxy v2's P orders **67/90 (0.744)**.
  - LOO RMSE of the linear map is 3.42 (P_HT) vs 3.23 (P).
- **H_dev cannot resolve within-tier human-transfer gaps.** Its normal-residual 10%-risk gap against formal H,
  with tier fixed effects, is **0.12**. That is wider than the whole within-tier range of formal H in the 0.8B
  (0.087), 4B (0.087) and 9B (0.072) tiers.

## Model set and integrity

37 models, the full preregistered list (27B excluded by amendment 2 item 10). Every model has one same-job collection
(`--panels ht-dev,css-pilot,typed-dev`) on node A and a stored post-key formal `REPORT.json`.

| Tier | Models |
| --- | --- |
| 0.6B (8) | Kai1, Lex1, Bosun 0.6B, GLiNER2.5-Decide; 2.0: T soup (released DEV2.0-0.6B), V2, X, Z soups |
| 0.8B (8) | Eos1, Intern, Kev, JPT-0.8B; 2.0: E8F soup (DEV2.0-0.8B lineage), E8F s1, B8F s1, E8V soup |
| 2B (5) | Sol1, Decider 2B, This-That 1.2, Bosun 1.7B; 2.0: S2T soup (DEV2.0-2B) |
| 4B (10) | Nox1, Decider 4B, Jet v6.2, Hopper-G, JPT-4B; 2.0: N4LKr, N4L, N4T, N4J soups, X2 |
| 9B (6) | Lux1 (node-A 16K comparator run), JPT-9B, Nimble v2; 2.0: L2 (8K), M3 B-s1, M3 DW |

- **Seals and reports.** All 37 `SEAL-HTDEV.json` files were verified: predictions sha256 matches, 3,240 items, 0
  missing, gold not read at seal time. All 37 `REPORT-HTDEV.json` files parse. The last 13 reports were written by two
  leftover copies of the build worker's `score_all.sh` (scoring code `425930a0b`). They ran concurrently a few seconds
  apart. Scoring is deterministic, so the duplicate writes are identical; the files were verified afterwards.
- **Readout parity with the formal run.** 29 collections match their formal `COLLECT.json` exactly in adapter spec,
  image, model revision and weights path.
  - `dec-e1-X2` and `9b-l2-8k` differ only in the `adapters.py` file hash; their adapter specs are identical.
  - The 7 adopted comparator runs (Kai1, Bosun 0.6B, Sol1, Decider 2B, Nox1, Decider 4B, JPT-9B) keep only
    `ADOPT.json`, which records the adopted prior-session predictions and receipts but no collection manifest. Their
    parity with the HT-DEV collections cannot be checked from files.

## §5 metrics (target: formal H = CSS15 median task macro-F1)

126 within-tier pairs. Decidable pairs at |ΔH_formal| ≥ 0.02: 89.

| Metric | H_dev | H_dev,mean | H_pilot | Pilot three-task mean |
| --- | ---: | ---: | ---: | ---: |
| **Agreement, \|ΔH_formal\| ≥ 0.02 (primary)** | **0.539** | 0.494 | **0.539** | 0.607 |
| Agreement, all pairs | 0.532 | 0.500 | 0.540 | 0.563 |
| Agreement, ≥ 0.01 | 0.523 | 0.486 | 0.550 | 0.586 |
| Agreement, ≥ 0.03 | 0.558 | 0.532 | 0.545 | 0.610 |
| **Within-tier Pearson r** | **0.396** | 0.333 | **0.402** | 0.523 |
| Cross-tier Spearman | 0.773 | 0.765 | 0.810 | 0.824 |
| Cross-tier Pearson | 0.793 | 0.785 | 0.805 | 0.835 |
| Within-tier r vs CSS15 task mean (lower-noise target) | 0.458 | 0.489 | 0.559 | 0.602 |
| Agreement ≥ 0.02 vs CSS15 task mean (70 pairs) | 0.600 | 0.614 | 0.600 | 0.614 |
| Median item-bootstrap SD of the score | 0.023 | — | 0.020 | 0.012 |

Paired model bootstrap: 5,000 draws, seed 20260929, models resampled within tiers. H_dev vs H_pilot:
P(better) = 0.51 on agreement and 0.47 on r. H_dev vs the three-task mean: 0.27 and 0.20.

Agreement also barely rises with HT-DEV's own gap: 0.42 for |ΔH_dev| < 0.01, 0.57 for 0.01–0.02, 0.53 for 0.02–0.04
and 0.57 above 0.04.

## Per tier (descriptive; the decision rule pools tiers)

Correct orderings on decidable pairs (|ΔH_formal| ≥ 0.02):

| Tier | Decidable pairs | H_dev | H_pilot | Three-task mean | Formal H range |
| --- | ---: | ---: | ---: | ---: | --- |
| 0.6B | 25 | 18 | 21 | 24 | 0.266–0.498 |
| 0.8B | 20 | 5 | 7 | 10 | 0.374–0.461 |
| 2B | 9 | 7 | 5 | 6 | 0.380–0.525 |
| 4B | 28 | **16** | 9 | 11 | 0.476–0.563 |
| 9B | 7 | 2 | 6 | 3 | 0.497–0.569 |
| All | 89 | 48 | 48 | 54 | |

HT-DEV helps at 4B and 2B and hurts at 9B, 0.8B and 0.6B. That does not serve the 9B and 4B motivation: at 9B it
orders 2 of 7 decidable pairs correctly.

## Why HT-DEV does not track formal H (interpretation)

- **The target is noisy at within-tier resolution.** Release-gate paired CIs of formal human transfer span about
  0.13–0.18. Examples: DEV2.0-4B vs Decider 4B +0.024 [−0.095, +0.073]; DEV2.0-0.6B vs GLiNER2.5-Decide +0.039
  [−0.013, +0.167]; 0.6B M6 mxcx vs released −0.021 [−0.064, +0.061]. So most pairs with |ΔH_formal| of 0.02–0.05 are
  inside the target's own noise, and no development panel can agree much above chance on them.
- **HT-DEV is dataset-isolated by design, and formal H is dataset-specific.** The pilot's three tasks are CSS15 tasks
  and share sources and label schemes with the target. HT-DEV's 13 tasks come from other datasets. It still
  tracks formal H across tiers (Spearman 0.77), but its within-tier differences reflect different skills.
- **Lineage.** HT-DEV over-rates the JPT peers:
  - JPT-9B has the highest H_dev in its tier (0.667) and the lowest formal H (0.497).
  - JPT-0.8B and JPT-4B are also high on H_dev and last or near last on formal H.
  - Excluding the three JPT models (post hoc, not decision-bearing) does not rescue HT-DEV: 41/72 vs the pilot's
    44/72 (three-task mean 51/72), within-tier r 0.412 vs 0.485 (0.620).
- **The median-over-tasks functional is as noisy as the pilot's.** H_dev has median item-bootstrap SD 0.023 despite
  3,240 items, against 0.020 for the pilot. The lower-noise task mean (H_dev,mean) tracks formal H worse, not better
  (0.494, r 0.333). So panel noise is not the main issue; construct mismatch is.

## §6 proxies (record only; §5 did not pass)

Target: post-key v3. Decision pairs are within-tier pairs with |Δv3| ≥ 2 (90 pairs).

| Proxy | Decision pairs | All within-tier pairs | Cross-tier Spearman | Linear map | LOO MAE / RMSE / max | Tie band |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| **P (proxy v2)** | **67/90 = 0.744** | 0.690 | 0.917 | v3 ≈ 21.08 + 0.606·P | 2.68 / 3.23 / 7.0 | 10 |
| P_HT = 100·√(T_dev·H_dev) | 58/90 = 0.644 | 0.611 | 0.922 | v3 ≈ 9.26 + 0.727·P_HT | 2.87 / 3.42 / 7.1 | 9 |
| P_HT,mean | 62/90 = 0.689 | 0.659 | 0.924 | v3 ≈ 10.24 + 0.720·P_HT,mean | 3.15 / 3.60 / 7.1 | 10 |

On these 37 models P reproduces proxy v2's picture:

- Order accuracy is 0.86 at |ΔP| ≥ 8 (3.6% reversed by ≥ 2 v3 points), 0.70 at 4–8 and 0.58 below 4.
- The normal-residual gap is 9.7.
- Proxy v2's official map (v3 ≈ 22.4 + 0.58·P on 51 models) and its tie band of 8 stand unchanged.

## What this means for tracks

- **Keep proxy v2** for shortlisting against v3: drop only candidates ≥ 8 P points behind the best, and send the rest
  to the formal runner, at most three finalists per tier.
- **Human transfer is decided only by the formal paired CI** on CSS15 (release gates). No development panel,
  whether HT-DEV, the pilot median or the pilot three-task mean, orders within-tier candidates on formal H reliably.
  Never select on H_dev, H_pilot or a single task.
- **HT-DEV stays registered** as a development panel for diagnostics only. Examples are gross human-transfer
  regressions and 5-level Score level usage on its `empathy/empathic_reactions` task. It is not a proxy, screen or
  selection criterion.
- A fitted multi-task HT-DEV model is not attempted. It would have to be fitted against formal H, which the prereg
  forbids (formal H is only the validation target), and 37 models could not support it anyway.

## Compute

- **This validation:** CPU only on node A (extract ≈ 1 min, analyze ≈ 1 min) plus the CPU scoring of the last 13
  models. **0 GPU-h.**
- **HT-DEV collections:** 40 runs from 23:19Z to 00:21Z on node A GPU0 (0.907), GPU1 (0.855) and GPU5 (0.866).
  Collect 2.577 + smoke 0.051 = **2.628 GPU-h**, all shared-lease `owner.eval-htdev`.
- **HT-DEV programme total:** build 0.64 + collections 2.63 = **3.27 GPU-h**, against the 8 GPU-h cap.

## Artifacts

- Private Hub dataset `llm-semantic-router/decision-2.0-eval-artifacts`, commit
  `ad4b958ee94476d0204bbe61f6d59bb6a2e30670`, holds two new folders. The download was verified by sha256.
  - `htdev/v1/collect/<model>/`: `COLLECT.json`, `GPU-TIME.json`, `SEAL-HTDEV.json`, `REPORT-HTDEV.json` and the
    three prediction files; no prompts, gold or logs.
  - `htdev/v1/validation/`: spec, features, analysis and the GPU ledger.
- Node A: `/data/dev2/runs/eval/htdev/{collect,validation}`.
- Hashes (sha256): `spec.json` `b00ffe8a…`, `features.json` `784adc4f…`, `analysis.json` `702d4aa0…`.

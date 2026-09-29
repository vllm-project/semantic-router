# Decoder Milestone 4 — finalist selection (development panels only; 2026-09-28 22:55 UTC)

Rules: [prereg](dec-m4-prereg-2026-09-29.md) rules 1, 2 and 5, with
[amendment 1](dec-m4-amendment-1-2026-09-29.md): R = P_mean3 = 100·√(T_dev·H_mean3), tie band 9. Tool:
`ops/m4/m4-select.py --cross N4LX N4LR N4LR2 N4XA N4XF N4LRQ` on node B (output `m4/select-2.json`,
`986c466c…`). All numbers below are **development readouts**, typed DEV (1,600) + CSS pilot (3 tasks); no v3,
public-231 or `mlx-diag` result existed when this was written.

| Artifact | R (P_mean3) | seed R (s1 / s2 / s3) | T_dev | H_mean3 | pilot median | P (v1) | typed C / N / S | Eligible |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| **N4LX soup** (N4LR + N4XF, 6 seeds) | **63.40** | 6-seed mean 61.44 | .7300 | .5506 | .5382 | 62.68 | 516 / 273 / 379 | yes |
| N4LR soup (Lux on all rows, KL 1.0) | 63.07 | 63.44 / 58.81 / 60.39 | .7288 | .5458 | .5098 | 60.95 | 518 / 288 / 360 | yes |
| **N4XF soup** (XL r2 full subsample) | 62.94 | 62.57 / 60.16 / 63.30 | .7044 | .5625 | .5364 | 61.47 | 501 / 264 / 362 | yes |
| N4LR2 soup (KL 2.0) | 62.65 | 63.10 / 61.55 / 58.49 | .7250 | .5413 | .5017 | 60.31 | 584 / 256 / 320 | yes |
| N4LRQ soup (A7q/k/s swap) | 62.63 | 60.44 / 59.36 / 64.11 | .7094 | .5530 | .5376 | 61.76 | 530 / 238 / 367 | yes |
| N4XA seed s3 (XL r2 A7-only; soup 56.27 < seed mean 56.91) | 58.19 | 58.58 / 53.95 / 58.19 | .6038 | .5609 | .5218 | 56.13 | 536 / 226 / **204** | **no** (Score < 284; Noul ≥ 95% one side) |
| *M3 N4LKr soup (reference)* | 59.06 | | .6456 | .5402 | .5305 | 58.52 | 505 / 225 / 303 | |
| *Nox 1.0* | 55.33 | | .6663 | .4595 | .4131 | 52.46 | 460 / 228 / 378 | |

Every soup is at or above its seed mean, so the soup is the artifact (rule 1), except N4XA.

## Finalists

1. **N4LX soup** is the highest R.
2. **N4XF soup** takes slot 2. N4LR (63.07) and N4XF (62.94) differ by 0.13 < 1 R unit, so slot 2 goes to the
   higher min(C/460, N/228, S/378): N4XF .958 vs N4LR .952.

All five eligible artifacts lie within 0.8 R of each other, far inside the tie band of 9. The formal runner
decides.

## Development contrasts (paired bootstrap, 10,000 draws; P_mean3 unless stated)

- **N4LR − N4LKr (own-Lux instead of own-Nox on the retention rows):** +4.01 [+2.82, +5.20].
  - T_dev +.083 [+.065, +.102]; H_mean3 level.
  - The largest single lever of M4: Lux soft targets on Nox's own A7 curriculum rows carry new typed-reasoning
    information.
- **N4LR2 − N4LR (KL 2.0 vs 1.0):** −0.42 (a tie). Choice rises (584 vs 518) while Noul and Score fall.
- **N4LRQ − N4LR (A7q/k/s at 20% of tokens in place of v2-M rows):** −0.44 (a tie).
  - Pilot transfer rises (H_mean3 .553 vs .546; pilot median .538 vs .510).
  - Typed Score 367 vs 360; typed Noul falls (238 vs 288).
- **N4XA vs N4LR (XL A7-only recipe vs the M3 mixture):** typed DEV collapses (T .604 vs .729; Score levels
  mostly 0 / 2) while H_mean3 is the best of any artifact (.561). Not eligible.
- **N4XF vs N4XA (adding the v2 pools and the gold-only H7 / H8 gap arms inside XL):** repairs typed DEV (T .704,
  Score 362) and keeps the highest eligible H_mean3 (.5625).
- **N4LX (N4LR + N4XF seeds):** the typed profile is at or above Nox 1.0's development counts on every type
  (min ratio 1.003). Its R is above both parents.

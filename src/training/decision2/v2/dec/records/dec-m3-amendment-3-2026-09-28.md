# Decoder Milestone 3 — amendment 3 (4B: stronger own-Lux trust region, N4LK)

Parents: [prereg](dec-m3-prereg-2026-09-28.md) (`5c8bbc569`),
[amendment 1](dec-m3-amendment-1-2026-09-28.md) (`271502d8f`),
[amendment 2](dec-m3-amendment-2-2026-09-28.md) (`74ce7d0d0`). Written 2026-09-28
≈01:00 UTC+8 (09-29), after the three 4B formal results and before any N4LK job.

## Evidence (post-key same-panel, node A, 16K, vs adopted Nox1 56.470)

| 4B soup | v3 | Δ [95% CI] | T | H (CSS15) | Dev P (node B) | Dev H (CSS pilot) |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Nox 1.0 | 56.470 | — | .6144 | .5190 | 52.46 | .413 |
| N4T own Nox | 56.506 | +0.04 [−4.13, +4.27] | .6450 | .4950 | 58.49 | .498 |
| N4J AutoJev | 54.212 | −2.26 [−5.97, +2.65] | .6072 | .4840 | 60.09 | .539 |
| **N4L own Lux** | **58.903** | **+2.43 [−4.03, +5.44]** | .6462 | **.5369** | 58.27 | .533 |

Only the Lux teacher raised CSS15 human transfer at 4B (+.018 vs Nox 1.0, +.042 vs
N4T). All three soups raised the three-task CSS pilot by a similar amount
(+.08 to +.13), so the pilot does not separate teachers at 4B. Lux 1.0 is also
stronger than Nox 1.0 on `mlx-diag` (type macro .828 vs .795), where every 4B
v2-M soup loses about 3 points.

## Arm N4LK (one factor vs N4L)

Identical to N4L — Nox 1.0 `@cde2a68d` start, mixture `13804ac6…`, the
own-Lux composite teacher `f1df0549…`, recipe M3F, seeds 20260926 / 27 / 28 — except
**KL weight 1.0** instead of 0.5. Cap 1.6 GPU-h per seed. Node B GPU1, GPU2 (idle
since N4L) and GPU4 (after E8V-s2).

## Rules

The prereg's artifact, finalist, formal and release rules apply unchanged. In
particular, the release reading still needs a v3 paired lower bound > 0 against
the adopted Nox1 run. Among qualifying 4B artifacts the amendment-2 rule applies
(the clean-teacher artifact with the highest v3 point estimate). N4L itself stays HOLD
whatever N4LK shows; no artifact is re-selected after its formal result.

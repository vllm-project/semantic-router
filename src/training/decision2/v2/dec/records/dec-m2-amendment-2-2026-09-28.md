# Decoder Milestone 2 — amendment 2 (formal order and checkpoint transfer)

Parent: [prereg](dec-m2-prereg-2026-09-28.md) `143cfc214`, [amendment 1](dec-m2-amendment-1-2026-09-28.md)
`f316bad37`. Written 2026-09-28 ≈18:35 UTC+8, before any 0.8B or X4K formal
prediction exists. No arm, rule or reading changes; only order and transport.

## Development outcomes that trigger preregistered steps

Node B, M2 image, same-image 1.0 controls (development only):

| Arm | T | H | P | Choice / Noul / Score | Rule outcome |
| --- | ---: | ---: | ---: | --- | --- |
| Eos 1.0 | .4956 | .1887 | 30.58 | 510 / 198 / 85 (all level 0) | — |
| E8F-s1 | .5563 | .2558 | 37.72 | 486 / 236 / 168 (levels 0, 2) | clears the 0.8B rule: +7.14 [+4.07, +10.71] |
| B8F-s1 | .6838 | .2592 | 42.10 | 723 / 221 / 150 (levels 0, 2) | clears the 0.8B rule: +11.52 [+8.74, +15.01]; B8F − E8F +4.38 [+1.46, +7.78] |
| Nox 1.0 | .6663 | .4131 | 52.46 | 460 / 228 / 378 | — |
| X4K-s1 | .6669 | .4353 | 53.88 | 468 / 225 / 374 | passes the 4B screen |
| X4K-s2 | .6588 | .4240 | 52.85 | 460 / 223 / 371 | passes the 4B screen |
| Sol 1.0 | .5869 | .3191 | 43.27 | 410 / 218 / 311 | — |
| S4R-s1 / s2 | .5875 / .5756 | .3260 / .3466 | 43.76 / 44.67 | Score 296 / 285 | fail the Score floor → **no 2B formal slot (2B HOLD)** |
| S4C-s1 / s2 | .5594 / .5931 | .3370 / .3351 | 43.42 / 44.58 | Choice 376 / 425 | s1 fails the Choice floor → no formal slot |

Per the prereg, E8F-s2 and B8F-s2 (seed 20260927, identical configuration)
were launched at 18:30 on node B GPU0/GPU1, and both X4K seeds take formal
runs (amendment 1).

## Order

The formal runs of E8F-s1, B8F-s1, X4K-s1 and X4K-s2 start now on node A GPU5;
E8F-s2 / B8F-s2 formal runs follow when they complete. Every seed that the
prereg sends to formal is scored whatever the earlier seed shows, and the
release reading (s1 qualifies only if its paired lower bound vs the 1.0 is
> 0 **and** s2's point estimate is above the 1.0) is unchanged, so the order
carries no selection.

## Transport

Node B's uplink through the workstation runs at ≈0.5 MB/s (a 154 MB adapter
took ≈5 minutes; a 2.9 GB full checkpoint would take ≈100 minutes). Finalist
checkpoints (with their CAL calibration) therefore go from node B to a
private HF staging model repo `llm-semantic-router/dev2-dec-staging` and are
downloaded on node A, where every file hash is compared with node B's before
collection. The staging repo is private and outside the "Decision 2.0"
collection; it is also where a qualifying candidate would be reported from.

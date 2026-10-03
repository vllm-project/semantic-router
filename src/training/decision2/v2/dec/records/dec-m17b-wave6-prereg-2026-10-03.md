# Decoder M17b (4B owner, M17 continuation): wave 6 preregistration (2026-10-03 ≈18:05Z)

Written before any wave-6 soup is built and before any wave-6 Index result exists. The only 4B Index results known
at writing are M17's (private values; `4b-SDMLxALL`, released as Nox-4B `d55528d1`, is the highest). The arm
factory's partial run `AF-4b-LHS17IB4ML-bf16` (6 of 8 shards, interrupted at the node F return) is not scored or read.

Split (COORDINATION 2026-10-03 01:47): the arm factory trains arms; the 4B owner builds the soups, runs every 4B
Index, gates and releases. The factory's preregistered 4B candidates (`v2/af/records/af-amendment-2-2026-10-02.md`,
`af-amendment-4-2026-10-03.md`) are taken over unchanged, with the factory's tools (`v2/af/ops/af-soup.sh`,
`af-stage.sh`, `af-measure.sh`, M10's `ixchain.sh`) from this branch's mirror.

## Candidates (uniform FP32 averages; one arm, one vote; a factory seed extends the owner's arm soup)

| Candidate | Members |
| --- | --- |
| `4b-AFxALL2` | amendment 2: `SDMLxALL`'s seven with IB4 → `-x5`, SDMLIB4 → `-x5`, UP → `-x4`, plus `4b-LHS17IB4-lrh`, `4b-LHS17IB4ML`, `4b-LHS23IB4` (10) |
| `4b-AFxALL3` | amendment 4: `SDMLxALL`'s seven at their most seeds (M15 `4b-LHA10SDML`, `4b-LHS17SD`, `4b-LHS17UP-x4`, `4b-LHS17IB4-x5`, `4b-LHS17IB4X-x4`, `4b-SDMLIB4-x5`, `4b-LHS17ML-x4`) plus `4b-LHS17IB4-lrh`, `4b-LHS17IB4ML`, `4b-LHS23IB4` (10). Amendment 4's eleventh member `4b-SDMLIB4-lrh` failed at the node F return (both seeds) and is left out, as the factory's rule says |
| `4b-AFxALL` | amendment 2: `SDMLxALL9`'s nine with IB4-x3 → `-x5`, SDMLIB4-x3 → `-x5`, UP → `-x4`, plus the three new arms (12) |
| `4b-AFxALL4` (contingent) | `4b-AFxALL3` plus `4b-SDMLIB4-lrh` (amendment 4's exact eleven), built only if the factory's re-run of both seeds finishes |

- Seed soups (members only, never measured): `4b-LHS17IB4X-s34` and `4b-LHS17ML-s34` (the factory's batch-2 seeds
  s3 / s4, each LoRA BEST checkpoint merged first with `m10_merge.py` and its SELECT agreement check, node C), then
  `-x4` = `[M17 two-seed soup × 2, factory s3–s4 soup × 2]`; `4b-LHS17UP-x4` likewise with `4b-LHS17UP-s34`.
- Information points (single arms; measured only after the candidates, if GPUs are free within budget):
  `AF-4b-LHS17IB4ML-bf16` (its two missing shards, re-run whole), then `4b-LHS23IB4`, `4b-LHS17IB4-lrh`.

## Measurement (unchanged method)

- Each candidate is measured once, on its BF16 release copy (`v2.release.bf16_copy`, source fingerprint = the soup's
  model SHA-256), restaged onto `DEV2.0-4B-13d42143` (identity, loaded 4,208,383,488, calibration none), with the
  IX1 harness (image host2, kit `87d4650b`, the 86-request parity gate, panel-8, dual scoring) through `ixchain.sh`.
- Reference: the current release's run `DEV2.0-4B-SDMLxALL-bf16` (node F copy `ix1/af/refs`, results hash pinned).
  The full-panel paired bootstrap (2,000 replicates, seed 20261002) is the gate's IF1 evidence; the transfer-only
  bootstrap is a reference.
- Order: `4b-AFxALL2`, `4b-AFxALL3`, `4b-AFxALL`, then `4b-AFxALL4` if built, then the information points.
  Measuring follows build order; no candidate is rebuilt, re-weighted or re-measured after a result is read.

## Release rule (Index-first, unchanged)

- A candidate qualifies if the 95% lower bound of (candidate − `d55528d1`'s run) is above 0, plus: formal typed-FINAL
  no-collapse (R3) on its formal run, one row-level contamination audit of every distinct member TRAIN file (IF3),
  the 86-request package parity gate, and `gate evaluate`; fast-path post-upload checks; then the purge of the
  superseded weights (`rewrite_history=False`, node copy kept and re-hashed).
- Progressive: the first qualifier starts its release at once. If several have qualified by the time its package is
  ready to upload, the largest lower bound is uploaded. A later candidate is gated against the then-current release.
- The card's other-tier points use the current mains (Lux `f3122c7c`; Vega's new revision once public).

## Budget and stops

40 GPU-h for soups, Index runs and releases (COORDINATION 01:47). One Index run ≈ 2.2 GPU-h. A failed parity gate,
shard or scorer gate stops that candidate (recorded, not re-run). No arm is trained here.

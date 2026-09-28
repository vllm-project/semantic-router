# Formal lock: M2 batch 2 — 0.8B release artifact (E8F seed soup) and seed runs

Status: **locked before any post-key v3 or public231 prediction exists for
these checkpoints.** Each is collected once. Results are labeled **post-key
same-panel**. Rules: [amendment 3](dec-m2-amendment-3-2026-09-28.md) (`fc6d1e508`).

## Artifact choice (development only, node B, same image)

| E8F | T | H | P | Choice / Noul / Score |
| --- | ---: | ---: | ---: | --- |
| s1 (`checkpoint-0001776`) | .5563 | .2558 | 37.72 | 486 / 236 / 168 |
| s2 (`checkpoint-0001520`) | .5619 | .2768 | 39.44 | 523 / 193 / 183 |
| s3 (`checkpoint-0002041`) | .4063 | .2839 | 33.96 | 374 / 191 / 85 (all level 0) |
| seed mean | | | 37.04 | |
| **soup** (uniform FP32 average of s1/s2/s3) | .6131 | .2705 | **40.73** | 610 / 212 / 159 |

P_soup 40.73 ≥ seed mean 37.04 → **the release artifact is the soup**
(SELECT 605/700, .8425; vs Eos 1.0 +10.15 [+7.99, +13.78] development).

## Frozen identities

All from the private staging repo `llm-semantic-router/dev2-dec-staging` at
commit `16c0929ac0df649d8223483e1adf419d78a647ed` (37 batch-2 files verified
against node B's SHA-256 list).

| Run | Checkpoint | `model_sha256` | Calibration (file / CAL / temperatures C, N, S) | Limit |
| --- | --- | --- | --- | ---: |
| **E8F-soup (release artifact)** | `m2/E8F-soup/checkpoint` | `60356482ceeb669c4a97eb14dcfae5144b1b181f6c8b7a628ea5d02c86a6dd8b` | `9f76867d…`, CAL698 `19cc1a8c…`, frozen_checkpoint; 1.12262, 1.07053, 0.39532 | 16,384 |
| E8F-s2 (seed evidence) | `m2/E8F-s2/checkpoint-0001520` | `b248e25dd4fcb123127d27a3f937ebccf715e501873ae558bf4b3de96f1c82c2` | `175064b2…`, CAL700; 1.37477, 1.02621, 0.28753 | 8,192 |
| E8F-s3 (seed evidence) | `m2/E8F-s3/checkpoint-0002041` | `7805fd7c220f1abade079795e1423d96bb90d7d3d08aa0f4fa5d6935338b2ed3` | `7a501474…`, CAL700; 1.20529, 1.21404, 0.05 | 8,192 |
| B8F-s2 (information) | `m2/B8F-s2/checkpoint-0002026` | `0ce824e9d2da18c572bb8b0ac697074a620c117624554536f822ef45769d5601` | `12b342c5…`, CAL700; 0.92603, 0.99554, 0.22214 | 8,192 |

## Protocol

Frozen runner on node A GPU5 (image `f83b1d10…`), adapter spec
`v2/dec/adapter-spec-infer-dec.json`, typed-final + css15 + public231, a
persisted per-run autotune cache, 20-item smoke first, from an exact mirror of
the commit that adds this lock; order soup, s2, s3, B8F-s2. Compare with the
adopted Eos 1.0 run (42.547) and the same-limit controls (`eos1-16k` 42.361
for the soup; `eos1-8k` 42.361 for the seeds). Release reading (amendment 3):
the soup qualifies if its v3 paired 95% lower bound vs the adopted Eos 1.0 run
is > 0, the E8F seed mean on the development proxy (37.04) is above Eos 1.0
(30.58), and no decision type collapses.

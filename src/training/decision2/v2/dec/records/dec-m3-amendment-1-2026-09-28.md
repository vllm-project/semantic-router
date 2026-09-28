# Decoder Milestone 3 — amendment 1 (0.8B follow-up E8V; before it launches)

Parent: [M3 preregistration](dec-m3-prereg-2026-09-28.md) (`5c8bbc569`). Written
2026-09-28 ≈22:00 UTC+8 while the preregistered 2B/4B arms train (no M3 readout
exists yet). Adds the coordinator's lowest-priority item 5, which runs only on
GPU time the preregistered arms leave free. Nothing preregistered changes.

## Why this design

Item 5 asks for an "E8F + v2-M" 0.8B follow-up aimed at the disclosed losses of
the first release candidate (typed Score −13, CSS15 H −.021, `mlx-diag`
non-English Noul −5). Data v2 full-M adds exactly those skills: H5 multilingual
native Noul (16 languages), H6 human Score, G6 generated Score, H1 cross-domain
human labels, E11 evidence-removal twins. The arm repeats the E8F recipe from
the same start with v2-M added, so its three-seed soup is a like-for-like
contrast with the E8F soup (it needs no new training code).

## Arm E8V

| Field | Value |
| --- | --- |
| Start | own `llm-semantic-router/Decision-1.0-Eos-0.8B@363c4a5e` (E8F's start) |
| Mixture | `m3-e8f-v2m` (spec `specs/m3-e8f-v2m.json` `4489afdf…`, commit `8ea171315`): **TRAIN `60f1841b65106e19b429e8c393af68fb2fcee6affe4d02e99f42108d30a530b2`, 200,469 rows / 152,959,688 tokens** (Choice / Noul / Score 113,700 / 57,405 / 29,364) = pk1 A0s 6,547 (excluded families dropped) + v1 A1–A6h 26,073 + A7 **v3** all six sub-arms 130,138 (the 19 rows A7 v3 quarantined are gone; 2,817 A0s duplicates dropped) + the mx-v2-full-M pools H5 / H1 / H6 / H3 / E11 / G2 / G6 / G4h 37,711 |
| Recipe | **E8F unchanged:** full fine-tuning, backbone lr 1e-5, head lr 1e-4, no teacher, CE + 0.5 Brier, one epoch, token batches ≤ 32,768 tokens / ≤ 64 rows, ≥ 64 rows per update, max length 8,192, SELECT700 matrix-v1 even8, CAL698 (data dir `data-sel700-cal698`) |
| Seeds | s1 20260926, s2 20260927, s3 20260928 |
| Cap | 1.6 GPU-h per seed (≈ 1.2 h training at 0.8B throughput) |
| GPUs | node B GPU3 after S2J-s1, GPU4 after S2J-s3, GPU0 after N4J-s1 (each waits for its chain's last arm) |

## Rules (fixed now)

- **Artifact:** the prereg's soup rule (uniform soup of the three BEST checkpoints
  if its development P ≥ the seeds' mean P, else the median seed).
- **Finalist:** P(artifact) > P(E8F soup) − 4 on the same node and image (E8F
  soup P 40.73, typed-DEV Choice / Noul / Score 610 / 212 / 159), with every type
  count ≥ 75% of the E8F soup's (458 / 159 / 119).
- **Formal (finalist only):** CAL698 at 16,384 tokens, staging upload, node A
  frozen runner at 16,384 tokens, `mlx-diag`, compared with the E8F soup's scored
  run (`m2-E8F-soup-nodeA`, 50.236) and the Eos 1.0 comparators (adopted 42.547,
  16K 42.361).
- **Reading (post-first-release optimization, no automatic swap):** E8V is an
  improvement candidate if (a) its v3 paired lower bound vs Eos 1.0 is > 0,
  (b) its v3 point estimate is ≥ the E8F soup's, and (c) at least one disclosed
  loss shrinks (typed-FINAL Score count, CSS15 H, `mlx-diag` non-English Noul)
  without a new type collapse. The coordinator decides whether a later release
  replaces the E8F soup; E8V never delays the first release.

# 9B Milestone 6 formal post-key run, finalist K5-a12: lock

Frozen 2026-09-30 ~09:15 UTC+8, before any formal prediction of a Milestone 6 artifact (`formal-m6` does not exist on
node A). Preregistration [`lux9b-m6-prereg-2026-09-30.md`](lux9b-m6-prereg-2026-09-30.md) (`f41402e68`) + amendment 1
(`7380a3cbf`). **Post-key same-panel** comparison (the v3 labels were accessed earlier in the project); public 231 is a
public-subset reproduction. Development readouts are never release scores.

## Why K5-a12 is the only finalist

- **KA and KH stopped at their early rules**, so they have no second seed and no line: KA ΔP −0.36
  (`rules/early-KA-s4.json` `9e3cf68d…`); KH ΔP −0.43 and `rule_precedence` 327 < 346 (`early-KH-s4.json` `488d2b17…`).
- **Rules stage** (`m6/rules.sh e2daf2afc readout-lines`, 01:05Z; readout `m6/readout-lines/readout.json` `ceb10a86…`,
  runtime `3277dec9d`). The COORDINATION notes were re-read first (newest 08:40 UTC+8; no rule change).
  - Seed rule (`seed-K5.json` `ee9e8069…`): P(K5 soup) 69.84 ≥ the five seeds' mean 65.51 (k1 65.75, k2 65.10, k3 61.49,
    K-s4 67.06, K-s5 68.15), so the artifact is the soup.
  - α rule vs R = the fresh re-read of K-a13 (T .9250, H3 .5622, C / N / S 799 / 338 / 343, `rule_precedence` 338;
    floors C 775, N 326, S 331, RP 334) (`alpha-K5.json` `7e6c2a3e…`):

    | K5 point | T | G | H3 | P | C / N / S | RP | Eligible |
    | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
    | ⅓ | .9350 | +.0100 | .5639 | 73.47 | 800 / 339 / 357 | 339 | yes |
    | **½** | **.9550** | **+.0300** | **.5751** | **74.68** | **800 / 372 / 356** | **372** | **yes** |
    | ⅔ | .9494 | +.0244 | .5735 | 73.86 | 800 / 374 / 345 | 374 | yes |
    | 1 (soup) | .8950 | −.0300 | .5691 | 69.84 | 758 / 320 / 354 | 320 | no: Choice, Noul, family, RP floors |

    G\* = .0300 at ½; ⅓ is below 0.75·G\* = .0225, so **α\* = ½**. P 74.68 ≥ the drop floor 65.22.
  - Finalists (`finalists.json` `c500e7f4…`): **K5-a12**. K2 (report only, `alpha-K2.json` `01d080a9…`) would pick ½
    (T .9531, H3 .5690).

## Candidate

| Field | K5-a12 |
| --- | --- |
| Checkpoint (node A `/data/dev2/runs/9b/`) | `m6/K5-a12-build/soup` (16 files, 31.78 GB, FP32) |
| `model_sha256` (soup `console.log` = CAL698 `checkpoint_sha256` = both dev manifests) | `ff5558999565a5d7904de2675ab91696d0c1c8bc2e93422480164ec99e039629` |
| Weights | ½ K5 soup (`d2d44ff3250a…`) + ½ Lux 1.0 (`m3/pf-D-s1-zero/run/checkpoint-0000000`, `dc9b795ca75a…`); members 1 : 1 |
| K5 soup | uniform FP32 soup of M4 K-s1 / K-s2 / K-s3 (checkpoints 1,624 / 1,420 / 1,424; seeds 20260926 / 1 / 2) + M6 K-s4 / K-s5 (1,629 / 1,621; seeds 3 / 4); each the SELECT-chosen BEST |
| Training behind | Lux 1.0 full fine-tuning on x60 (122,651 rows / 60.2M tokens), CE + 0.5·Brier + 1.0·KL(own Lux) on every row; no PN1, no HS1, no AutoJev targets |
| CAL698 T, Choice / Noul / Score | 1.0718 / .7922 / .3605 (`m6/K5-a12-cal/calibration.json` `ba73a26815209b7a…`) |
| `SHA256SUMS` (soup + cal, 24 files) | `m6/K5-a12-build/SHA256SUMS` `8abf50e25ccf393e…` |
| Parameters | 7,940,895,744 (`formal.sh --loaded-parameters`) |

## Runtime and comparators

- Eval runner mirror `3277dec9d` (tree `0f30d20e2df9…`; `run_same_panel.sh` `4849283ce01b…`), adapter
  `v2/dec/adapter-spec-infer-dec.json` `50fb6744947c…`, image `sha256:f83b1d10f14d…`, **16,384 tokens**, over-length
  inputs invalid, no truncation; `infer_dec` refuses a calibration bound to another `model_sha256`.
- Autotune cache `formal-m6/triton-cache`: one copy of the frozen `formal-m3/triton-cache` (tree `af623300d71a…`,
  4,785 files, recomputed at lock time); `formal.sh` refuses another tree.
- M6 wrappers: mirror `e2daf2afc` (tree `ddc33f991aa4…`): `formal.sh` `8aa3af26…`, `chain-step.sh` `f21c561b…`,
  `lib.sh` `5c32663e…`, `chains/m6-formal.sh` `043c4bac…`, `chains/m6-post.sh` `5d0444de…`, `chains/m6-htdev2.sh`
  `27d81585…`.
- Comparators (node A, 16K): the incumbent I = the released T = 1 run `release/dev2-8b-t1-derived` (v3 67.737);
  native Lux1 `eval m1/d1-lux1-autotune-cache` (65.808); Nimble v2 `eval m2/q6-nimble2`; the same-renderer control
  `formal-m3/lux1-16k-shared` (65.231, descriptive); I's mlx answers `formal-m4/K-a13-16k-mlx`.

## Reading

- **Successor items 1–7 vs I** (prereg): (1) `PAIRED-vs-DEV2.0-9B-T1.json` `ci95.low` > 0; (2) its
  `axis_ci95.H.delta.high` ≥ 0; (3) every `types.json` verdict `OK`; (4) `MLX-PAIRED-vs-DEV2.0-9B-T1.json` card-eligible
  Choice + Noul `ci95.high` ≥ 0; (5) vs Lux1 `ci95.low` > 0, `H.delta.high` ≥ 0 vs Lux1 and Nimble v2, types `OK`;
  (6) exposure `x60.json` `14e7c0ca…` has `groups: []`; (7) `PUBLIC231-vs-DEV2.0-9B-T1.json` not `REGRESSION`.
  If the 23:15 rule ships T = 1, items 1–3 and 5 are read on the derived T = 1 run (answers unchanged).
- **Item 8** only for a passer: a frozen package on node A plus a C1 successor spec for the eval custodian.
- **Diagnostics, never selection:** `hs1-dev` for K5-a12 and the incumbent (CAL698 reused); HT-DEV v2 vs `9b-m4-K-a13`
  (reported only: M6's rule was frozen before the 04:10 note).
- Also reported: T / H axes, types and families, per-task CSS15, public 231 by tier, typed Brier / ECE, Score levels,
  mlx-diag.
- No checkpoint, calibration or limit change after collection; faults are recorded, not retried blindly. Nothing goes
  to Hugging Face.

## Launch (node A GPU6; VRAM 0, lease `track=9b-m6`)

Chain `m6-post-K5-a12` = `bash $L/launch.sh m6-post-K5-a12 $L/chains/m6-post.sh <size> <sha256> e2daf2afc621… 6 K5-a12`
from the `e2daf2afc` mirror (size and SHA-256 checked by `launch.sh`): `m6-formal.sh` (smoke, typed FINAL + CSS15 +
public 231, mlx-diag, seal, report, compares, gates, successor summary; `hs1-dev` K5-a12 and incumbent, `hs1.sh`;
`ship_cal.sh`; `derive_t1.sh` if it ships T = 1), then `m6-htdev2.sh`. Budget: 12.0 GPU-h used; this stage ≈ 0.5.

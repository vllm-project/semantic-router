# 9B Milestone 4: own-Lux trust region on XL r2, human arms, and interpolation chosen on development data (preregistration)

Status: frozen 2026-09-29 before any Milestone 4 GPU step (the two TRAIN builds below ran on
CPU). Goal: a 9B candidate that clearly beats Lux 1.0 — post-key same-panel JevArena v3 paired
95% interval lower bound > 0 against the native Lux1 16K run (65.808, eval run
`m1/d1-lux1-autotune-cache`; the stricter comparator; the same-renderer 16K control 65.231 is
also reported), with human transfer not significantly worse and no decision type collapsed.
Development readouts are never release scores; v3 is post-key.

Handoff (Milestone 3, [result](lux9b-m3-result-2026-09-28.md)): DW = ½·(arm D three-seed soup) +
½·Lux 1.0 reached v3 68.571, +2.763 [−2.085, +6.202]: typed gain significant (T +.076
[+.054, +.098]), human transfer flat with large task-level swings (H −.006 [−.080, +.049];
reddit humor −.167). The v3 interval is dominated by the H axis (median over 15 resampled
tasks), so the lever is to keep the typed gain while moving human-transfer behavior less, or
up. Full fine-tuning was seed-unstable (typed-DEV transition table collapsed in 3 of 5 runs);
soups and interpolation toward Lux repaired it. Arm D's own-Lux KL covered only its 23% recipe
rows; A7 / v1 rows were gold-only.

## Fixed configuration (every training arm)

| Field | Value |
| --- | --- |
| Start | own `llm-semantic-router/Decision-1.0-Lux-9B@bd45a30a…` (bundle `985ade73…`), 7,940,895,744 deployed parameters |
| Trainer | decoder track `v2/dec/train_dec.py`, unchanged; **full fine-tuning** with arm D's E8F settings: backbone lr 1e-5, head lr 1e-4, weight decay .01, warmup .05, one epoch; token-budget micro-batches ≤ 32,768 tokens / ≤ 64 rows, ≥ 64 rows per update; max length 8,192 (longest TRAIN row 7,754 native tokens), gradient checkpointing |
| Objective | CE + 0.5·Brier over offered options; teacher term per arm (below) |
| Selection / calibration | SELECT700 `32a4352d…`, eight evenly spaced checkpoints, family-macro accuracy with earliest tie (`matrix-v1`); **CAL698** `19cc1a8c…` per-type temperatures for every read artifact |
| Runtime | pinned image `sha256:f83b1d10…` with FLA 0.5.2 (the node-A image of every Milestone 3 run and of the Lux1 comparators); Milestone 4 training autotune cache = one copy of the Milestone 3 training cache; node A **GPU6–7** only |
| Code | `lux9b/m3_data.py` recipe budget (`e80b6796c`), node wrappers `lux9b/m4/` (`f7188b497`) |

## Data (both builds: `lux9b.m3_data`, CPU, node A; strict own-Lux coverage and all guards passed)

Source: research & data **XL r2** `mx-xl-full-r2` (private dataset revision
`100536133e192c54ec57c2599a5e4706f6d334ff`, ids `7843afb7…`, 365,970 rows / 179.2M native tokens;
A0s-strict base; A7 Stage curricula, A7q / A7s / A7r / A7k, data v2, v1 arms, H7, H8). Own-Lux
targets for every row: revision `75e557f170979bdbc428b6ea698a2047e2d2a5cd` (`coverage-r2.json`
`ecb6dc36…`; A0s-strict `9f24801b…`, RP-v2 waves 1–4, XL waves w1–w5, h-w1), joined by id and
checked by input hash and option keys. Recipe-level budget: **60,000,000 native tokens**, whole
`group_id` groups, strata pool × source × task type × language (proportional), seed
`20260929:recipe`. Guards: no sealed-C1 candidate source, no MASSIVE / PAWS-X / XNLI, isolation
against SELECT700 and CAL698, 0 duplicate inputs.

| TRAIN | Spec | Rows | Native tokens | C / N / S token share | Languages | train.jsonl | teacher.jsonl |
| --- | --- | ---: | ---: | --- | ---: | --- | --- |
| **x60** (all pools) | `m4-k-xl-r2-60m.json` `f1767dfe…` | 122,651 | 60,183,732 | .318 / .455 / .227 | 35 | `a66131b1…` | `cdcd99c1…` |
| **xn60** (without A7q, H1, H8) | `m4-kn-xl-r2-nohum-60m.json` `dc12cc52…` | 108,086 | 60,130,676 | .343 / .457 / .201 | 22 | `27e57b4d…` | `5bf3e3dd…` |

x60 holds 10.58M tokens of the human social-text / preference arms arm D lacked or under-sampled
(A7q OASST ratings 3.37M, v2 H1 cross-domain human 3.02M, H8 17-language human 4.19M) plus H7
long evidence 5.43M and A7 Stage curricula ≈ 17.5M (A7g 10.06M). xn60 replaces the three human
arms by proportionally more of every other pool (matched tokens). Full hashes are in the build
manifests (`/data/dev2/runs/9b/m4/data/<name>/build/manifest.json`, node A) and the result record.

## Arms (one factor per contrast; matched tokens)

| Arm | TRAIN | Teacher term | Seeds (order) |
| --- | --- | --- | --- |
| **K** — trust region | x60 | **+ 1.0·KL(own Lux ‖ student) on ALL rows** (strict coverage) | 20260926, 1, 2 |
| **P** — plain full fine-tuning | x60 (byte-identical to K) | none | 20260926, 1 |
| **KN** — K without the human arms | xn60 | as K | 20260926, 1 |

Contrasts (development readout, seed-paired): **K − P** = the own-Lux trust region on every row;
**K − KN** = the human arms at matched tokens. Each arm's artifact is the uniform FP32 soup of its
SELECT-chosen seeds if its development proxy P is ≥ its seed mean, otherwise its median seed
(coordinator seed rule).

## Interpolation lines and the development-only α rule

Every line is θ(α) = α·S + (1 − α)·Lux 1.0 (FP32; rational α built by repeated soup members),
CAL698 temperatures per artifact, read on typed DEV 1,600 + CSS pilot 1,430 through the same
runtime as a fresh Lux 1.0 reference readout (checkpoint-form Lux `dc9b795c…`). No development
prompt exceeds 5,135 tokens, so 8K and 16K readouts coincide.

| Line | S | α grid |
| --- | --- | --- |
| K | K artifact | ¼, ⅓, ½, ⅔, 1 |
| U | ½·(D soup `9a1d7db8…`) + ½·(K artifact) | ¼, ⅓, ½, ⅔ |
| D | Milestone 3 D soup (`9a1d7db8…`; α = ½ is DW) | ¼, ⅓, ½, ⅔, 1 |
| KN | KN artifact | ½, 1 |
| P | P artifact | ½, 1 |

**α rule (never v3).** Let T = typed-DEV family macro, c_t = typed-DEV correct count of type t,
n_t its items, F_f = typed-DEV family accuracy, H3 = CSS-pilot three-task mean macro-F1, and the
subscript L the Lux reference. α is **eligible** if (i) c_t ≥ c_t,L − 0.03·n_t for Choice, Noul
and Score; (ii) H3 ≥ H3_L; (iii) no typed-DEV family below F_f,L − 0.10. Let G(α) = T(α) − T_L and
G* the largest G over eligible α. A line has no pick if no α is eligible or G* < 0.01. Otherwise
**α\* = the smallest eligible α with G(α) ≥ 0.75·G\*** — the artifact nearest Lux that keeps
three quarters of the line's best typed gain, because Milestone 3 showed typed gains transfer
post-key while the development panels cannot see human-transfer drift.
If the eval track's broader human-transfer development panel appears in COORDINATION "Eval
runners" before the α readouts are scored, it replaces H3 in (ii) with that panel's
recommended aggregate (the α grid is then also read on it) and H3 becomes report-only.

## Finalists, formal runs and gate

- Candidate per line = θ(α\*). Proxy v2 drop rule only: an artifact whose P = 100·√(T·H_pilot
  median) is ≥ 8 below the best line's is dropped (P never ranks siblings).
- **At most three finalists**, taken in the fixed priority **K, U, D, KN, P** (hypothesis order,
  not development rank). D at α\* = ½ is DW, already run (68.571): its slot passes on. Selection
  happens once the K, U, D and KN lines are read; P fills a slot only if fewer than three remain.
- Formal: the frozen same-panel runner on node A (GPU6 or GPU7), 16,384 tokens, image
  `f83b1d10…`, one copy of the frozen Milestone 3 formal autotune cache (tree hash logged before
  and after), smoke then full panels (typed FINAL + CSS15 + public 231), then mlx-diag; seal,
  report, paired compare vs native Lux1 16K and the same-renderer control.
- **Gate:** v3 paired 95% lower bound > 0 vs native Lux1 (65.808); human-transfer axis interval
  not entirely below 0; no type collapsed (`v2.eval.gates types`). Also reported: T / H axes,
  per-type, per-task CSS15, public 231 (easy / standard / hard), typed Brier / ECE, mlx-diag.
- A passing finalist is staged privately in `llm-semantic-router/dev2-9b-staging` (`m4/<name>/`),
  in the most compact loadable form that stays within the 100 GB private cap (headroom checked
  first), and reported to the coordinator at once with its scored run directory.

## Schedule, budget and stop rules

- Waves (GPU6 / GPU7): K-s20260926 with preflights / (Lux reference + D-line readouts, then P
  preflights + P-s20260926); K-s1 / KN preflights + KN-s20260926; K-s2 / KN-s1; then K, U, KN
  lines read + finalists formal / P-s1. Preflights for K, P and KN (zero-step, one update,
  parity + bitwise reload) must PASS or the arm stops, without retry.
- **Budget ≈ 20 GPU-hours** (≈ 2.35 h per 60M-token seed at the measured 7,100 tokens/s; seven
  runs ≈ 16.5, preflights ≈ 0.45, ≈ 17 development readouts ≈ 1.0, three formal runs ≈ 0.75).
  **Hard cap 22:** a run whose projected completion would exceed it does not start (budget only,
  never results; P-s1 is last in line). A reproduced ROCm fault, OOM or nonfinite loss stops that
  arm. No failed arm is rerun to fill GPUs.
- GPU-hours = wall-clock × GPUs, including preflights, readouts and formal runs; soups and data
  builds are CPU.

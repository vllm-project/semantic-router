# Decoder Milestone 4 — preregistration (4B, own-Lux line; 2026-09-29)

Written and pushed before any Milestone 4 mixture is built or any arm is launched. Development readouts
(SELECT700, CAL698, typed DEV, CSS pilot) are never release scores; JevArena v3 / public JevBench 231 numbers are
post-key same-panel comparisons.

## Goal and bar

A 4B candidate whose paired post-key 95% CI lower bound on the v3 composite is > 0 against the **adopted Nox 1.0
run (56.470, `/data/dev2/runs/eval/m1-adopt/nox1`)**, the stricter of the adopted run and the 16K same-limit
control (55.689), with human transfer held (CSS15 H not below Nox 1.0's .519). Decider 4B (61.882) and Jet v6.2
(60.375) are reported. Handoff state (M3, [`dec-m3-results-2026-09-28.md`](dec-m3-results-2026-09-28.md)): the
best 4B soup N4LKr (own-Lux targets on the recipe rows, own-Nox on the retention rows, KL 1.0) is 59.539,
+3.07 [−4.90, +5.00]. v3 = 100·√(T·H) with H the median CSS15 task macro-F1, and the paired bootstrap resamples
the 15 tasks, so a lower bound > 0 needs a gain that is broad across tasks (roughly +5 v3).

## Inputs decided before this record (disclosed)

1. **XL recipes: r2, not r1.** The coordinator's brief names r1 (`ba848147…`, `m3/mixtures/xl/`). Research &
   data superseded it with r2 final (amendment 4, revision `100536133e192c54ec57c2599a5e4706f6d334ff`,
   `m3/mixtures/xl-r2/`): r1 holds 727 rows (305 groups) whose rescreen hits are on evaluation panels (CSS15,
   Decision Bench v4, JevBench-231, `mlx-diag`, `ml-parallel-dev`). M4 uses r2 only.
2. **The M3 mixture carries 57 rows of those 305 groups.** `m3-v2m-ret` (`13804ac6…`, used by N4T / N4J / N4L /
   N4LKr and by the 2B release candidate S2T) has 57 rows in r2-excluded groups: 45 with CSS15 hits and 15 with
   Decision Bench v4 hits (sources MuSiQue 41, SQuAD 2.0 8, QuAC 4, CommonsenseQA 2, HotpotQA 2; Noul 42 /
   Score 13 / Choice 2). Found while preparing M4, before any M4 training; reported to the coordinator as a
   disclosure item for those candidates. M4's M3-line arms drop them with the new builder option
   `exclude_group_ids` and the list [`specs/m4-r2-excluded-groups.json`](../specs/m4-r2-excluded-groups.json)
   (305 ids, `c265b704…`), derived by the amendment-4 rule from research & data's private rescreen receipt
   (`c01fe566…`, the hash pinned in the r2 manifest).
3. **Proxy v2 is preregistered by the eval track (`79482d863`) but has no result yet** (selection rule 3 below).
4. **Own-Lux coverage of the M3 retention rows.** The published own-Lux files (XL waves `w2` / `w3` / `w4` /
   `c-w1`) cover 7,165 of the 8,276 retention rows (id and input hash); 1,111 A7g rows are in no published file.
   They are labeled here with `v2.dec.teacher_label` (same Lux 1.0 revision, its published per-type temperatures,
   the kernel image, node B), with a 300-row cross-check against published targets.

## Frozen for every arm

- **Start:** own `llm-semantic-router/Decision-1.0-Nox-4B@cde2a68dbaa557ea65dc458104d410a0802ee259`.
- **Recipe M3F, unchanged from N4LKr:** full fine-tuning, backbone LR 5e-6 / head LR 5e-5, AdamW (weight decay
  0.01, clip 1.0), 5% linear warmup then cosine to 10%, one epoch, token batching (32,768 tokens / 64 rows per
  micro-batch, 64 rows per update), 8,192-token limit without truncation, CE + 0.5 Brier + w·KL(teacher ‖
  student), checkpoints on `even8`, SELECT700 matrix-v1 selection (earliest wins ties), CAL698 temperatures.
  Seeds 20260926 / 20260927 / 20260928 (M3's, so seed-paired contrasts are possible). One GPU per seed.
- **Teacher:** own `llm-semantic-router/Decision-1.0-Lux-9B@bd45a30aee8c84032791c245c70f86dee5389cc8` on
  **every row** of every arm (XF: all rows except its gold-only H7 / H8 rows, trained with `--teacher-partial`),
  composed by the new `v2.dec.compose_teacher` (first published source wins; values copied unchanged; overlaps
  reported). Sources, in precedence order (private dataset `llm-semantic-router/decision-2.0-training-data`):
  `m3/pk1/lux1/A0-train.canonical.jsonl@d8eae3e4` (`56627939…`); `m2/teachers/lux1/rp-v2/wave1@7885baf6`
  (`47049a6b…`), `wave2@6bd8eb4d` (`2c6ab38d…`), `wave3@002e5b42` (`450bdd7b…`), `wave4@03b1e72d` (`084adacd…`);
  `m3/teachers/lux1/xl/w1…w5, c-w1@100536133e` (`4a852b74…`, `a659c3e5…`, `6d141c96…`, `b8ae1374…`, `0da12210…`,
  `1a6b5a80…`); last, the M4 gap labels (item 4 above). Teacher provenance: own model, no third-party caveat;
  the published files come from the node-B Lux image without causal-conv1d (their disclosure applies).
- **Token budget:** 29,249,047 native tokens (the M3 mixture), matched within ±1% by every build (whole-group
  sampling); a build outside ±1% is a failed preflight.
- **Runtime:** node B GPU0–4, image `sha256:dbe5f32b2263…` (FLA + causal-conv1d), frozen autotune cache
  `/data/dev2/runs/dec/triton-cache/dbe5f32b2263`; SELECT700 / CAL698 from `/data/dev2/runs/dec/m3/data-sel700-cal698`.
- **Code:** exact mirror of the commit that adds this record; launchers `v2/dec/ops/m4/` (lock-guarded).

## Arms (three seeds + uniform soup each)

| Arm | Mixture (spec) | KL w | Single-factor contrast |
| --- | --- | ---: | --- |
| **N4LR** | `m4-v2m-ret-r2`: the M3 mixture minus the 57 excluded rows (A0s + v2-M pools + the same A7 v3 `dec10-stage4v2` retention draw) | 1.0 | vs M3 N4LKr: own-Lux instead of own-Nox targets on the retention rows |
| **N4LR2** | `m4-v2m-ret-r2` | 2.0 | vs N4LR: KL weight 2.0 vs 1.0 |
| **N4LRQ** | `m4-v2m-ret-r2-q20`: as N4LR, v2-M pools cut to 62.30% (whole groups) and A7k + A7s (all) + A7q (r2 a7v1 ids, 3,092,510-token draw) filling 20% of tokens | 1.0 | vs N4LR: human multilingual Score retention (A7q / A7k / A7s) in place of v2-M rows |
| **N4XA** | `m4-xl-a7v1-29m`: the M3 A0s rows + every other pool of `cx-xl-r2-a7v1-full` at 22.12% (whole groups): A7 g/h/i/k/m/o/p/q/r/s + v1 A1–A6h, no v2 | 1.0 | vs N4LR: the XL A7-only recipe vs the M3 mixture |
| **N4XF** | `m4-xl-full-29m`: the M3 A0s rows + every other pool of `mx-xl-full-r2` at 14.42%: as N4XA plus the v2 pools and the gold-only H7 / H8 gap arms | 1.0 | vs N4XA: the full XL recipe vs its A7-only control |

No arm contains A7x. The A0s rows (6,547; = A0s-strict) are identical in every arm, as are the retention rows of
N4LR / N4LR2 / N4LRQ (the M3 seed string and budget are kept so the same groups are drawn; checked by id).

## Preflights (stop and record on failure; no rerun to fill GPUs)

1. Build manifests: tokens within ±1% of 29,249,047; zero rows of the 305 excluded groups; A0s ids identical to
   M3's; N4LR-line retention ids identical to M3's.
2. Gap labels: argmax agreement with the published targets ≥ 0.97 on the 300 cross-check rows. Otherwise the
   1,111 gap rows keep own-Nox targets (the M3 state), disclosed.
3. Teacher coverage: every TRAIN row except XF's H7 / H8 rows; the trainer re-validates hashes, keys and sums.
4. Per seed (`drive_arm.sh`): zero-step run, one-step run + reload, preflight receipt PASS before the full run.
5. Cost cap 1.6 GPU-h per seed. Milestone budget ≈ 20 GPU-h: 15 seeds × ≈ 0.95 + soups ≈ 0.4 + labels ≈ 0.1 +
   formal ≤ 0.6 ≈ 15.4 planned.

## Selection (development panels only; v3 never used)

1. **Artifact per arm:** the uniform soup if its selection statistic R is ≥ the seed mean of R, else the median
   seed by R.
2. **Eligibility:** typed-DEV correct counts per type ≥ 75% of Nox 1.0's (Choice ≥ 345, Noul ≥ 171, Score ≥ 284),
   every answer valid, and no type answered with one constant value.
3. **R:** the eval track's recommended proxy v2 if its result record is on the integration or eval branch when
   the last per-arm soup readout lands; otherwise **P_mean3 = 100·√(T_dev·H_mean3)** (H_mean3 = mean of the three
   CSS-pilot tasks' macro-F1). The pilot median H is reported, never used. Tie band: proxy v2's band B, or 4 P
   units for P_mean3.
4. **Finalists (at most two):** the two highest-R eligible artifacts, excluding any whose R is below the M3 N4LKr
   soup's R (same readout) by more than the tie band. If the second and third are within the tie band, the one
   with the higher min(C/460, N/228, S/378) (typed-DEV per type relative to Nox 1.0) goes.
5. **One cross-arm soup, N4LX:** the uniform soup of the six seeds of the two highest-R eligible arms (same start
   and recipe family), built after the per-arm readouts; same eligibility; it competes for the two slots.

## Formal runs (finalists only)

Frozen formal runner, node A GPU5 with the image and frozen autotune-cache procedure of the adopted Nox 1.0 run and
the M3 formal runs (comparability rule): package `qwen-full` at 16,384 tokens; CAL698 16K temperatures adopted
only if they do not worsen typed-DEV and CSS-pilot ECE and Brier (the 23:15 calibration rule), else T = 1. v3 +
public 231 + `mlx-diag`. Paired comparisons: adopted Nox 1.0 (the bar), Nox 1.0 16K control
(`/data/dev2/runs/dec/formal/m3/nox1-16k`), Decider 4B (`m1-adopt/decider4b`), Jet v6.2 (`m2/q5b-jet62`), M3
N4LKr (`/data/dev2/runs/dec/formal/m3/m3-N4LKr-soup-nodeA`).

**Qualifies** if the lower bound vs the adopted Nox 1.0 run is > 0, CSS15 H ≥ .519, typed-FINAL types ≥ 75% of
Nox 1.0's, and no type collapsed. A qualifying soup is staged privately (`llm-semantic-router/dev2-dec-staging`,
soups only, after an HF storage-headroom check) and reported at once with its scored run dir. `mlx-diag` and
every per-task / per-language regression are reported.

## Optional 2B probe (not required)

Only if ≥ 3 GPU-h remain after the 4B formal runs: the best 4B recipe on own Sol 1.0 (three seeds + soup),
development readout; a formal run against the 2B release candidate S2T only if its R beats S2T's by the tie band.

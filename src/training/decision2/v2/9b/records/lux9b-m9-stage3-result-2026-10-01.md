# 9B Milestone 9, stage 3 result: K-a13IB (K-a13 recipe + IB1-r3 + IB2 at matched tokens) passes every development gate and items 2–7, but fails item 1 (v3 +0.29 [−1.57, +1.20]); no successor

Run under the [preregistration](lux9b-m9-prereg-2026-10-01.md) (`8570a5896`) and
[amendment 3](lux9b-m9-prereg-amendment-3-2026-10-01.md) (`787abdc54`, stage 3), pushed before any stage-3 job; formal
lock [`lux9b-m9-formal-lock-s3-2026-10-01.md`](lux9b-m9-formal-lock-s3-2026-10-01.md) (`5d0c1c234`), pushed before any
formal prediction. Development readouts are never release scores; the formal comparison is **post-key same-panel**.
**DEV2.0-9B (K-a13 at T = 1) stays the 9B model.** Nothing was uploaded; no Index row was read; C1 was not opened
(item 8 runs only for a passer of items 1–7).

## Verdict

- **Official stage-3 rules (16:06Z): one finalist, K-a13IB.** It passes all seven development gates (typed T .9344 vs
  .9250, HT-DEV v2 TIE, **less** near-miss yes-bias than C0: PN1 clean gold-no −.015 [−.028, −.003], `hs1-dev`
  false-yes .134 vs .152; MLX-DEV-9B and retention level). K-a13IBX (without the in-distribution families) fails the
  Noul type floor (319 < 326), the `rule_precedence` floor (319 < 334) and Y1 (+.032 [+.019, +.045]).
- **Formal (node A GPU6, 16:09–16:22Z): K-a13IB v3 68.024 vs the released T = 1 run 67.737: +0.29 [−1.57, +1.20], so
  item 1 fails.** Items 2–7 pass: human transfer +.007 [−.021, +.020]; types OK; card-eligible mlx-diag −.004 [−.011,
  +.003]; tier gates (vs Lux1 16K +2.22 [+0.11, +3.95], vs Nimble v2 +5.97 [+3.03, +9.54]); exposure 0 (TRAIN ⊂ x60 ∪
  IB1-r3 ∪ IB2, both receipts 0 groups); public 231 182 vs 178 (+4, McNemar p .125).
- **Why item 1 fails:** the typed-DEV gain (+.009, 15 of 1,600 items) did not reach typed FINAL (T .8069 vs .8106,
  −.004 [−.019, +.012]; Choice 732 vs 736, Noul 704 vs 715, Score 255 vs 246). Human transfer is up but not
  significantly (H .5735 vs .5660).
- **No successor, so no item 8, no release and no private Index run of a release.** The custodian's C1 content
  recheck r1 ([`c1-recheck-r1-2026-10-01.md`](../../eval/records/c1-recheck-r1-2026-10-01.md), verdict `0823a1a8…`)
  maps K-a13IB and K-a13IBX with exposure 0, so item 8 would have been allowed; it was not run because item 1 failed.
  The optional CAL-only Noul T+b study applies only to a chosen passer of items 1–7, so it was not run.
- **Breadth on the proven recipe is clean** (H4, H5 hold): IB is learned and survives the ⅓ interpolation (IB1 DEV
  +.040, IB2 DEV +.081 vs C0; ≈ 80% of the α 1 soup's gain) with no yes-bias or transfer cost. At ⅓ from Lux,
  though, the typed gain stays inside development noise.

## Design as run

| Arm | Seed TRAIN (`data/READY3.json`) | Seeds' BEST (of planned) | ⅓ point (FP32, node A) |
| --- | --- | --- | --- |
| K-a13IB | `kib` `2cd09292…`: 102,172 x60 rows (whole-group stratified cut, 50.27M tokens) + 48,843 IB1-r3 / IB2 rows (10.17M tokens; IB share 16.8%); 151,015 rows, 60.44M tokens | 2,083 / 2,081 / 2,084 (of ≈ 2,083) | [KIB soup, Lux, Lux]; `model_sha256` `4701ba41…` |
| K-a13IBX | `kibx` `548b61a5…`: 107,588 x60 rows (52.88M) + 37,892 IB rows without `isarc` / `w2c` / `hover` / `gsm2` (7.50M; 12.4%); 145,480 rows, 60.38M tokens | 1,999 / 1,489 / 1,745 (of ≈ 1,990) | [KIBX soup, Lux, Lux]; `model_sha256` `d559b85c…` |

- **Recipe = K-a13's** (M4 K-s1..s3 trainer contract): full fine-tuning of Lux 1.0 `bd45a30a…` (backbone LR 1e-5,
  head 1e-4), CE + 0.5·Brier on gold + 1.0·KL(own Lux) on every x60 row, IB rows gold only (`--teacher-partial`),
  64 rows per update, one epoch, `even8` checkpoints with SELECT700 selection, seeds 20260926 / 1 / 2. Node C GPU3 / 6
  / 7 (K-a13IB, launched 10:47Z with s1's pre-warm first, done 13:23–13:28Z) and GPU2 / 1 / 4 (K-a13IBX, 12:51–15:32Z);
  all six preflights passed; no seed failed.
- **Item 6 inputs:** both TRAIN files are line-level subsets of x60 (`a66131b1…`, the released K file) ∪ IB1-r3 TRAIN
  (`1e1b08f3…`) ∪ IB2 TRAIN (`ee137efa…`): `exposure/kib-subset.json` `1dfb1843…` (0 of 151,015 rows outside),
  `kibx-subset.json` `af6ca288…` (0 of 145,480); exposure receipts x60 (`m6/exposure/x60.json` `14e7c0ca…`) and IB
  (`exposure/ib1-ib2-train.json` `9fd9d3db…`) list 0 groups.
- **Artifact:** uniform FP32 soup of the three seeds' BEST checkpoints (node C, CPU), then [arm soup, Lux 1.0, Lux 1.0]
  (α = ⅓) on node A with the zero-step checkpoint of K-a13IB-s1 as the Lux member (amendment-3 fallback; byte-identical
  backbone and head to K-a13's own base, re-checked after the pull).

## Development readouts (node A, the M9 path; vs C0 = DEV2.0-9B's weights)

Official rules `select/9b-finalists-s3.json` `4b02f31f…` (16:06Z), typed readout `lines/readout/m9-s3.json`
`d312f8e7…`.

| Point | typed T | C / N / S | RP | H3 | HT-DEV v2 Δ vs C0 | PN1 clean gold-no (Δ) | hop Δ | `hs1-dev` false-yes | MLX-DEV-9B Noul / Choice Δ | retention macro (Δ) | Gates |
| --- | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | --- | --- | --- |
| C0 | .9250 | 799 / 338 / 343 | 338 | .5622 | — | .262 | — | .152 | — | .792 | ref |
| **K-a13IB** | **.9344** | 800 / 344 / 351 | 344 | .5566 | −.006 [−.017, +.005] TIE | .247 (−.015 [−.028, −.003]) | .000 | .134 | −.001 [−.008, +.005] / −.000 [−.015, +.014] | .792 (−.000 [−.009, +.009]) | **all pass** |
| K-a13IBX | .906 | 800 / 319 / 331 | 319 | .5537 | −.013 [−.022, −.003] TIE | .294 (+.032 [+.019, +.045]) | .000 | .149 | −.001 [−.006, +.005] / +.002 [−.010, +.014] | .788 (−.004 [−.014, +.005]) | Noul type floor, RP floor, Y1 |
| L9IB (stage 2) | .8881 | 770 / 340 / 311 | 340 | .5716 | −.021 FLAG | .379 (+.117) | +.004 | .229 | +.027 / +.013 | .809 (+.017) | 5 gates |
| L9IBX (stage 2) | .9381 | 800 / 364 / 337 | 364 | .5920 | −.022 FLAG | .320 (+.058) | +.008 | .271 | +.023 / +.013 | .791 (−.001) | 3 gates |

- K-a13IB per probe: MMLU .775 / ARC .967 / GSM8K .633 (C0 .776 / .966 / .635); Score5-typed-DEV check half clean
  for both stage-3 points. PN1 PAWS-X-6 / all-8 yes-rate Δ: K-a13IB −.006 / −.006; K-a13IBX +.014 / +.015.

### IB DEV (report only; micro accuracy, Δ vs C0)

| Slice | C0 | K-a13IB (⅓) | KIB soup (α 1) | K-a13IBX (⅓) | KIBX soup (α 1) | L9IB | L9IBX |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IB1 DEV (2,067) | .924 | .963 (+.040 [+.031, +.048]) | .971 (+.048) | .957 (+.034) | .959 (+.036) | .971 | .964 |
| `args` / `copa` / `poem` / `sentfin` | .874 / .944 / .929 / .945 | .920 / .954 / .993 / .959 | .944 / .944 / .995 / .968 | .922 / .963 / .998 / .960 | .939 / .954 / .995 / .966 | .941 / .963 / .998 / .970 | .941 / .963 / .998 / .968 |
| `sms` / `snips_rel` / `snips_sel` / `isarc` | .823 / 1.0 / .990 / .720 | .935 / 1.0 / .997 / .951 | .952 / 1.0 / 1.0 / .951 | .952 / 1.0 / .997 / .732 | .952 / 1.0 / 1.0 / .671 | .935 / 1.0 / 1.0 / .915 | .935 / 1.0 / 1.0 / .744 |
| IB2 DEV (1,272) | .809 | .890 (+.081 [+.061, +.102]) | .910 (+.101) | .789 (−.020) | .798 (−.011) | .905 | .796 |
| `argq` / `ytspam` / `gsm2` / `hover` | .995 / .935 / .670 / .770 | .995 / .978 / .802 / .885 | .998 / .967 / .830 / .930 | .995 / .967 / .626 / .755 | .998 / .978 / .670 / .675 | .998 / .957 / .830 / .905 | .998 / .967 / .657 / .700 |

(CIs: paired bootstrap, 2,000 draws, as read at 14:00Z for K-a13IB; the other columns are point values.)

### Contrasts (report only)

| Contrast | typed T Δ (C / N / S) | HT-DEV v2 Δ | retention Δ | PN1 clean gold-no Δ |
| --- | --- | --- | --- | --- |
| K-a13IB − K-a13IBX (H6) | +.028 (0 / +25 / +20) | +.007 [−.004, +.017] TIE | +.004 [−.006, +.014] | −.047 [−.062, −.032] |
| K-a13IB − L9IB | +.046 (+30 / +4 / +40) | +.015 [−.000, +.029] TIE | −.017 [−.030, −.003] | −.132 [−.156, −.109] |
| K-a13IBX − L9IBX | −.032 (0 / −45 / −6) | +.009 [−.004, +.023] TIE | −.003 [−.015, +.009] | −.026 [−.045, −.008] |

## Formal (K-a13IB; node A GPU6, 16,384 tokens; vs I = the released T = 1 run, v3 67.737)

Chain `formal-s3` (mirror `787abdc54`) after `status/formal-s3.GO` (16:09:13Z): CAL698 fit, gold-free smoke, typed
FINAL + CSS15 + public 231 (`formal-m9/K-a13IB-16k`, 441 GPU-seconds), mlx-diag, seal, report, compares, gates
(16:22Z). The autotune cache tree stayed `48a2611aa852…` before and after every collection. The C0F formal-path
parity is exact (0 answer differences vs I), so I is the bar. Items from `formal-m9/K-a13IB.gates/items.json`
`3e00533b…` (`lux9b/m9/items.py verdict`, mirror `f1bb2a50f`) over `successor.json` `5e4a7258…`.

| | K-a13IB | I (DEV2.0-9B, T = 1) |
| --- | --- | --- |
| v3 = 100·√(T·H) | **68.024** | 67.737 |
| typed FINAL T (Choice / Noul / Score correct) | .8069 (732 / 704 / 255) | .8106 (736 / 715 / 246) |
| CSS15 H | .5735 | .5660 |
| public 231 (easy / standard / hard) | 182 (48 / 67 / 67) | 178 (48 / 66 / 64) |
| mlx-diag card-eligible Choice + Noul | .801 | .805 |
| Score levels used / level-0 recall | 5 / .43 | 5 / .51 |

| Item | Result | Pass |
| --- | --- | --- |
| 1 v3 vs I | +0.29 [−1.57, +1.20] (T −.004 [−.019, +.012], H +.007 [−.021, +.020]) | **no** |
| 2 human transfer vs I | `H.delta.high` +.020 ≥ 0 | yes |
| 3 types | Choice / Noul / Score OK | yes |
| 4 mlx-diag card-eligible vs I's mlx answers | −.004 [−.011, +.003] | yes |
| 5 tier | vs adopted Lux1 16K +2.22 [+0.11, +3.95]; `H.delta.high` +.042 vs Lux1 and +.097 vs Nimble v2 (v3 +5.97 [+3.03, +9.54]); types OK | yes |
| 6 exposure | x60 and IB receipts 0 groups; 151,015 of 151,015 TRAIN rows in x60 ∪ IB1-r3 ∪ IB2 | yes |
| 7 public 231 vs I | +4 (McNemar p .125), OK | yes |
| 8 C1 post-key | not run (item 1 failed) | — |

CSS15 per-task macro-F1 vs I: 7 of 15 tasks up (largest `raop` +.021, `flute` +.011, `reddit_humor` +.011), 8 down
(largest `indian_english_dialect` −.046, `conv_go_awry` −.019, `media_ideology` −.012).

## Reading

- **H4 (breadth on the proven recipe passes the seven gates): holds.** Unlike stage 2's from-base adapter, the K
  recipe with ⅓ interpolation absorbs a 16.8% IB share with *less* near-miss yes-bias than C0 and no transfer or
  retention cost.
- **H5 (the IB DEV gain survives ⅓): holds.** K-a13IB keeps 83% (IB1, +.040 of +.048) and 80% (IB2, +.081 of +.101)
  of the α 1 soup's gain.
- **H6 (the in-distribution families): at the K recipe they help.** K-a13IB − K-a13IBX: typed +.028 (Noul +25, Score
  +20), PN1 clean gold-no −.047 [−.062, −.032], IB2 DEV +.101 (`gsm2`, `hover`). In stage 2 the same families raised the
  yes-bias of the from-base adapter (L9IB − L9IBX +.059). At ⅓ toward the soup, `isarc` / `w2c` / `hover` / `gsm2`
  act as extra "no"-labelled near-miss practice rather than as a source of bias.
- **Formal:** K-a13IB is level with DEV2.0-9B (v3 +0.29, H +.007, public 231 +4, mlx −.004, all n.s.). The +.009
  typed-DEV gain was within noise and turned into −.004 on typed FINAL (Noul −11 items). K-a13IB adds breadth at no
  measured cost but is not a successor under item 1.
- **What this says about the 9B lever:** every 9B point that kept the yes-bias in check (K-a13, K-a13IB) sits ⅓ from
  Lux, and at that distance the typed gain is small. The largest formal gain over DEV2.0-9B so far came at α ½
  (K5-a12, +1.62 [−0.19, +2.41]), which also missed item 1 and failed item 4 through the PAWS-X yes-bias that grows
  with distance from Lux. IB is the first data block that *lowers* that bias on the K recipe.

## Proposed next 9B lever (not launched; a coordinator decision and a new amendment / milestone first)

**K-IB at α ½, then a five-seed IB soup, with typed-row self-distillation as the fallback.**

1. **K-a12IB (≈ 1 GPU-h, no training).** The ½ point [KIB soup, Lux 1.0] of the existing three-seed KIB soup, read on
   the same eight panels under the same seven gates (the yes-bias guard and MLX-DEV-9B are the binding risk, as K5-a12
   failed item 4), preregistered before any readout; formal if it passes. Rationale: IB at ⅓ cut the PN1 yes-bias by
   .015 [.003, .028] and the `hs1-dev` false-yes rate by .018 with MLX-DEV level, so ½ may keep the yes-bias within
   the guard while moving typed accuracy toward K5-a12's +.030.
2. **Five-seed KIB soup (≈ 6 GPU-h).** Two more K-a13IB seeds (seeds 3 / 4, M6's K-s4 / K-s5 seeds) on `kib`, then
   the ⅓ / ½ points of the five-seed soup (M6's five-seed K soup gave the largest 9B formal gain, at ½).
3. **If the typed gain stays short: typed-row self-distillation** from DEV2.0-9B on x60's typed rows with IB additive
   (full x60 + IB, M13's `4b-LHA10SD` fix, which kept typed T while adding +.046 IB transfer at 4B), three seeds on
   the K recipe (≈ 9 GPU-h).
4. **IB3:** not release-safe (COORDINATION 2026-10-02 00:10); when IB3-r2 lands release-safe, its maths block can join
   the IB share (custodian C1 recheck required).

Every candidate keeps items 1–8 in force; M9 has ≈ 78 of its 120 GPU-h left. An optional private Index diagnostic of
K-a13IB (IX1 harness, ≈ 2 GPU-h on idle node C GPUs, values private) would size the breadth effect at 9B, as the
coordinator ordered for `08b-RA`; it was not run here because K-a13IB is not a release.

## Resources

Stage 3 ≈ 16.5 GPU-h (cap 30): K-a13IB seeds 7.74 and K-a13IBX seeds 7.75 (node C, preflights included), node-A
builds and readouts ≈ 0.75, formal K-a13IB ≈ 0.21 (CAL698 fit, smoke, typed FINAL + CSS15 + public 231, mlx-diag).
**M9 total ≈ 41.5 of 120 GPU-h** (node C 38.81 by `gpuh.py`; node A readouts 2.29; C0F parity 0.20; formal 0.21).
Node A GPU6–7 and node C GPU1–5 leases released at the close (16:27Z; node C GPU6–7 at 14:05Z).

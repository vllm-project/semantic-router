# 9B Milestone 9, stage 2 result: IB1-r3 + IB2 on the L9 recipe gives no finalist (both arms fail HT-DEV v2 and the yes-bias guard)

Run under the [preregistration](lux9b-m9-prereg-2026-10-01.md) (`8570a5896`) and
[amendment 2](lux9b-m9-prereg-amendment-2-2026-10-01.md) (`51c80ddc9`, stage 2), pushed before any stage-2 job.
DEV2.0-9B (K-a13 at T = 1) stays the 9B model. Development readouts are never release scores. Nothing was uploaded;
C1 was not opened; no Index row was read. Stage 3 ([amendment 3](lux9b-m9-prereg-amendment-3-2026-10-01.md),
`787abdc54`) is running and is reported separately.

## Verdict

- **No stage-2 finalist, so no stage-2 formal run.** Rules `select/9b-finalists-s2.json` (`df0ef550…`), typed readout
  `lines/readout/m9-s2.json` (`cda87bcf…`); the formal chain stopped by rule at 13:28Z.
  - **L9IB** (x60 + IB1-r3 + IB2 on the L9 recipe) fails five checks: the Choice type floor (770 < 775), the Score
    type floor (311 < 331), **HT-DEV v2 FLAG** (−.021 [−.035, −.006]), **Y1** (PN1-dev clean gold-no yes-rate +.117
    [+.094, +.139]) and **Y3** (`hs1-dev` false-yes .229 vs C0's .152).
  - **L9IBX** (without the in-distribution families `isarc`, `w2c`, `hover`, `gsm2`) fails three: **HT-DEV v2 FLAG**
    (−.022 [−.037, −.007]), **Y1** (+.058 [+.039, +.077]) and **Y3** (.271 vs .152).
- **IB is learned.** On the IB DEV slices L9IB gains +.048 [+.038, +.059] (IB1) and +.096 [+.074, +.117] (IB2) over
  C0. L9IBX gains +.040 [+.031, +.050] on IB1; on IB2 it is level (−.013 [−.031, +.004]), because IB2's gain sits in
  the in-distribution families it leaves out (`gsm2` .830 vs .670, `hover` .905 vs .770 for L9IB).
- **IB raises the near-miss yes-bias of the from-base adapter.** L9IB − L9 clean gold-no +.074 [+.055, +.094].
  The in-distribution families carry most of it (L9IB − L9IBX +.059 [+.041, +.078]; L9IBX − L9 +.015 [.000, +.031]).
  The `hs1-dev` unmet-condition false-yes rate rises in both arms (.229 / .271 vs L9 .140).
- **Breadth does not repair the base start's transfer loss.** HT-DEV v2 vs L9 is +.011 (L9IB) and +.009 (L9IBX), both
  TIE, so both stay FLAG vs C0.

## Design as run

| Arm | TRAIN (`data/READY2.json`) | Seeds' BEST | Artifact (FP32, merged) |
| --- | --- | --- | --- |
| L9IB | `a2398217…`: x60 + IB1-r3 + IB2, 171,494 rows, 70,357,087 tokens (IB rows gold only) | s1 1,757, s2 2,344 (of 2,343–2,344) | uniform soup; readout identity `d7c48f9a…` |
| L9IBX | `ec17d822…`: the same without `isarc` / `w2c` / `hover` / `gsm2`, 160,543 rows (IB 7.50M tokens) | s1 1,922, s2 1,914 (of ≈ 2,190) | uniform soup; readout identity `695c0ce2…` |

- Recipe = L9's: LoRA r 128 / α 256 / dropout .05 from Qwen3.5-9B-Base `68c46c4b…`, fresh candidate head (init seed
  20261001), LR 1e-4, own-Lux KL 1.0 on x60 rows, CE + 0.5·Brier, 64 rows per update, one epoch, seeds 20260926 /
  20260927; node C GPU5 / 2 / 4 / 1. All four preflights passed; every seed finished (07:46–12:53Z).
- Merges agree with the adapters on 128 / 128 SELECT rows, except L9IBX s1 at 127 / 128 (one near-tie flip; maximum
  probability drift ≤ .012 everywhere). Soups built 12:56–12:58Z, pulled to node A with matching content manifests.

## Development readouts (node A, the M9 path: `v2.dec.infer_dec` 16K, T = 1; MLX-DEV-9B and IB DEV via `eval_rows`)

| Point | typed T | C / N / S | RP | H3 | HT-DEV v2 Δ vs C0 | PN1 clean gold-no (Δ) | hop | `hs1-dev` false-yes | MLX-DEV-9B Noul / Choice Δ | retention macro (Δ) | IB1 / IB2 DEV acc |
| --- | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | --- | --- | --- |
| C0 | .9250 | 799 / 338 / 343 | 338 | .5622 | — | .262 | .987 | .152 | — | .792 | .924 / .809 |
| L9 (stage 1) | .9369 | 789 / 330 / 380 | 330 | .5841 | −.031 FLAG | .305 (+.042) | .987 | .140 | +.025 / +.007 | .814 (+.022) | .921 / .807 |
| **L9IB** | .8881 | 770 / 340 / 311 | 340 | .5716 | −.021 [−.035, −.006] FLAG | .379 (+.117 [+.094, +.139]) | .992 | .229 | +.027 [+.018, +.035] / +.013 [−.002, +.028] | .809 (+.017 [+.004, +.030]) | .971 / .905 |
| **L9IBX** | .9381 | 800 / 364 / 337 | 364 | .5920 | −.022 [−.037, −.007] FLAG | .320 (+.058 [+.039, +.077]) | .996 | .271 | +.023 [+.014, +.031] / +.013 [−.004, +.030] | .791 (−.001 [−.014, +.011]) | .964 / .796 |

- Score5-typed-DEV check half: clean for both. Family floors: L9IB `transition_table` 370 / 400 vs C0 399 (within
  the 0.10 floor); PN1 PAWS-X-6 yes-rate Δ +.055 (L9IB) / +.027 (L9IBX), all-8 +.059 / +.032 (report only).
- Retention per probe: L9IB MMLU .747 / ARC .960 / GSM8K .720; L9IBX .739 / .962 / .671 (C0 .776 / .966 / .635).

### Contrasts (report only)

| Contrast | typed T Δ (C / N / S) | HT-DEV v2 Δ | retention Δ | PN1 clean gold-no Δ |
| --- | --- | --- | --- | --- |
| L9IB − L9 | −.049 (−19 / +10 / −69) | +.011 [−.003, +.024] TIE | −.005 [−.017, +.006] | +.074 [+.055, +.094] |
| L9IBX − L9 | +.001 (+11 / +34 / −43) | +.009 [−.003, +.021] TIE | −.024 [−.035, −.013] | +.015 [.000, +.031] |
| L9IB − L9IBX | −.050 (−30 / −24 / −26) | +.001 [−.010, +.014] TIE | +.018 [+.005, +.030] | +.059 [+.041, +.078] |

### IB DEV per family (accuracy; Δ vs C0 on the slice with a paired bootstrap, 2,000 draws)

| Slice | C0 | L9 | L9IB | L9IBX |
| --- | --- | --- | --- | --- |
| IB1 DEV (2,067) | .924 | .921 (−.003 [−.010, +.004]) | .971 (+.048 [+.038, +.059]) | .964 (+.040 [+.031, +.050]) |
| `args` / `copa` / `poem` / `sentfin` | .874 / .944 / .929 / .945 | .869 / .954 / .932 / .939 | .941 / .963 / .998 / .970 | .941 / .963 / .998 / .968 |
| `sms` / `snips_rel` / `snips_sel` / `isarc` | .823 / 1.0 / .990 / .720 | .790 / .917 / .994 / .720 | .935 / 1.0 / 1.0 / .915 | .935 / 1.0 / 1.0 / .744 |
| IB2 DEV (1,272) | .809 | .807 (−.002 [−.020, +.015]) | .905 (+.096 [+.074, +.117]) | .796 (−.013 [−.031, +.004]) |
| `argq` / `ytspam` / `gsm2` / `hover` | .995 / .935 / .670 / .770 | .980 / .924 / .689 / .735 | .998 / .957 / .830 / .905 | .998 / .967 / .657 / .700 |

## Reading

- **Data alone does not move the from-base adapter past the 9B constraints.** The IB blocks teach their families (the
  transfer-only families reach .94–1.0 on DEV), but human transfer stays where the base start put it, and the
  balanced IB Noul families do not cut the paraphrase yes-bias. They raise it, mostly through the in-distribution
  `isarc` / `hover` / `gsm2` / `w2c` families.
- L9IBX is the strongest typed point of M9 so far (T .938, Noul `rule_precedence` 364 vs C0 338) and level on MLX-DEV,
  but it fails Y1 and Y3: the extra Noul confidence comes with more "yes" on near-misses and unmet conditions.
- **Stage 3** tests the same IB blocks on the released K-a13 recipe (Lux start, full fine-tuning, ⅔ interpolation back
  to Lux), where the interpolation is the trust region that kept K-a13's yes-bias at .262.

## Resources

Stage 2 ≈ 13.0 GPU-h: L9IB 6.27 and L9IBX 6.05 (node C, preflights included), merges 0.09, node-A readouts ≈ 0.6.
M9 total ≈ 34.8 of 120 at 13:35Z (stage 3 running).

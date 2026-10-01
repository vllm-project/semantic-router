# 9B Milestone 9, stage 1 result: the 4B from-base LoRA recipe gives no 9B finalist

Run under the [preregistration](lux9b-m9-prereg-2026-10-01.md) (`8570a5896`), [amendment 1](lux9b-m9-prereg-amendment-1-2026-10-01.md)
(`0b84e0db4`, B0 read through the base's untied LM head) and [amendment 2](lux9b-m9-prereg-amendment-2-2026-10-01.md)
(`51c80ddc9`, stage 2), each pushed before the GPU jobs it governs. The released `llm-semantic-router/DEV2.0-9B`
(K-a13 at T = 1) stays the 9B model. Development readouts are never release scores. Nothing was uploaded; C1 was not
opened; no Index row was read. Stage 2 (IB1-r3 + IB2 on the L9 recipe) is still training and is reported separately.

## Verdict

- **No stage-1 finalist, so no stage-1 formal run** (rules `select/9b-finalists.json` `7b43dad3…`; the formal chain
  stopped by rule).
  - **L9** (rank-128 LoRA from Qwen3.5-9B-Base + fresh candidate head) fails three gates: the Noul
    `rule_precedence` floor (330 < 338 − 4), **HT-DEV v2 FLAG** (−.031 [−.046, −.018]) and the yes-bias guard Y1
    (PN1-dev clean gold-no yes-rate +.042 [+.023, +.060]).
  - **L9L** (the same LoRA on Lux 1.0, the lineage control) fails only Y1: clean gold-no +.086 [+.066, +.106]. Every
    other gate passes (typed and family floors, Noul and Score floors, HT-DEV v2 TIE −.006, MLX-DEV-9B, retention).
- **The 4B result does not carry to 9B.** At 4B the released lineage had lost knowledge against its base and the
  from-base adapter recovered it while keeping human transfer. At 9B:
  - the released model already retains at least the base's knowledge on the probes (C0 .792 vs base .781);
  - the from-base adapter **loses human transfer** to held-out decision families (HT-DEV v2 FLAG vs C0, and −.025
    [−.040, −.012] vs the Lux-start adapter at the same data and adapter);
  - both adapters carry **more near-miss paraphrase yes-bias than K-a13**, the 9B binding constraint since M5.
- Hypotheses (preregistered readings): **H1 retention: supported by the rule, with a caveat** (L9 macro +.022
  [+.011, +.034] vs C0, but the gain is GSM8K +.089 while MMLU falls −.022 [−.037, −.009]; x60 holds GSM8K-derived
  rows, so this is maths in distribution rather than kept base knowledge). **H2 transfer: not supported.** **H3
  lineage:** the base start transfers worse and is less yes-biased than the Lux start.

## Design as run

| Arm | Start | Seeds' BEST (SELECT700 family macro) | Artifact (FP32, merged) |
| --- | --- | --- | --- |
| C0 | DEV2.0-9B = K-a13 soup (`m4/K-a13-build/soup`) | — | reference |
| L9 | Qwen3.5-9B-Base `68c46c4b…`; LoRA r 128 / α 256 / dropout .05 (all text projections), LR 1e-4; fresh head (init seed 20261001), LR 1e-4 | s1 1,624 (.8766), s2 1,621 (.8689) | uniform soup, identity `56237917…` |
| L9L | Lux 1.0 `bd45a30a…` (the K recipe's package); the same LoRA; Lux's head continued at 1e-4 | s1 1,624 (.8646), s2 1,216 (.8763) | uniform soup, identity `baf566e1…` |
| B0 | Qwen3.5-9B-Base, untrained, label-token prompt read through its own LM head (amendment 1) | — | retention ceiling only |

- Data and objective = K-a13's: x60 `a66131b1…` (122,651 rows, 60.2M tokens), own-Lux KL 1.0 on every row (`cdcd99c1…`),
  CE + 0.5·Brier, token batching ≤ 32,768 / ≤ 64 rows, 64 rows per update (1,621–1,624 updates), one epoch, max length
  8,192, seeds 20260926 / 20260927. Node C GPU1–4, image `f83b1d10…`; L9-s1's preflights ran alone first (pre-warm);
  all four preflights passed; ≈ 2.6 h per seed.
- Merges agree with the adapters on 128 / 128 SELECT rows (max probability drift ≤ .013).
- B0's preregistered label-token zero-step stopped at model build (untied LM head at 9B); amendment 1 read it with a
  9B-local reader instead.

## Development readouts (node A, the M9 path: `v2.dec.infer_dec` 16K, T = 1; MLX-DEV-9B via `eval_rows`)

C0's M9-path answers equal M7's stored K-a13 readouts exactly (typed DEV, CSS pilot, HT-DEV v2, PN1 dev).

| Point | typed T | C / N / S | RP | H3 | HT-DEV v2 Δ vs C0 | PN1 clean gold-no (Δ) | PN1 PAWS-X-6 yes | hop | `hs1-dev` false-yes | MLX-DEV-9B Noul / Choice / Score Δ |
| --- | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | --- |
| C0 | .9250 | 799 / 338 / 343 | 338 | .5622 | — | .262 | .626 | .987 | .152 | — |
| L9 | .9369 | 789 / 330 / 380 | 330 | .5841 | −.031 [−.046, −.018] FLAG | .305 (+.042 [+.023, +.060]) | .643 | .987 | .140 | +.025 [+.017, +.033] / +.007 [−.007, +.023] / +.028 |
| L9L | .9287 | 784 / 336 / 366 | 336 | .5558 | −.006 [−.019, +.006] TIE | .348 (+.086 [+.066, +.106]) | .667 | .987 | .152 | +.021 [+.013, +.030] / +.007 [−.011, +.024] / +.029 |
| Lux 1.0 (report) | .8762 | 799 / 272 / 331 | 272 | .5283 | −.003 TIE | .187 (−.075) | .585 | .962 | .128 | −.045 / −.074 / −.049 |

Score5-typed-DEV check half: clean for every point. Family floors: L9 `transition_table` 389 / 400 and L9L 384 / 400
vs C0 399 (within the 0.10 floor); `set_reconciliation` +37 / +23.

### Retention probes (M9 panel: the 4B M10 panel minus 250 GSM8K items overlapping x60; 2,839 items)

| Point | MMLU (1,265) | ARC (824) | GSM8K (750) | Macro | Δ macro vs C0 [95% CI] |
| --- | ---: | ---: | ---: | ---: | --- |
| B0 = Qwen3.5-9B-Base | .760 | .954 | .628 | .781 | −.011 [−.027, +.004] |
| Lux 1.0 | .776 | .966 | .537 | .760 | −.032 [−.044, −.020] |
| **C0 = DEV2.0-9B** | .776 | .966 | .635 | **.792** | — |
| L9 | .753 | .966 | .724 | .814 | +.022 [+.011, +.034] |
| L9L | .760 | .970 | .691 | .807 | +.015 [+.004, +.026] |

Lux 1.0 lost GSM8K against the base (−.091) and K-a13's x60 training restored it; neither adapter keeps MMLU at C0's
level.

### Gates (`m9_rules.py`; Y1 margin +.02 with CI lower ≤ 0)

| Gate | L9 | L9L |
| --- | --- | --- |
| 1 typed type / family floors | pass | pass |
| 2 Noul `rule_precedence` ≥ 334 | **fail (330)** | pass (336) |
| 3 Score5-typed-DEV | pass | pass |
| 4 HT-DEV v2 not FLAG | **fail (−.031)** | pass (−.006) |
| 5 yes-bias Y1 / Y2 / Y3 | **Y1 fail** / pass / pass | **Y1 fail** / pass / pass |
| 6 MLX-DEV-9B upper ≥ 0 | pass | pass |
| 7 retention upper ≥ 0 | pass | pass |

## Reading

- **Why not an α line on L9L toward Lux?** It is the closest arm (one failing gate), and moving toward Lux would cut
  its yes-bias. But its typed gain over C0 is only +.004 and its HT-DEV v2 is −.006; a successor needs about +2 v3
  over K-a13 (K5-a12's development +.030 gave formal +1.62 [−0.19, +2.41]). An interpolation that passes Y1 would give
  back even that typed gain, so it cannot be expected to pass item 1. Not run (it would also need an amendment).
- **MLX-DEV-9B and PN1 disagree again**, as in M8: both adapters read as better than C0 multilingually (Noul-ML +.02
  to +.03) while PN1's paraphrase construction reads them as more yes-biased. The guard keeps the PN1 reading, which
  is the one that predicted K5-a12's item-4 failure.
- **Training method is not the 9B lever on its own.** At equal data and objective, replacing the trust-region full
  fine-tune (interpolated ⅔ back to Lux) by a rank-128 adapter loses held-out transfer from the base and does not cut
  the yes-bias from Lux. The remaining 9B levers are data that reduce the paraphrase yes-bias without hurting
  `hop` (PN1-r2 at a much lower dose, or IB's balanced Noul families — stage 2 measures the latter), or keeping the
  ⅔-Lux interpolation structure with new data.

## Stage 2 and remaining work

Stage 2 (amendment 2: L9IB = x60 + IB1-r3 + IB2, L9IBX without `isarc` / `w2c` / `hover` / `gsm2`; two seeds each on
the L9 recipe) trains on node C until ≈ 13:00Z; its post chains read it on the same path and write
`select/9b-finalists-s2.json`; its formal chain waits for `status/formal-s2.GO` (see `m9-state.md`). The IB TRAIN
exposure receipt against the K-a13 payload lists 0 groups. No stage-1 model goes to C1, release or the private Index.

## Resources

Stage 1 ≈ 12.1 GPU-h: L9 5.38 and L9L 5.42 (node C, preflights included), merges 0.08, B0 0.04, node-A readouts of
C0 / Lux 1.0 / L9 / L9L ≈ 1.03, the formal-path parity run of DEV2.0-9B 0.20 (CAL fit + smoke + panels + mlx-diag).
Stage 2 had used ≈ 3.8 GPU-h at 10:15Z. Cap 120.

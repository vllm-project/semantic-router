# 9B Milestone 9, stage 4 result: the ½ point K-a12IB fails the yes-bias guard (Y1); no formal run

Stage 4 = [amendment 4](lux9b-m9-prereg-amendment-4-2026-10-01.md) (`78b01441e`, pushed before the build). Chain
`m9/post-a4.sh` on node A GPU6 (mirror `78b01441e`), 16:44–17:09Z. Development readouts only; they are never release
scores.

## Outcome

- **K-a12IB = [KIB soup, Lux 1.0]** (α ½; built 16:46Z, `model_sha256` `68fed4cb…`) raises typed DEV T to **.950**
  (C0 .925, K-a13IB .9344) and keeps HT-DEV v2, MLX-DEV-9B and retention inside their gates.
- **It fails one gate, Y1:** the PN1-dev clean gold-no yes-rate rises +.018 [+.002, +.032] vs C0. The point estimate is
  inside +.02, but the CI lower bound is above 0.
- **No stage-4 finalist** (`select/9b-finalists-s4.json` `4cc03edd…`, typed readout `lines/readout/m9-s4.json`
  `86b9e486…`). So there is no formal run, no items 1–7, no item 8 and no release hand-off. The gates ran once.

## Gates vs C0 (`m9_rules.py` unchanged)

| Gate | C0 | K-a12IB | Verdict |
| --- | --- | --- | --- |
| typed T; C / N / S | .925; 799 / 338 / 343 | **.950**; 800 / 353 / 367 | pass |
| Noul `rule_precedence` (floor C0 − 4) | 338 | 353 | pass |
| Score5-typed-DEV check | — | no flags | pass |
| HT-DEV v2 Δ | — | −.018 [−.030, −.006] TIE | pass |
| **Y1** PN1 clean gold-no yes Δ (≤ +.02, CI lower ≤ 0) | .262 | .280, **+.018 [+.002, +.032]** | **fail** |
| Y2 hop Δ (≥ −.03) | .987 | .987, .000 | pass |
| Y3 `hs1-dev` false-yes (≤ +.05) | .152 | .161 | pass |
| MLX-DEV-9B Noul / Choice / Score Δ (upper ≥ 0) | — | +.011 [+.004, +.018] / +.004 [−.013, +.021] / +.012 [−.002, +.025] | pass |
| retention Δ (upper ≥ 0) | .792 | .788, −.004 [−.015, +.008] | pass |

CSS pilot H (mean of three tasks): C0 .562, K-a12IB .570. PN1 PAWS-X-6 / all-8 yes-rate Δ vs C0: +.008 / +.010.

## Report only

**IB DEV** (micro accuracy):

| Slice | C0 | K-a13IB (⅓) | **K-a12IB (½)** | KIB soup (α 1) |
| --- | --- | --- | --- | --- |
| IB1 DEV (2,067) | .924 | .963 | **.970** | .971 |
| IB2 DEV (1,272) | .809 | .890 | **.898** | .910 |

**Contrast K-a12IB − K-a13IB:**

- typed T +.016 (C / N / S 0 / +9 / +16);
- HT-DEV v2 −.012 [−.022, −.002] TIE;
- retention −.004 [−.012, +.004];
- **PN1 clean gold-no +.033 [+.021, +.046]**.

## Reading

- **H7 (K-a12IB passes the seven gates): fails, on Y1 only.** Moving from ⅓ to ½ toward the KIB soup brings back the
  near-miss yes-bias that ⅓ had removed: K-a13IB −.015 vs C0, K-a12IB +.018 vs C0, so +.033 between them. The IB block
  lowers the bias, but not enough to offset the extra distance from Lux. This is the K5-a12 pattern (α ½ failed item 4
  formally) seen one step earlier, on the development guard.
- **H8 (typed DEV above K-a13IB): holds.** T .950 vs .9344 (Noul +9, Score +16). Most of the IB DEV gain is already
  present at ⅓.
- **For the 9B lever:** α ½ trades the yes-bias guard for typed DEV gain, so interpolation alone does not give a 9B
  successor. The remaining levers in the stage-3 record need training: a five-seed KIB soup at ⅓, or typed-row
  self-distillation with IB. Any α between ⅓ and ½ would be an α line that this amendment excluded, and choosing it
  after seeing these readouts would be post hoc.

## Resources

Node A GPU6, about 0.36 GPU-h (eight panels plus IB DEV, 16:47–17:08Z; the build ran on CPU). M9 total ≈ 41.9 of 120
GPU-h. The node A GPU6 lease was released at 17:10Z.

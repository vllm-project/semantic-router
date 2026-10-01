# Decoder Milestone 15 — results (multilingual-preserving breadth + MLX-DEV guard; 2026-10-01)

Prereg `dec-m15-prereg-2026-10-01.md` (`c62a78619`), data lock `dec-m15-data-lock-2026-10-01.md` (`30ec83646`), ops
`ops/m15/` (`6abb5358e`), running log `dec-m15-state.md`. Development readouts are never release scores; Index numbers
are private and appear nowhere in this repository. Evidence (rules outputs, MLX-DEV compares, data and panel
reports): `dec-m15-results-2026-10-01/`.

## Outcome

- **No M15 point passes its tier's development gates** (rules run once per tier), so there is no formal run, no
  successor evaluation and no hand-off. 4B fails on retention only; 2B on the Score type floor; 0.8B on the choice type
  floor and the `attribute_gate` family floor (and, for the full-dose arm, the new MLX-DEV guard).
- **Keeping the released multilingual token share did not remove the typed-head losses**, and it was not neutral:
  vs the M13 SD arm it modifies, each ML arm moved typed items between heads (4B choice +34 / Noul −15; 2B Score −63 /
  choice +53; 0.8B choice −34 / Score −26) while breadth stayed (transfer +.048 / +.152 / +.192).
- **The MLX-DEV guard does not see the known mlx-diag losses.** Read on the two IB points whose formal mlx-diag loss is
  known (report only), MLX-DEV-M15 shows no loss: M13 `4b-LHA10SD` vs LH Noul-ML +.009 / Choice-ML −.006 (formal
  mlx-diag −.029); M12 `08b-RA` vs DEV2.0-0.8B +.006 / .000 (fast track −.014). The guard caught one M15 point
  (`08b-RASDML`), which also failed three typed floors. A development multilingual guard that tracks mlx-diag needs a
  different panel (the mlx-diag construction itself, held out), not MLX-DEV.
- Part A (private Index diagnostic of the frozen `4b-LHA10SD`, node D GPU7): parity gate PASS (86 requests), the full
  panel collected (8 of 8 shards, 120,226 rows; 2 unsupported, as the LH run) and dual-scored (scorers pass); the
  numbers went to the coordinator privately.

## Data (data lock)

ML-preserving upsampling restored each tier's released multilingual token share exactly (4B .4142, 2B .4547, 0.8B
.4327; built within .0001) by adding one copy of whole non-English released groups, stratified by language: 4B 2.05M
tokens (+17% of ML), 2B 6.01M (+45%), 0.8B 23.0M (+38%); 10% arms 2.40M (2B) and 10.4M (0.8B). SD targets were extended
to every copy. MLX-DEV-M15: 4B and 0.8B = the whole decoder MLX-DEV panel (9,386 rows / 3,802 groups; no overlap); 2B
7,742 / 3,395 (407 groups shared with the 2B released TRAIN dropped).

## Development gates (vs the tier reference read on the same node; no waivers)

| Point (reference typed C / N / S) | Typed C / N / S | HT-DEV v2 | Retention Δ [95% CI] | Transfer Δ | MLX-DEV Noul-ML / Choice-ML Δ [95% CI] | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 4B `4b-LHA10SDML` (LH 728 / 290 / 371) | 742 / 292 / 370 | −.008 TIE | **−.030 [−.054, −.006]** | +.048 | +.016 [+.009, +.023] / .000 [−.015, +.015] | retention |
| 2B `2b-RASDML` (DEV2.0-2B 489 / 230 / 257) | 532 / 244 / **174** | +.041 GAIN | +.010 [−.016, +.038] | +.152 | +.001 [−.007, +.008] / −.009 [−.024, +.007] | Score floor (< 245); `set_reconciliation` |
| 2B `2b-RA10SDML` | 472 / 234 / **235** | +.011 TIE | +.009 [−.019, +.040] | +.147 | +.003 [−.004, +.011] / −.004 [−.016, +.009] | Score floor (< 245) |
| 0.8B `08b-RASDML` (DEV2.0-0.8B 610 / 212 / 159) | **540** / 201 / 179 | +.047 GAIN | +.033 [+.008, +.060] | +.192 | **−.012 [−.021, −.003] / −.030 [−.057, −.003]** | choice (< 586); `attribute_gate` 271; `rule_precedence` 201; MLX-DEV guard |
| 0.8B `08b-RA10SDML` | **557** / 220 / 238 | +.030 GAIN | −.016 [−.034, +.001] | +.193 | −.006 [−.016, +.003] / +.006 [−.019, +.029] | choice (< 586); `attribute_gate` 276 (< 280) |

hs1-dev false-yes (guard ≤ reference + .10): 4B .211 vs .199; 2B .607 vs .625 (both arms); 0.8B .774 / .726 vs .720.

Report-only contrasts: `2b-RASDML` vs M13 `2b-RASD` HT +.024 GAIN; `2b-RA10SDML` vs `2b-RASDML` HT −.030 FLAG, Score +61;
`08b-RA10SDML` vs `08b-RASDML` retention −.049 [−.074, −.024], typed choice +17 / Noul +19 / Score +59; `4b-LHA10SDML` vs
M13 `4b-LHA10SD` HT −.005 TIE, retention −.008.

## GPU-hours (launch receipts: `m15_gpuh.py table` on E and F; Part A from the harness receipt)

| Node | Arms | Readouts / merges | Total |
| --- | --- | --- | ---: |
| E | `08b-RASDML` 3.21, `08b-RA10SDML` 2.72 | 0.68 | 6.61 |
| F | `4b-LHA10SDML` 2.29, `2b-RASDML` 1.17, `2b-RA10SDML` 0.97 | 1.74 | 6.17 |
| D (Part A) | — | IX1 parity 0.06 + 8 shards 2.30 | 2.36 |
| **M15** | | | **15.14 of 70** |

Every arm stayed within its cap (4B 5.0, 2B 2.5, 0.8B 5.5 GPU-h).

## Incidents

- Part A's first launch was refused by the harness's lease check (node D GPU7's owner file was an older one-line form);
  no container ran. Its write-once `launcher-parity.json` stub was moved to the private `void/`, the owner file was
  rewritten in the multi-line form (still `track=eval-ix1`) and the run launched once.
- The 0.8B 10% arm's post chain waited ≈ 20 min for the 0.8B reference's MLX-DEV read (done by the full-dose arm's
  chain); no effect on results.

## For the coordinator

- Breadth vs multilingual: at 4B the trade-off is now visible in Part A (private) for `4b-LHA10SD`; the development
  gates cannot arbitrate it, because MLX-DEV does not track mlx-diag. Any further IB arm needs either a held-out
  mlx-diag-like development panel or a formal mlx-diag read before selection.
- The typed-head losses remain the binding constraint at 2B (Score) and 0.8B (choice / `attribute_gate`); 0.8B
  `08b-RA10SDML` is the nearest (choice −29 vs its floor, `attribute_gate` −4).

# Decoder M15 — state (prereg `dec-m15-prereg-2026-10-01.md`)

## 2026-10-01 18:55Z — 0.8B rules (run once): no finalist; M15 has no finalist at any tier

| Point (vs `08b-C0-e`: typed C / N / S 610 / 212 / 159) | Typed T; C / N / S | HT-DEV v2 | Retention Δ | Transfer Δ | hs1 false-yes | MLX-DEV Noul-ML / Choice-ML Δ [95% CI] | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `08b-RASDML` | .575 vs .613; 540 / 201 / 179 | +.047 [+.025, +.066] GAIN | +.033 [+.008, +.060] | +.192 | .774 vs .720 | **−.012 [−.021, −.003] / −.030 [−.057, −.003]** | choice 540 < 586; `attribute_gate` 271 < 280; `rule_precedence` 201 < 208; **MLX-DEV guard (both)** |
| `08b-RA10SDML` | .634; 557 / 220 / 238 | +.030 [+.010, +.049] GAIN | −.016 [−.034, +.001] | +.193 | .726 | −.006 [−.016, +.003] / +.006 [−.019, +.029] (passes) | choice 557 < 586; `attribute_gate` 276 < 280 |

- Contrasts (report only): `08b-RASDML` vs M13 `08b-RASD`: HT +.006 TIE; typed choice −34, Noul −7, Score −26.
  `08b-RA10SDML` vs `08b-RASDML`: HT −.017 TIE; retention −.049 [−.074, −.024]; typed choice +17, Noul +19, Score +59.
- **Guard validation, 0.8B (report only):** M12's `08b-RA` vs `08b-C0-e`: Noul-ML +.006, Choice-ML .000, M_dev .000 —
  as at 4B, MLX-DEV does not reproduce that model's formal mlx-diag loss (−.014).
- No formal, no hand-off. GPU-h (launch receipts): node E 6.61, node F 6.17; Part A pending (≈ 2.4).

## 2026-10-01 18:31Z — 4B rules (run once): no finalist

| Point (vs `4b-LH-f`: typed C / N / S 728 / 290 / 371) | Typed T; C / N / S | HT-DEV v2 | Retention Δ | Transfer Δ | hs1 false-yes | MLX-DEV Noul-ML / Choice-ML Δ [95% CI] | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `4b-LHA10SDML` | .878 vs .868; 742 / 292 / 370 | −.008 [−.022, +.005] TIE | **−.030 [−.054, −.006]** | +.048 | .211 vs .199 | +.016 [+.009, +.023] / .000 [−.015, +.015] (guard passes) | retention CI upper < 0 |

- Contrast vs M13 `4b-LHA10SD` (report only): HT −.005 TIE; retention −.008 [−.032, +.015]; typed choice +34, Noul −15,
  Score −2. MLX-DEV: M13's point +.009 / −.006 vs LH, this point +.016 / .000.
- 0.8B: `08b-RA10SDML` s1 / s2 DONE 18:07 / 18:15Z; `08b-RASDML` seeds finishing.

## 2026-10-01 18:17Z — 2B rules (run once): no finalist

| Point (vs `2b-C0-f`: typed C / N / S 489 / 230 / 257) | Typed C / N / S | HT-DEV v2 | Retention Δ | Transfer Δ | MLX-DEV Noul-ML / Choice-ML Δ [95% CI] | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| `2b-RASDML` | 532 / 244 / **174** | +.041 [+.025, +.057] GAIN | +.010 [−.016, +.038] | +.152 | +.001 [−.007, +.008] / −.009 [−.024, +.007] (guard passes) | Score floor 174 < 245; `set_reconciliation` floor |
| `2b-RA10SDML` | 472 / 234 / **235** | +.011 [−.004, +.026] TIE | +.009 [−.019, +.040] | +.147 | +.003 [−.004, +.011] / −.004 [−.016, +.009] (guard passes) | Score floor 235 < 245 |

- Contrasts (report only): `2b-RASDML` vs M13 `2b-RASD`: HT +.024 [+.008, +.040] GAIN; typed Δ choice +53, Noul +11,
  Score −63. `2b-RA10SDML` vs `2b-RASDML`: HT −.030 FLAG; Score +61, choice −60. Both pass the MLX-DEV guard; the Score
  head decides (as in M12 / M13).
- **Guard validation, 4B (report only):** M13's `4b-LHA10SD` vs `4b-LH-f` on MLX-DEV-M15: Noul-ML +.009, Choice-ML
  −.006, M_dev +.010 — the development panel does **not** reproduce that model's formal mlx-diag loss (−.029).

## 2026-10-01 18:05Z

- Node F training finished: `2b-RA10SDML` s1 / s2 DONE 17:50 / 17:54Z, `4b-LHA10SDML` s1 / s2 DONE 17:54 / 17:59Z. Both
  2B soups built (17:57Z) and being read; reference MLX-DEV-M15 reads `2b-C0-f` and `4b-LH-f` done (exit 0); the 4B post
  chain is on the guard-validation read, then the merge / soup. 0.8B seeds training on E.
- Part A: shards 0–4 done, shard 5 running.

## 2026-10-01 17:40Z

- `2b-RASDML` s1 / s2 DONE (17:21 / 17:25Z); `2b-RA10SDML` s1 / s2 training (F GPU2 / 3). 4B and 0.8B seeds training.
- Part A: shards 0–2 done, shard 3 running.

## 2026-10-01 17:16Z

- Every preflight PASS (pre-warms 16:49–16:53Z); all eight seeds training (2B near the end of its cosine schedule; 4B
  ≈ 6 of 8 checkpoints; 0.8B ≈ 2–4). Post chains queued on their GPUs' flocks.
- Part A: shards 0 and 1 done (exit 0, ≈ 17 min each), shard 2 running; private.

## 2026-10-01 16:50Z

- Timeline: prereg `c62a78619` (committed ≈16:22Z; its header's "≈16:40Z" is a slip, nothing M15 had run); IX1
  diagnostic entry `cdaae0f32`; ops `6abb5358e` (13 CPU tests, with M13's); data lock `30ec83646`
  (`dec-m15-data-lock-2026-10-01.md`: five arms byte-identical on E and F, multilingual shares equal the released
  shares; MLX-DEV-M15 panels 4B / 0.8B whole, 2B 7,742 rows).
- **Part A** (node D GPU7, harness lease `eval-ix1`): restaged onto `13d42143` (12 model files replaced); the first
  launch was refused by the harness's lease check (GPU7's owner file was an older one-line form; no container ran)
  and its write-once `launcher-parity.json` stub was moved to the private `void/`; the owner file was rewritten in the
  harness's multi-line form (still `track=eval-ix1`) and the run launched once: **parity gate PASS (86 requests)**
  16:33Z; panel-8 shards running one after another. Results stay private.
- **Part B**: chains launched 16:45Z from mirror `6abb5358e`; pre-warm seeds `08b-RA10SDML` s1 (E GPU0), `2b-RASDML`
  s1 (F GPU2), `4b-LHA10SDML` s1 (F GPU6); the other seeds wait for their tier's marker. Post chains queued: E GPU3
  `08b-RASDML` (+ `08b-C0-e` and `08b-RA-m12` MLX-DEV), E GPU1 `08b-RA10SDML`, F GPU7 `4b-LHA10SDML` (+ `4b-LH-f` and
  `4b-LHA10SD-m13`), F GPU6 `2b-RASDML` (+ `2b-C0-f`), F GPU3 `2b-RA10SDML`.
- M14 untouched (no M14 container on E / F GPUs). GPU-h so far: Part A parity ≈ 0.06.

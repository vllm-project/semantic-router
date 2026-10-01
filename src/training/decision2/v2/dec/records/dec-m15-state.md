# Decoder M15 — state (prereg `dec-m15-prereg-2026-10-01.md`)

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

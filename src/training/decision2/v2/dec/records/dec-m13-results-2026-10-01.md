# Decoder Milestone 13 — results (typed-head protection; 2026-10-01)

Prereg [`dec-m13-prereg-2026-10-01.md`](dec-m13-prereg-2026-10-01.md) (`c2610143b`), data lock
[`dec-m13-datalock-2026-10-01.md`](dec-m13-datalock-2026-10-01.md) (`debd36e1a`), amendments
[1](dec-m13-amendment-1-2026-10-01.md) (formal entries with absolute paths), [2](dec-m13-amendment-2-2026-10-01.md)
(two-bar formal scoring, mlx-diag step) and [3](dec-m13-amendment-3-2026-10-01.md) (mlx-diag from the collecting
mirror). State log: [`m13-state.md`](m13-state.md). Development readouts are never release scores; v3 / public 231 /
mlx-diag are post-key same-panel comparisons. No Index row was read.

## Summary

- **No M13 model passes successor items 1–7; nothing is handed off.** LH, DEV2.0-2B and DEV2.0-0.8B stay the releases.
- **4B `4b-LHA10SD`** (LH + 10% IB additive + typed-row self-distillation from LH) passed every development gate.
  Self-distillation removed M12 `LHA10`'s typed-head regression (Score floor) and kept the breadth gain. **Formally
  it ties LH** (v3 67.19 vs 67.34, −0.16 [−2.84, +3.64]; item 1 fails) and **loses multilingual decisions**
  (mlx-diag card-eligible −.029 [−.039, −.018]; item 4 fails).
- **2B `2b-RASD`** fails the Score type floor (237 < 245); **0.8B `08b-RASD` / `08b-RAAG`** fail the choice type floor
  (574 / 498 < 586; RAAG also `attribute_gate` 228 < 280). No exceptions (COORDINATION 23:15).
- **GPU-h 14.21 of the 80 cap.**

## Development gates (rules run once per tier)

| Tier / arm (reference) | Typed T; C / N / S (ref) | HT-DEV v2 | Retention macro Δ | IB transfer Δ | hs1 false-yes | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 4B `4b-LHA10SD` (LH) | .867 vs .868; 708 / 307 / 372 (728 / 290 / 371) | −.003 [−.018, +.012] TIE | −.022 [−.047, +.002] | +.046 [+.034, +.058] | .202 vs .199 | **finalist** |
| 4B `4b-LHA5` (LH) | .912; 800 / 266 / 393 | +.019 [+.004, +.034] TIE | −.033 [−.057, −.010] | +.044 [+.031, +.057] | .256 | Noul floor 266 < 278, `rule_precedence`, retention |
| 2B `2b-RASD` (DEV2.0-2B) | .593 vs .610; 479 / 233 / 237 (489 / 230 / 257) | +.017 [−.001, +.034] TIE | +.008 [−.018, +.035] | +.155 [+.126, +.186] | .494 vs .625 | Score floor 237 < 245 |
| 0.8B `08b-RASD` (DEV2.0-0.8B) | .617 vs .613; 574 / 208 / 205 (610 / 212 / 159) | +.040 [+.019, +.060] GAIN | +.026 [+.001, +.054] | +.201 [+.165, +.237] | .783 vs .720 | choice floor 574 < 586 |
| 0.8B `08b-RAAG` (DEV2.0-0.8B) | .547; 498 / 216 / 162 | +.038 [+.016, +.058] GAIN | +.009 [−.018, +.037] | +.194 [+.157, +.231] | .798 vs .720 | choice floor 498 < 586; `attribute_gate` 228 < 280 |

Report-only contrasts against the M12 arm each M13 arm modifies:

- `4b-LHA10SD` vs `LHA10`: HT −.007 [−.020, +.007]; retention +.001 [−.027, +.030]; Score 372 vs 329.
- `4b-LHA5` vs `LHA10`: HT +.016 [+.001, +.031]; retention −.010.
- `2b-RASD` vs `2b-RA`: HT +.020 [+.006, +.034] GAIN; retention +.018 [−.007, +.044]; Score 237 vs 227, choice 479
  vs 558. Self-distillation moved the loss from the Score head to the choice head without clearing the Score floor.
- `08b-RASD` vs `08b-RA`: HT −.009 [−.027, +.009]; retention −.018 [−.048, +.012]; `attribute_gate` 293 vs 277
  (now above its floor), choice 574 vs 600 (now below).
- `08b-RAAG` vs `08b-RA`: HT −.012 [−.030, +.008]; retention −.035 [−.062, −.007]. Upweighting the proxy families
  ×3 lowered `attribute_gate` itself (228) and choice.

## 4B formal (node F GPU6 / 7, M6 library, image `dbe5f32b`; scored on node A)

The parity run `m13-4b-LH` (the LH soup on the M13 path) reproduces the released LH's stored formal run exactly (0
answer changes on typed FINAL, CSS15 and public 231; v3 67.345), so both bars give identical verdicts. Evidence:
[`dec-m13-results-2026-10-01/`](dec-m13-results-2026-10-01/).

| Item (vs the released LH, v3 67.345) | `m13-4b-LHA10SD` |
| --- | --- |
| 1 v3 paired lower bound > 0 | **FAIL** v3 67.187, −0.158 [−2.836, +3.636] |
| 2 H upper bound ≥ 0 | PASS [−.036, +.062] |
| 3 types | PASS (choice / Noul / Score OK) |
| 4 mlx-diag card-eligible upper bound ≥ 0 | **FAIL** −.0287 [−.0394, −.0184] (full panel −.0215 [−.0304, −.0129]) |
| 5 tier gates (vs adopted Nox 1.0, peers) | PASS +10.72 [+6.53, +13.66] |
| 6(a) exposure of the TRAIN (72,847 rows, `d41cdd1a…`) | PASS 0 groups |
| 6(b) reduced panels | not evaluated: the overlap spec asked to reproduce a pair outside the tier set (`n4xf-ref`); the step stopped and was not rerun (scorer fixed afterwards) |
| 7 public 231 | PASS 174 vs 172 (OK) |
| Report only: vs DEV2.0-4B | +4.04 [+1.57, +9.64] |

## Reading

- **Typed-row self-distillation protects the typed heads on development data at 4B** (Score 329 → 372 with the same
  IB dose), but it does not make the IB additive a formal improvement over LH. The breadth gain (IB transfer +.046)
  does not show up in v3, and the multilingual decisions regress (mlx-diag −.029).
- This matches the 0.8B fast track's `08b-RA` (failed item 4, mlx-diag −.014): **IB additives cost multilingual
  decisions at 4B and 0.8B, and no development gate screens for it.** Before more IB-additive arms, add a development
  multilingual guard (an mlx-diag-style slice of development data) to the rules.
- At 2B and 0.8B, self-distillation moves the typed loss between heads (2B: Score → choice; 0.8B: `attribute_gate`
  → choice) instead of removing it. Family upweighting at 0.8B hurt the targeted family.

## Hand-offs

- **None: no model passes items 1–7.** No package, no item-8 spec, no Index request.
- C1 content recheck (eval custodian, worker 3c7679b0): registered, not yet run,
  [`v2/eval/records/c1-recheck-r1-2026-10-01.md`](../../eval/records/c1-recheck-r1-2026-10-01.md) (branch
  `decision-2-eval-c1recheck`, `15ba0ac0d`). It covers IB1-r3 + IB2; no M13 item 8 depends on it.
- M14 same-tier candidates (`vllm-sr-dev2-dec-m14`, ×2 upweighting): 2B `2b-RAUP` closed (Score floor 236 < 245);
  **`4b-LHA10UP`** and **`08b-RAUP`** are still training. With nothing released from M13 they are judged against the
  current releases (LH, DEV2.0-0.8B). Given the 4B result here, the 4B one needs item 4 (mlx-diag) as well as item 1.

## GPU-hours (launch receipts; `m13_gpuh.py table` on E and F, formal `GPU-SECONDS.jsonl`)

| Node | Arms | Other | Total |
| --- | --- | --- | ---: |
| E | `08b-RAAG` 3.52, `08b-RASD` 2.83 | readouts 0.40, SD labeling 0.28 | 7.04 |
| F | `4b-LHA10SD` 2.19, `4b-LHA5` 2.08, `2b-RASD` 1.03 | readouts 1.17, SD labeling 0.33, merges 0.06 | 6.86 |
| F formal | CAL fits, smokes, collections, mlx-diag (2 points) | | 0.31 |
| **M13** | | | **14.21 of 80** |

Every arm stayed within its cap (4B 5.0, 2B 2.5, 0.8B 5.5).

# 9B Milestone 6 result: K5-a12 is +1.62 v3 but not a successor (items 1 and 4); DEV2.0-9B stands

Run under the [preregistration](lux9b-m6-prereg-2026-09-30.md) (`f41402e68`) and
[amendment 1](lux9b-m6-prereg-amendment-1-2026-09-30.md) (`7380a3cbf`), with the
[lock](lux9b-m6-formal-lock-2026-09-30.md) (`e352dae8f`) frozen before any formal prediction. The released
`llm-semantic-router/DEV2.0-9B` (K-a13 at T = 1; weights of package `53bac735`) stays the 9B model. Nothing was uploaded.

**Verdict.**

- Both treatment arms stopped at their preregistered early rule:
  - KA (AutoJev-27B soft targets on the human-rated rows) at ΔP −0.36;
  - KH (the HS1 substitution) at ΔP −0.43, with Noul `rule_precedence` 327 < 346.
- The only rule pick, and so the only finalist, is **K5-a12** (½ five-seed K soup + ½ Lux 1.0).
- Post-key it scores **69.362 vs 67.737** for the released T = 1 run: **+1.62 [−0.19, +2.41]**.
  - Typed T gains significantly: +.030 [+.018, +.042].
  - Human transfer is level: +.006 [−.022, +.018].
- It fails successor item 1 (the lower bound is below 0) and item 4 (card-eligible mlx-diag −.010 [−.020, −.001]).
- It passes items 2, 3, 5, 6 and 7. There is no passer, so there is no item-8 hand-off and no successor.

## Early rules (first seed, ⅓ toward Lux; vs the control K-s4's ⅓ point)

| ⅓ point | T | H3 | H | P | C / N / S | rule_precedence | Rule |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| K-s4 (own-Lux control) | .9231 | .5654 | .5846 | 73.46 | 800 / 346 / 331 | 346 | control |
| KA-s4 (AutoJev on S) | .8988 | .5731 | .5945 | 73.10 | 800 / 272 / 366 | 272 | **stop**: ΔP −0.36 < +0.5 (`early-KA-s4.json` `9e3cf68d…`) |
| KH-s4 (HS1 F3 + ½ F1) | .9250 | .5432 | .5765 | 73.03 | 800 / 327 / 353 | 327 | **stop**: ΔP −0.43; RP 327 < 346 (`early-KH-s4.json` `488d2b17…`) |

- **KA:** H3 went up (+.008), and so did typed Score (+35; more level-1 answers). But Noul `rule_precedence` fell 74 items
  to Lux 1.0's level. That is the same trade as M5's A7 dose.
- **KH:** Score went up (+22), but Noul fell (−19, all of it `rule_precedence`) and H3 fell (−.022). The HS1 "condition not
  met" and quote-check rows did not transfer to typed Noul.
- Neither arm ran a second seed or a line, and the chains followed the stop (`m6-gpu6` ended after the KH rule).

## Lines and the rule (development; never v3)

Readout `m6/readout-lines/readout.json` `ceb10a86…` (typed DEV 1,600 + CSS pilot 1,430, 16K, runtime `3277dec9d`). R is the
fresh re-read of K-a13, which reproduces M4 exactly.

| Artifact | T | H3 | H | P | C / N / S | RP | Rule (vs R) |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| R = K-a13 | .9250 | .5622 | .5795 | 73.22 | 799 / 338 / 343 | 338 | anchor; floors C 775, N 326, S 331, RP 334 |
| K5 ⅓ | .9350 | .5639 | .5773 | 73.47 | 800 / 339 / 357 | 339 | eligible, G +.010 |
| **K5 ½ = K5-a12** | **.9550** | **.5751** | **.5840** | **74.68** | **800 / 372 / 356** | **372** | **eligible, G +.030 = G\*; α\* = ½** |
| K5 ⅔ | .9494 | .5735 | .5746 | 73.86 | 800 / 374 / 345 | 374 | eligible, G +.024 |
| K5 soup (α 1) | .8950 | .5691 | .5450 | 69.84 | 758 / 320 / 354 | 320 | ✗ Choice, Noul, family, RP |
| K2 ⅓ (report only) | .9356 | .5618 | .5743 | 73.30 | 800 / 333 / 364 | 333 | ✗ H3, RP |
| K2 ½ (report only) | .9531 | .5690 | .5776 | 74.20 | 800 / 356 / 369 | 356 | eligible (report-only pick) |
| K2 soup (α 1) | .8938 | .5642 | .5444 | 69.76 | 800 / 283 / 347 | 283 | ✗ Noul, family, RP |

- **Seed rule:** P(K5 soup) 69.84 ≥ the seed mean 65.51 (`seed-K5.json` `ee9e8069…`).
- **α rule:** ⅓ gains less than 0.75·G\* = .0225, so α\* = ½ (`alpha-K5.json` `7e6c2a3e…`).
- **Finalists:** K5-a12 only (`finalists.json` `c500e7f4…`). The KA and KH slots are empty and are not refilled.
- **Report-only contrasts:**
  - K5 − K3 at equal α (M4 K line): ⅓ T +.010, ½ +.016 (Choice 800 vs 793, Noul 372 vs 363, H3 .5751 vs .5748), ⅔
    +.018, soup +.049 (Choice 758 vs 704). The two new seeds were the strongest K seeds (P 67.06 / 68.15 vs 61.5–65.8).
  - Seed level (α 1), KA-s4 / KH-s4 vs K-s4: P 70.33 / 67.72 vs 67.06; H3 .5756 / .5613 vs .5451.

## Formal post-key run (node A GPU6, 16K; all steps exit 0)

| | K5-a12 | Incumbent (released T = 1) |
| --- | ---: | ---: |
| v3 | **69.362** | 67.737 |
| T / H | .8406 / .5723 | .8106 / .5660 |
| Choice / Noul / Score (typed FINAL) | 745 / 737 / 263 | 736 / 715 / 246 |
| constraint competition / exception stack / resource ledger | .863 / .843 / .657 | .840 / .787 / .615 |
| typed Brier / ECE (T = 1) | .0964 / .0689 | .0997 / .0165 |
| public 231 (easy / standard / hard) | 179 (48 / 66 / 65) | 178 (48 / 66 / 64) |
| mlx-diag type macro (incl. XNLI part) / non-English | .8169 / .8094 | .8224 / .8157 |

- **CSS15:** H is the median task. It rose (.572 vs .566, emotion), but 9 of 15 tasks fell slightly, and the 15-task
  mean macro-F1 is .540 vs .544. The largest falls are reddit humor −.021, MRF −.020, FLUTE −.019, IBC −.013 and media
  ideology −.012; the largest gains are persuasion +.018 and wiki politeness +.012.
- **Development to formal:** the typed gain carried over in full (dev T +.030 → formal T +.030). The dev H3 gain
  (+.013) did not show up as a formal human-transfer gain (+.006, n.s.).
- **Calibration** (the 23:15 rule, `devcal.json` `1630ed7c…`): CAL698 improved typed-DEV ECE but worsened CSS-pilot Brier
  (.5611 vs .5587) and ECE (.0665 vs .0575), so the rule ships **T = 1**. The derived T = 1 run
  (`formal-m6/K5-a12-16k-t1`, REPORT `87536dbb…`) has identical answers and identical v3, T, H and gates.
- **Checks:**
  - Every output manifest binds `ff555899…` and calibration `ba73a268…`. Image `f83b1d10`.
  - Invalid answers: 0 typed, 0 public, 4 CSS15 (over the limit, as for the incumbent).
  - The cache changed only in 13 FLA `__grp__` JSONs (new `formal-m6` paths); every kernel and autotune file is
    byte-identical to `af623300…`.
  - `SHA256SUMS` re-hashed 24 / 24.

## Successor rule vs the released T = 1 run (`K5-a12.gates/`, `successor.json` `f97d5366…`)

| Item | Reading | Verdict |
| --- | --- | --- |
| 1. v3 `ci95.low` > 0 | +1.62 [−0.19, +2.41] (`PAIRED-vs-DEV2.0-9B-T1.json` `7bd623a3…`) | **FAIL** |
| 2. H `delta.high` ≥ 0 | +.006 [−.022, +.018] | pass |
| 3. types | Choice / Noul / Score `OK` (`types.json` `5348ee70…`) | pass |
| 4. card-eligible mlx-diag `ci95.high` ≥ 0 | −.010 [−.020, −.001]: Choice −.003 [−.010, +.003], Noul −.017 [−.034, .000]; Japanese −.033 [−.062, −.005] (`MLX-PAIRED…` `c5e001e9…`) | **FAIL** |
| 5. tier gates | vs Lux1 16K +3.55 [+1.35, +5.64], H high +.047; vs Nimble v2 +7.31 [+4.33, +11.21], H high +.100; types `OK` | pass |
| 6. overlap exposure | x60 `14e7c0ca…`: `groups: []` | pass |
| 7. public 231 vs I | +1, p 1.0, `OK` (`PUBLIC231…` `d7e21fae…`) | pass |
| 8. C1 post-key guard | not reached (only a passer of items 1–7 is handed to the custodian) | — |

On the T = 1 derived run the readings are identical (`K5-a12-16k-t1.gates/successor.json` `db102c54…`).

## Diagnostics (never selection)

- **`hs1-dev`** (`m6/hs1/K5-a12.json` `8093620d…`; the incumbent read with its CAL698): no skill change, as expected with
  no HS1 rows.
  - Quote adoption (F1) .749 vs .749; false yes on unmet conditions (F3) .161 vs .152.
  - Accuracy F1 .562 vs .568, F2 .653 vs .657, F3 .841 vs .846; every paired CI includes 0.
- **HT-DEV v2** (reported only: M6's rule was frozen before the 04:10 note; `K5-a12-htdev2/readout.json` `a500a420…`):
  H_dev2 .5569 vs `9b-m4-K-a13` .5636, Δ −.0068 [−.0169, +.0033], **TIE**. This agrees with the formal human-transfer
  reading (level).
- **Score levels:** all five used, level-0 recall .457, largest answer "4" at 27.5%, 0 invalid.

## Disclosures and lessons

- **Data (02:25 note):**
  - The finalist was trained only on x60: no PN1, no HS1 and no AutoJev targets. So the 02:25 disclosure does not apply.
  - KH used HS1 at `171e6f0c` (F3 + ½ F1). The template typo is F2-only, so KH's block had none of it, and KH never
    became a finalist.
  - KA's AutoJev provenance caveat applies to no finalist.
- **Lessons:**
  - Teacher soft targets on the human rows (KA), like M5's A7 dose, trade Noul `rule_precedence` for Score at 9B.
  - HS1 substitution (KH) lowered both Noul and the CSS pilot at matched tokens.
  - More K seeds in the soup is what helped. Typed transferred fully to formal (+.030), but human transfer stayed
    level, and the ½ point costs mlx-diag (−.010, Japanese −.033), the same risk M5 flagged for α ½ artifacts
    (KN-a12 −.020).
  - A 9B successor still needs a human-transfer or multilingual lever on top of the K5 typed gain.
- **Kept on node A** (no HF upload): `m6/K5-a12-build/soup` + `m6/K5-a12-cal` (`SHA256SUMS` `8abf50e2…`, 24 files),
  the K5 / K2 soups (`m6/K5-soup-build`, `m6/K2-soup-build`), every seed run, and the scored runs
  `formal-m6/K5-a12-16k` (SEAL `269bb7a7…`, REPORT `48412092…`) and `-16k-t1`. They are available if a later milestone
  pairs K5 with a multilingual lever.

## Resources

- **GPU-hours ≈ 12.30 of the 24 cap** (projection ≈ 20):
  - 12.08 by M6 job receipts: the AutoJev waves 0.87, four seeds with preflights and in-arm readouts, the early-rule and
    line readouts, and `hs1-dev`;
  - plus 0.22 for the formal collections (smoke 0.033, 16K 0.118, mlx 0.036, HT-DEV v2 0.032).
  - Soups, rules, gates and the T = 1 derivation were CPU.
- **Chains:**
  - `m6-g6` / `m6-g7` (mirror `e611b96b4`) ended 00:26Z / 00:58Z.
  - `m6-post-K5-a12` (mirror `e2daf2afc`) ran 01:08–01:30Z.
  - The node A GPU6–7 leases are `track=9b-m6 status=idle` (01:33Z), free for reassignment.

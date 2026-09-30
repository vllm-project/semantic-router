# Decoder Milestone 8-small — results (DEV2.0-27B → DEV2.0-2B / DEV2.0-0.8B distillation; 2026-09-30)

Preregistration [`dec-m8s-prereg-2026-09-30.md`](dec-m8s-prereg-2026-09-30.md) (`c99cf9b7f`), amendments
[1](dec-m8s-amendment-1-2026-09-30.md) (GPUs), [2](dec-m8s-amendment-2-2026-09-30.md) (4B M8 Noul / Score floors),
[3](dec-m8s-amendment-3-2026-09-30.md) (Score floor when the incumbent is itself flagged), data lock
[`dec-m8s-datalock-2026-09-30.md`](dec-m8s-datalock-2026-09-30.md) (parts 1 / 2 `16044feab` / `9585d5d39`).
Development readouts are never release scores; formal v3 / public 231 are post-key same-panel comparisons. Nothing was
uploaded; C1 was not opened. Aggregates: [`dec-m8s-results-2026-09-30/`](dec-m8s-results-2026-09-30/).

## Bottom line

**No successor in either tier; DEV2.0-2B (`a53cf66a`) and DEV2.0-0.8B (`bede7938`) stand.**

- **2B: no finalist.** Every point of every line (D1, D2, C × α 1 / ½ / ⅓) fails the typed-DEV Score floor:
  `set_reconciliation` 172 / ≈190 / ≈215 against a floor of 245 (the incumbent reads 257). The drop is the same in
  all three arms at equal α, so it comes from the top-up itself, not the teacher. No formal run.
- **0.8B: two finalists, both fail item 1.** `08b-D2-a1_2` (½ D2 soup + ½ released): post-key v3 50.447, **+0.18
  [−0.52, +2.44]** against the node-B collection of the released weights; `08b-C-a1_3` (⅓ control soup + ⅔ released):
  50.172, −0.09 [−0.89, +1.20]. Both keep human transfer level, pass types, the tier gate, card-eligible mlx-diag
  (significant gains, +.0075 and +.0057) and public 231; neither passes items 1–7, so there is no C1 candidate and no
  item-8 hand-off.
- **The distillation signal is weak at these doses.** A20r KD (λ 1) moved development behaviour (typed DEV +.02–.03
  and HT-DEV v2 +.013–.016 over the matched control at 0.8B α 1; SELECT700 +.03 at 2B), but nothing reached formal
  v3 or human transfer.

## Design as run

Top-ups of the released BF16 weights (`--init decision2`), one epoch over a 6.0M-native-token file per tier (whole
groups of the tier's r2-clean recipe, 1:1 human / typed by the frozen source partition), 3 seeds per arm with paired
data order; D1 = CE + 0.5·Brier + 1.0·KL(A20r) on every row, D2 = the same on human rows and gold on typed rows,
C = the released objective (2B own-Sol KL 0.5; 0.8B none). A20r targets at T = 1 on its scored kernel path (parity
bit-exact on CAL698). Each seed: 159–163 updates (2B), 144–149 (0.8B). Every preflight passed; nothing was rerun.

## Early rule (seed 1; HT-DEV v2 vs C-s1 not FLAG and SELECT700 ≥ C − 0.03)

| Arm | HT-DEV v2 Δ vs C-s1 [95% CI] | SELECT700 family macro (D / C) | Verdict |
| --- | --- | --- | --- |
| 2B D1 | +.0075 [−.0038, +.0185] | .871 / .841 | PASS |
| 2B D2 | −.0059 [−.0170, +.0054] | .870 / .841 | PASS |
| 0.8B D1 | +.0095 [−.0034, +.0222] | .869 / .864 | PASS |
| 0.8B D2 | +.0120 [−.0013, +.0251] | .856 / .864 | PASS |

## Development lines (node B, `dbe5f32b`, 16K, T = 1; gates vs I read the same way)

References: `2b-I` T .610, H3 .4278, P 47.14, typed C / N / S 489 / 230 / 257, Score5 check clean (top share .38);
`08b-I` T .613, H3 .3872, P 40.73, 610 / 212 / 159, Score5 check COLLAPSE / NO-GAIN (accuracy .20 at chance, top
share .31; amendment 3). Both references reproduce the eval track's HT-DEV v2 collections of the same weights
exactly (Δ 0.0000, CI [0, 0]), and `2b-I` reproduces M7's node-A readout (T .610 / H3 .4278 / P 47.14).

**2B** (floors: Choice ≥ 465, Score ≥ 245, `rule_precedence` ≥ 226, families −.10):

| Line | α | T | H3 | P | C / N / S | HT-DEV v2 Δ vs I [CI] | Gate |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| C | 1 | .609 | .416 | 45.59 | 570 / 233 / 172 | −.0177 [−.030, −.005] TIE | Score |
| C | ½ | .590 | .428 | 45.93 | 524 / 229 / 191 | −.0030 TIE | Score |
| C | ⅓ | .596 | .426 | 46.36 | 511 / 227 / 216 | −.0014 TIE | Score |
| D1 | 1 | .517 | .417 | 43.15 | 432 / 223 / 172 | −.0138 [−.026, −.001] TIE | Choice, Score, Noul, families |
| D1 | ½ | .559 | .428 | 45.31 | 474 / 229 / 191 | −.0043 TIE | Score |
| D1 | ⅓ | .581 | .423 | 45.46 | 488 / 225 / 216 | −.0010 TIE | Score, Noul |
| D2 | 1 | .538 | .416 | 44.90 | 453 / 236 / 172 | **−.0348 [−.048, −.021] FLAG** | Choice, Score, families, HT-DEV v2 |
| D2 | ½ | .561 | .426 | 45.63 | 482 / 228 / 188 | −.0107 [−.020, −.002] TIE | Score |
| D2 | ⅓ | .580 | .425 | 45.43 | 488 / 226 / 214 | −.0021 TIE | Score |

**0.8B** (floors: Choice ≥ 586, Score ≥ 147, `rule_precedence` ≥ 208, families −.10; Score5 no new concentration):

| Line | α | T | H3 | P | C / N / S | HT-DEV v2 Δ vs I [CI] | Gate |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| C | 1 | .617 | .368 | 40.45 | 658 / 223 / 106 | −.0003 TIE | Score |
| C | ½ | .621 | .383 | 41.00 | 638 / 219 / 136 | −.0011 TIE | Score |
| C | ⅓ | .621 | .385 | 40.66 | 631 / 215 / 147 | −.0023 TIE | **PASS** (finalist, slot 2) |
| D1 | 1 | .636 | .379 | 42.39 | 654 / 232 / 132 | +.0164 [+.001, +.030] TIE | Score |
| D1 | ½ | .625 | .386 | 41.63 | 632 / 224 / 144 | +.0079 TIE | Score |
| D1 | ⅓ | .618 | .386 | 40.75 | 628 / 215 / 145 | +.0036 TIE | Score |
| D2 | 1 | .642 | .384 | 42.67 | 662 / 230 / 135 | +.0125 [−.002, +.026] TIE | Score |
| D2 | ½ | .626 | .390 | 42.07 | 634 / 220 / 148 | +.0054 TIE | **PASS** (pick; finalist, slot 1) |
| D2 | ⅓ | .624 | .389 | 41.46 | 627 / 218 / 154 | +.0070 [+.000, +.014] TIE | PASS |

No point reached an HT-DEV v2 GAIN; the D2 pick is the larger passing α. Score5 check-half top shares stayed
.29–.37 everywhere (no concentration); 0.8B accuracy stayed at chance (.205–.228) in every point.

## Formal (0.8B finalists; node B collection, node A scoring)

The 0.8B node-B reference `m8s-ref-08b-I` (the released weights on a fresh cache, then frozen) was **not**
answer-identical to the node-A T = 1 binding (category changes typed FINAL 9, CSS15 20, public 231 1; max drift
.028), so per the prereg it is the paired bar ("bar-t1"; v3 50.263, T .5741, H .4401 vs node A's 50.236). The node-A
pairing is kept for disclosure. CAL698 16K fits were rejected by the 23:15 rule for both finalists (T = 1).

| Item | `08b-D2-a1_2` (½ D2 soup + ½ released) | `08b-C-a1_3` (⅓ C soup + ⅔ released) |
| --- | --- | --- |
| Revision (per-file list) | `d50fad619a46…` | `2a6a12275ce1…` |
| v3 / T / H | 50.447 / .5781 / .4402 | 50.172 / .5741 / .4385 |
| Typed FINAL C / N / S (bar 531 / 610 / 107) | 543 / 613 / 102 | 535 / 611 / 105 |
| 1. v3 vs bar (node B) | +0.18 [−0.52, +2.44] **FAIL** | −0.09 [−0.89, +1.20] **FAIL** |
| (node-A binding, disclosed) | +0.21 [−0.47, +2.45] | −0.06 [−0.69, +1.15] |
| 2. H vs bar | [−.011, +.040] OK | [−.015, +.020] OK |
| 3. Types | OK | OK |
| 4. mlx-diag card-eligible vs node-B reference | **+.0075 [+.0033, +.0124]** OK | +.0057 [+.0024, +.0097] OK |
| 5. Tier gate vs adopted Eos 1.0 | +7.90 [+4.44, +14.02] OK | +7.63 [+3.88, +13.41] OK |
| 6. Overlap | (a) top-up receipt 0 groups; (b) not evaluated (item 1 decides) | same |
| 7. `gates public231` vs bar | 155 vs 155 OK (easy 48, standard 64, hard 43) | 158 vs 155 OK (p .25; hard 46) |
| 8. C1 | not requested (fails item 1) | not requested |
| vs Intern-0.8B | +6.91 [+1.79, +11.16] | +6.64 [+0.83, +10.54] |
| vs Kev-0.8B | +7.23 [+2.35, +12.07] | +6.96 [+1.43, +11.51] |

Every M8s point descends from the released weights (the top-ups start there and the lines mix them back in), so any
M8s model inherits the incumbent's disclosed exposure (0.8B 33 groups; 2B 24 groups / 57 rows); the successor tool's
"inherited exposure: none" does not see that. Item 6(b)'s overlap run stopped because the spec lacked the mlx panel
the inherited receipt needs; it was not repeated since item 1 already fails.

Paired CIs vs Decider 2B / This-That 1.2: none (no 2B finalist).

## Diagnostics (never selected on)

`hs1-dev` (F1 quote adoption, ideal .50; F3 false yes on unmet conditions, ideal 0):

| Point | Adopt | False yes | Point | Adopt | False yes |
| --- | ---: | ---: | --- | ---: | ---: |
| 2b-I | .780 | .625 | 08b-I | .755 | .720 |
| 2b-C-a1 | .779 | .571 | 08b-C-a1 | .726 | .777 |
| 2b-D1-a1 | .754 | .554 | 08b-D1-a1 | .775 | .738 |
| 2b-D2-a1 | .759 | .530 | 08b-D2-a1 | .755 | .729 |
| | | | 08b-D2-a1_2 (finalist) | .754 | .723 |
| | | | 08b-C-a1_3 (finalist) | .754 | .726 |

**Matched-token effects at α 1 (development):** 0.8B D1 − C typed T +.019, HT-DEV v2 +.017; D2 − C +.025 / +.013;
2B D1 − C T −.092 (Choice −138), HT-DEV v2 +.004; D2 − C −.071 / −.017. A20r argmax agreed with gold far more than
the 2B own-Sol teacher on the same rows (human C / N / S .902 / .845 / .656 vs .750 / .724 / .446).

## Findings

1. **The small tiers do not take this teacher at this dose.** Distillation from a 72-v3 teacher into 2B / 0.8B top-ups
   changed development behaviour but not formal v3 or human transfer. At 0.8B the D2 point is +0.18 v3 (n.s.) with a
   significant mlx-diag gain; at 2B KD on every row cost typed Choice sharply (D1 α 1 432 vs 489).
2. **The 2B typed-Score loss comes from the top-up composition, not the teacher.** All three 2B arms land on the same
   `set_reconciliation` count at equal α (172 / ≈190 / ≈215). The 1:1 human / typed file over-weights human Score rows
   (1,847 human vs 914 typed Score rows) relative to the recipe; the shared Score head moves toward them. The 4B M8
   slice is proportional; a 2B retest should be too.
3. **HT-DEV v2 moved the right way only at 0.8B** (D arms +.013 to +.017 over the control at α 1), never to GAIN, and
   the ½ finalist's formal H was level (+.0001).
4. **Node-B formal path for 0.8B:** the released 0.8B is not answer-identical across the node-A / node-B formal paths
   (30 category changes); future 0.8B node-B work should use `m8s/formal/cache-frozen-08b` with the node-B bar.

## Incidents

- The first `ref08b` attempt stopped at the eval adapter's argument check (`model_id` missing; 6.9 s on GPU2, no model
  load); fixed in `8e7a133be`, moved to `m8s/formal/attempts/`, relaunched.
- An amendment-2 unit test was committed failing in `84dc45192` (a superseded case; the code was right); fixed in
  `a757cc710` before any use of the test.
- Two copies of the final `hs1-dev` diagnostic job raced (duplicate launch); the `08b-C-a1_3` readout's receipt
  write collided after its container finished. The 2,396 predictions were complete and were scored by hand; about
  one GPU-minute has no receipt.
- Header times in the prereg (≈18:05) and amendment 1 (≈17:50) are late; the commits were at 09:30Z / 09:38Z, before
  the first GPU job (09:39Z).

## GPU-hours (node B GPU2 / 6 / 7; receipts `m8s/GPU-SECONDS.jsonl`)

| Item | GPU-h | Cap |
| --- | ---: | ---: |
| A20r parity gate + two label shards | 1.014 | 3.0 |
| 2B arms D1 / D2 / C (3 seeds each, seed-1 preflights) | 0.291 / 0.289 / 0.300 | 1.2 each |
| 0.8B arms D1 / D2 / C | 0.216 / 0.224 / 0.208 | 0.9 each |
| Early-stop HT-DEV v2 collections | 0.104 | 0.6 |
| Lines, references, Score5, `hs1-dev` (≈0.02 unreceipted) | 1.42 | 4.0 |
| Formal (0.8B node-B reference + mlx, 2 CAL fits, 2 smokes, 2 collections, 2 mlx) | 0.381 | 4.0 |
| **Total** | **≈4.45** | 24 |

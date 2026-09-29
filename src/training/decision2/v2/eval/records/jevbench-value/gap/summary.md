# JevBench public 231 — item-level gap analysis (2026-09-29)

Scope: why DEV2.0 models look good on JevArena (v3 = 100·√(T·H), typed FINAL + CSS15) but weak on
public 231. CPU only, node A. This file holds aggregates only. Per-item joins, discordant lists and cue
flags stay in the private gap directory on node A. Numbers: `gap.json`. Drivers: `drivers/`.

Panel facts that bound the analysis:

- Easy is 48/48 for every run. Standard ranges 66–72 of 72, with all 12 standard four-level Score items
  right for every run. So public 231 is in effect a 111-item hard panel.
- Input length and state format are fully confounded with tier. Easy and standard states are all under
  200 characters. All 35 dict states and all 37 inputs over 4k characters are in hard. Length and format
  bins therefore only re-cut hard.
- The per-item results of DEV2.0-4B and its released re-score are identical. So are those of DEV2.0-8B
  and its released re-score, of Lux 1.0 same-renderer and adopted, of Nox 1.0 at 8k, 16k and adopted,
  and of 27B C0 and F0. Duplicated pairs are reported once.

## 1. Pairs (paired, 231 items; b = candidate-only right, c = other-only right)

| Candidate vs other | Correct | b / c | Δ items [95% CI] | MDD80 | exact p | hard b / c | Verdict |
|---|---|---|---|---|---|---|---|
| DEV2.0-4B vs Nox 1.0 | 171 / 173 | 5 / 7 | −2 [−8.8, +4.8] | 9.7 | .77 | 4 / 7 | (a) noise |
| DEV2.0-4B vs Decider 4B | 171 / 192 | 6 / 27 | −21 [−31.9, −10.1] | 15.6 | .0003 | 6 / 23 | (b) skill |
| DEV2.0-8B vs Lux 1.0 (either renderer) | 178 / 183 | 2 / 7 | −5 [−10.8, +0.8] | 8.3 | .18 | 2 / 6 | (a) noise; weak (d) hint |
| DEV2.0-27B F1 vs Eikos-27B | 198 / 212 | 2 / 16 | −14 [−22.1, −5.9] | 11.6 | .0013 | 2 / 15 | (b) skill |
| DEV2.0-27B F1 vs AutoJev-27B | 198 / 201 | 6 / 9 | −3 [−10.6, +4.6] | 10.8 | .61 | 6 / 8 | (a) noise |
| DEV2.0-27B F1 vs 27B C0 | 198 / 200 | 12 / 14 | −2 [−12.0, +8.0] | 14.3 | .85 | 9 / 13 | (a) net; (d) composition |
| JPT-4B vs Decider 4B (peer reference) | 203 / 192 | 22 / 11 | +11 [−0.2, +22.2] | 16.0 | .08 | 21 / 7 | (a) borderline |

Where the discordant items sit (b / c):

- **DEV2.0-4B vs Decider 4B.** 23 of the 27 losses are hard.
  - Hard families: long_policy 0 / 7 (p = .016), temporal_numeric 1 / 4, multi_hop 2 / 4, ambiguous 0 / 2,
    probability 0 / 2. Standard: adequacy 0 / 3.
  - By type: Choice 3 / 18, Noul 2 / 9.
  - By length: inputs over 4k characters 2 / 11, 1k–4k 3 / 10.
  - By state format: string states 4 / 25, dict states 2 / 2.
- **DEV2.0-27B F1 vs Eikos-27B.** 15 of the 16 losses are hard.
  - Hard families: long_policy 0 / 5, temporal_numeric 0 / 4, ambiguous 0 / 2, judge_hard 0 / 2,
    probability 0 / 2.
  - By type: Noul 0 / 7.
  - By length: 1k–4k characters 0 / 10.
- **DEV2.0-8B vs Lux 1.0.** multi_hop 0 / 4 (p = .125). Three of the 7 losses are dict states.
- **DEV2.0-4B vs Nox 1.0.** Losses are Noul 0 / 5; wins are Choice 5 / 2.

## 2. Format and parsing artefacts (all 44 runs checked)

- **Nothing to parse around.** Every run has 0 invalid answers, 0 renormalized probability sets,
  0 point-choice ≠ argmax, 0 truncated questions and 0 adapter errors. Probability sums deviate by at
  most 1e-4. Exact top ties: Decider 2, Nimble 3, AutoJev 3, Eikos 1; none in any DEV2.0 run.
  **No part of any 4B or 8B deficit comes from format or parsing (0 items).**
- **Score.** Standard four-level items are 12/12 for every run and the predicted level histogram equals
  the gold histogram. Hard Score (6 items: five 4-level, one 5-level) is 2–4/6 for everyone. There is
  no mass-shaped failure. DEV2.0-4B's mean mass on the standard levels is ⅓ / ⅙ / ⅓ / ⅙, the same
  shape as the gold distribution. DEV2.0-4B wins 1 Score item against Decider and loses none.
- **Choice position.** Predicted positions follow gold positions (gold 26 / 38 / 34 / 28 / 11 / 2).
  DEV2.0-4B picks the first option 35 times against 26 gold, a mild primacy lean. Nox 1.0 picks it 31
  times and peers 26–29. It is not a driver of the deficit.
- **Longest-option cue** (131 items with a unique longest option; gold is the longest in 44). The
  longest option is picked 38–44 times by every model, near its gold rate. Accuracy when the gold is the
  longest: DEV2.0-4B 35/44, Decider 38/44, Eikos 41/44. When it is not: 69/87, 79/87, 81/87. No model
  exploits the cue. At most 5 of Decider's 27 wins over DEV2.0-4B are gold-longest items the peer got
  right, and a reweighting moves at most about 2 items (the leak audit agrees).
- **Noul yes-bias (a skill, not an offset).** On hard Noul, gold is "yes" on 17 of 38 items.

| Run | Hard Noul "yes" answers (gold 17/38) | Hard gold-"no" right | Standard "yes" (gold 12/24) | False "yes" on 62 std + hard Noul | …of which p > 0.8 | Best in-sample threshold gain |
|---|---|---|---|---|---|---|
| Nox 1.0 | 24 | 10/21 | 16 | 15 | 12 | 0 |
| DEV2.0-4B | 25 | 7/21 | 16 | 18 | 10 | +2 |
| Decider 4B | 21 | 11/21 | 13 | 11 | 3 | 0 |
| JPT-4B | 21 | 15/21 | 14 | 8 | 4 | +3 |
| Lux 1.0 | 24 | 10/21 | 16 | 15 | 6 | +4 |
| DEV2.0-8B | 24 | 10/21 | 16 | 15 | 8 | +2 |
| 27B C0 | 19 | 16/21 | 10 | 5 | 3 | +1 |
| DEV2.0-27B F1 | 21 | 15/21 | 13 | 7 | 1 | +2 |
| Eikos-27B | 17 | 20/21 | 12 | 1 | 1 | 0 |

  The false "yes" answers are confident. Moving the threshold, even fitted in-sample, recovers at most
  2 items for DEV2.0-4B and 4 for Lux 1.0. So this is a failure to detect that a condition is *not* met,
  inherited from the 1.0 models, not a calibration offset.

- **Confidence on errors.** Wrong answers with confidence ≥ 0.9: DEV2.0-4B 12 (10 of them hard),
  Nox 1.0 12, Decider 6, DEV2.0-8B 4, Lux 1.0 4, F1 2, Eikos 2. Brier (valid items): DEV2.0-4B .174 vs
  Decider .122; DEV2.0-8B .143 vs Lux 1.0 .135; F1 .089 vs Eikos .064.

## 3. Skill attribution (items read privately; abstract descriptions only)

The high scorers win on hard items that need the skills below. The first two are absent from our
training and typed panels.

1. **Checking a planted human conclusion instead of adopting it.** The state quotes a person or role
   (a note, a draft recommendation, a reviewer or planner remark, a customer claim) who reaches a
   plausible but wrong conclusion. The right answer needs an independent recomputation or rule check.
   46 of 111 hard items carry such a quote: long_policy 13/19, probability 8/10, ambiguous 5/7,
   tradeoff 5/6, trap 5/8, multi_hop 5/18, temporal_numeric 5/15.
2. **Applying a long natural-language policy packet.** These are 8k–13k-character documents with
   amendments, superseded editions, precedence clauses and exception chains, where the case record
   must be matched to the controlling clause.
3. **Exact numeric and time arithmetic under stated rules.** Examples: unit conversion with
   round-up-before-compare, pro-rata refunds, elapsed-time windows across time zones or daylight-saving
   changes, currency and fee exclusions. Every model is weak here (1–7 of 15).
4. **Base-rate estimation from the rule-defined comparable subset**, not from a quoted overall rate.
   These are the probability items; the misleading overall rate is usually the planted quote.
5. **Detecting that a condition fails** (the Noul "no" side). This is where the yes-bias above shows up.
6. For the 8B: **multi-hop over structured records.** The chain runs alias → index → footnote →
   superseded-source rule, inside dict states.

Accuracy on hard items with and without the planted quote:

| Run | Hard with quote (/46) | Hard without (/65) | long_policy (/19) |
|---|---|---|---|
| Nox 1.0 / DEV2.0-4B | 22 / 18 | 37 / 38 | 5 / 3 |
| Decider 4B / JPT-4B | 30 / 37 | 43 / 50 | 10 / 14 |
| Lux 1.0 / DEV2.0-8B | 24 / 23 | 44 / 41 | 8 / 9 |
| JPT-9B / Nimble v2 | 32 / 28 | 49 / 41 | 11 / 10 |
| 27B C0 / F1 / F2 | 37 / 29 / 30 | 46 / 50 / 48 | 15 / 10 / 9 |
| Eikos-27B / AutoJev-27B | 38 / 34 | 54 / 47 | 15 / 14 |

Coverage in our data (sampled rows; the same regex; aggregates in `gap.json` → `corpus_shape`):

- **Planted quote.** Public-231 hard 41.4%. typed FINAL 0% (1,600 states). A7 arms: A7g 0%, A7o 0%,
  A7p 0%, A7i 0%, A7m 0.03%, A7h 0.97%, A7q 0.47%, A7s 0.05%, A7r 0%, A7k 0%.
- **Length.** typed FINAL states are all dicts, median 403 characters, none over 1k. Public-231 hard has
  a median of 1,583 characters and 33% over 4k. A7g has long rows (19% over 4k), but they are
  programmatic generator rows (tables, registers, automata), not policy prose with a planted
  conclusion.
- **Verdict on coverage.** The typed FINAL families (constraint competition, exception stack, evidence
  join, resource ledger) and the A7 typed curricula train rule composition on short synthetic states.
  Skill 1 is not covered at all, and skill 2 is covered only in a structural, non-prose form.

## 4. Lineage (public-231 counts; hard family columns)

Column key: LP = long_policy, MH = multi_hop, JH = judge_hard, TN = temporal_numeric, PR = probability,
TR = trap, AM = ambiguous, TO = tradeoff, AD = adversarial, RH = routing_hard, Quote = hard items with a
planted quote (out of 46).

| Run | v3 | T | H | Pub | Std | Hard | LP | MH | JH | TN | PR | TR | AM | TO | AD | RH | Quote |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Nox 1.0 | 56.47 | .614 | .519 | 173 | 66 | 59 | 5 | 12 | 9 | 3 | 6 | 7 | 3 | 3 | 6 | 5 | 22 |
| X2 | 55.99 | .584 | .537 | 174 | 67 | 59 | 6 | 11 | 9 | 2 | 7 | 7 | 3 | 3 | 6 | 5 | 22 |
| X4R-s2 | 53.71 | .601 | .480 | 179 | 67 | 64 | 6 | 13 | 9 | 2 | 9 | 7 | 3 | 4 | 6 | 5 | 24 |
| X4K-s1 | 55.36 | .598 | .512 | 173 | 67 | 58 | 5 | 11 | 9 | 2 | 7 | 7 | 3 | 3 | 6 | 5 | 21 |
| N4T | 56.51 | .645 | .495 | 173 | 67 | 58 | 3 | 12 | 9 | 3 | 6 | 7 | 3 | 4 | 6 | 5 | 19 |
| N4L | 58.90 | .646 | .537 | 173 | 67 | 58 | 3 | 13 | 9 | 2 | 6 | 7 | 4 | 3 | 6 | 5 | 18 |
| N4LKr | 59.54 | .629 | .563 | 171 | 67 | 56 | 4 | 11 | 9 | 1 | 6 | 7 | 3 | 4 | 6 | 5 | 19 |
| N4LX | 60.37 | .646 | .564 | 173 | 67 | 58 | 4 | 12 | 9 | 3 | 7 | 7 | 3 | 3 | 5 | 5 | 20 |
| **DEV2.0-4B (N4XF)** | 63.15 | .688 | .580 | 171 | 67 | 56 | 3 | 13 | 9 | 2 | 6 | 7 | 3 | 3 | 5 | 5 | 18 |
| N5N | 60.06 | .638 | .565 | 174 | 67 | 59 | 6 | 12 | 9 | 2 | 6 | 8 | 3 | 3 | 5 | 5 | 20 |
| Decider 4B | 61.88 | .689 | .555 | 192 | 71 | 73 | 10 | 15 | 9 | 5 | 8 | 8 | 5 | 2 | 6 | 5 | 30 |
| JPT-4B | 54.56 | .626 | .476 | 203 | 68 | 87 | 14 | 15 | 13 | 6 | 9 | 8 | 6 | 5 | 6 | 5 | 37 |
| Lux 1.0 | 65.81 | .776 | .558 | 183 | 67 | 68 | 8 | 14 | 9 | 4 | 7 | 8 | 4 | 3 | 6 | 5 | 24 |
| L2-8k | 65.36 | .751 | .569 | 184 | 66 | 70 | 9 | 14 | 10 | 4 | 7 | 8 | 4 | 3 | 6 | 5 | 25 |
| B-s1-16k | 64.60 | .759 | .550 | 181 | 66 | 67 | 8 | 13 | 9 | 5 | 7 | 8 | 4 | 2 | 6 | 5 | 23 |
| DW-16k | 68.57 | .852 | .552 | 179 | 67 | 64 | 9 | 10 | 9 | 3 | 7 | 8 | 5 | 2 | 6 | 5 | 25 |
| **DEV2.0-8B (K-a13)** | 67.74 | .811 | .566 | 178 | 66 | 64 | 9 | 10 | 9 | 4 | 7 | 8 | 4 | 2 | 6 | 5 | 23 |
| U-a13 / KN-a12 | 67.87 / 67.98 | .811 / .820 | .568 / .564 | 180 / 181 | 67 / 67 | 65 / 66 | 9 / 9 | 10 / 11 | 9 | 4 | 7 | 8 | 4 / 5 | 3 / 2 | 6 | 5 | 24 / 24 |
| 27B C0 | 56.75 | .555 | .581 | 200 | 69 | 83 | 15 | 13 | 12 | 5 | 8 | 8 | 7 | 4 | 6 | 5 | 37 |
| C1 / S1 / K1 (m2) | 60.1 / 62.7 / 60.3 | .61–.68 | .58–.60 | 192 / 199 / 197 | 65–69 | 79 / 82 / 80 | 12 / 11 / 11 | 13 / 15 / 13 | 13 / 13 / 15 | 3 / 4 / 3 | 8 | 8 | 7 | 4 / 5 / 4 | 6 | 5 | 32 / 34 / 32 |
| **DEV2.0-27B F1 (M3-A, +A7)** | 67.21 | .787 | .574 | 198 | 71 | 79 | 10 | 16 | 14 | 2 | 7 | 8 | 5 | 6 | 6 | 5 | 29 |
| F2 (M3-S control) | 64.47 | .688 | .604 | 194 | 68 | 78 | 9 | 15 | 12 | 3 | 8 | 8 | 7 | 5 | 6 | 5 | 30 |
| Eikos-27B / AutoJev-27B | 69.29 / 72.13 | .818 / .887 | .587 / .587 | 212 / 201 | 72 / 72 | 92 / 81 | 15 / 14 | 14 / 14 | 16 / 12 | 6 / 1 | 9 / 9 | 8 | 7 | 6 / 5 | 6 | 5 | 38 / 34 |

What the lineage shows:

- **4B.** The deficit against peers is inherited: Nox 1.0 already had 173 correct, 59 hard and 5/19
  long_policy. Along the lineage, public 231 moves only 171–179. The m3 distillation soups (N4T, N4J,
  N4L, N4LKr) took long_policy from 5–6 to 3–4 and planted-quote items from 22–24 to 18–20, and
  N4XF kept that. Meanwhile v3 rose by 6.7, from T (.614 → .688) and H. X4R-s2, the best hard count
  (64), had the lowest v3 (53.7).
- **9B.** Hard went 68 (Lux 1.0) → 70 (L2) → 64 at DW; the decline is multi_hop 14 → 10. Every
  full-fine-tune soup (DW, K, U, KN) has multi_hop 10–11. Taking ⅔ Lux 1.0 in K-a13 did **not** restore
  it (same as DW's ½). Meanwhile T rose from .776 to .811–.852.
- **27B.** Hard went 83 (C0) → 79–82 at m2 → 79 (F1). long_policy fell 15 → 11–12 at m2 and to 10 at
  F1. Planted-quote items fell 37 → 32–34 → 29, while multi_hop (+3) and judge_hard (+2) rose. F1 vs
  C0 on quote items is b / c = 1 / 9; on the rest it is 8 / 4 (Fisher p = .012). K1 vs C0 has the same
  shape (p = .044). A7 itself is not the cause: F1 (with A7) vs its no-A7 control F2 is 198 vs 194,
  quote items 29 vs 30. The shift comes with the shared own fine-tune mixture already in m2.
- **Across 36 distinct runs**, v3 correlates only weakly with public 231: Pearson .34 with the total and
  .32 with hard. T alone gives .30 and H alone .27.

## 5. Verdicts

- **DEV2.0-4B vs Nox 1.0: (a) noise.** −2 items [−8.8, +4.8], below the 9.7-item MDD. The small
  quote-item erosion at m3 (0 / 4) is consistent with (d) but not significant.
- **DEV2.0-4B vs Decider 4B: (b) real skill gap.** −21 items, p = .0003, 23 of the 27 losses hard.
  The missing skills are checking a planted human conclusion (quote items 18 vs 30) and applying long
  policy packets (long_policy 3 vs 10, b / c 0 / 7). The inherited Noul yes-bias adds to this
  (18 vs 11 false "yes"). Format or parsing: 0 items.
- **DEV2.0-8B vs Lux 1.0 (either renderer): (a) noise.** −5 [−10.8, +0.8], below the 8.3-item MDD.
  The remaining signal is multi_hop over structured records (0 / 4), a weak (d) from full-fine-tune
  soups that the ⅔ Lux weight did not undo.
- **DEV2.0-27B F1 vs Eikos-27B: (b) real skill gap.** −14 [−22.1, −5.9], p = .0013, 15 of the 16
  losses hard. Same skills: quote items 29 vs 38, long_policy 10 vs 15, the Noul "no" side 0 / 7,
  and exact time arithmetic.
- **F1 vs AutoJev-27B: (a) noise.** −3 [−10.6, +4.6].
- **F1 vs 27B C0: (a) on the total, (d) in composition.** The own fine-tune mixture traded
  planted-quote and long-policy items for multi-hop and judge items (Fisher p = .012). A7 is not the
  specific cause (F1 ≈ F2).
- **JPT-4B vs Decider 4B: (a) borderline** (+11, p = .08, hard 21 / 7). JPT-4B reaches 37/46 on quote
  items and 14/19 on long_policy, so the skill is attainable at 4B.

## 6. Gold audit (2026-09-29)

Upstream says the hard golds were written and cross-reviewed by two LLMs, with no human review.
Here p is the share of the 66 distinct models (stats helper matrix) that answer an item correctly.
Drivers: `drivers/gap_gold_select.py`, `drivers/gap_gold_sens.py`, `drivers/gap_cue_spot.py`. Private
judgments are on node A.

### Difficulty (items per p bin)

| Tier | < .10 | .10–.25 | .25–.50 | .50–.75 | .75–.90 | ≥ .90 |
|---|---|---|---|---|---|---|
| easy (48) | 0 | 0 | 0 | 0 | 0 | 48 |
| standard (72) | 0 | 1 | 4 | 7 | 7 | 53 |
| hard (111) | 12 | 18 | 26 | 22 | 20 | 13 |

### What was audited

- **All 19 items with p < .15**, all of them hard; no standard item is below .15.
- **Every item where at least 3 of 7 strong models agree on the same non-gold answer with
  confidence ≥ .8.** The strong models are Eikos-27B, AutoJev-27B, JPT-4B, JPT-9B, F1, C0 and
  Decider 4B. This selects 5 items: 4 of them are already in the p < .15 set and 1 has p = .47.
- **Total: 20 items.** By family: temporal_numeric 7, judge_hard 5, long_policy 5, probability 2,
  multi_hop 1.

Each item was read privately and its gold recomputed from the stated rules. The arithmetic, date and
time windows across clock changes, counting, aggregation and precedence steps were redone by hand or
with a scratch calculation.

### Results

| Grade | hard: temporal_numeric | hard: judge_hard | hard: long_policy | hard: probability | hard: multi_hop | Total |
|---|---|---|---|---|---|---|
| (a) clearly correct | 7 | 5 | 5 | 2 | 1 | **20** |
| (b) defensible but contestable | 0 | 0 | 0 | 0 | 0 | **0** |
| (c) likely wrong | 0 | 0 | 0 | 0 | 0 | **0** |

- **Several golds are deliberately knife-edge but unambiguous under the stated rules.** A total can
  land exactly on its threshold, a service count can fall 4 days short, a duration can be exactly 4 h
  where the rule needs "more than" 4 h, and a rest period can be exactly the minimum.
- **None of the 5 strong-model consensus answers is defensible.** Each one traces to a single missed
  step:
  - using the unconditional base rate instead of the conditional one;
  - skipping an aggregation rule;
  - trusting a tool's summary line;
  - taking the next override window instead of the current one;
  - accepting a judged response that contains an arithmetic error.
- **Planted quote among the 20 audited items.** The regex flags 9 of 20. Manual reading finds 12 of 20:
  12 of the 15 non-judge items carry one. The 5 judge_hard items have none; there, the judged response
  itself is the claim to check.

### Sensitivity (b / c; Δ items; exact p)

| Pair | All 231 | Excluding (c) | Excluding (b)+(c) | Worst case: all 20 audited items excluded (n = 211) |
|---|---|---|---|---|
| DEV2.0-4B vs Nox 1.0 | 5 / 7, −2, .77 | same | same | 5 / 6, −1, 1.0 |
| DEV2.0-4B vs Decider 4B | 6 / 27, −21, .0003 | same | same | 6 / 24, −18, .0014 |
| DEV2.0-8B vs Lux 1.0 | 2 / 7, −5, .18 | same | same | 1 / 7, −6, .07 |
| F1 vs Eikos-27B | 2 / 16, −14, .0013 | same | same | 2 / 10, −8, .039 |
| F1 vs AutoJev-27B | 6 / 9, −3, .61 | same | same | 4 / 6, −2, .75 |

No item was graded (b) or (c), so both prescribed exclusions leave every delta unchanged. Even
dropping all 20 of the hardest items keeps both real gaps significant.

### How the planted-quote flag works and how precise it is

- **Detection.** The flag is a heuristic regex (`drivers/gap_cue.py`), not manual. It looks for a role
  or utterance keyword (note, comment, draft, recommend…, message, e-mail, chat, says, reviewer,
  clerk, analyst, manager, lead…), followed by a colon or comma and an opening quote within 80
  characters.
- **Spot-check** (seeded sample of 15 flagged and 15 unflagged hard items, read privately):
  - Flagged: 14 of 15 are true, so precision is about .93. The false positive is a quoted customer
    request that states no conclusion.
  - Unflagged: 4 of 15 are misses, so the estimated recall is about .72. The misses are reported
    speech without a listed keyword, a chat transcript without quote marks, a paraphrased note inside
    a dict state, and a team "working assumption" line.
- **Estimated true rate.** About 60 of the 111 hard items (about 54%) carry a planted conclusion,
  against 46 flagged. The flag undercounts, so the quote/no-quote contrasts in section 3 are diluted,
  not inflated.

### How the "≤ 1% of A7 rows" figure was computed

- **Strict figure.** The same regex was run on a seeded reservoir sample of 4,000 rows per A7 arm
  `train.jsonl` (A7k: all 2,190 rows), taken from two cached snapshots of the training-data
  repository. The figure is the share of states with a match; the maximum is 0.97% (A7h).
- **Loose upper bound.** A looser pattern (any role or utterance keyword followed by a quote or a
  colon) matches much more:

  | Corpus | Loose match rate |
  |---|---|
  | public-231 hard | 60.4% |
  | A7o | 31.9% |
  | A7g | 21.0% |
  | A7p | 8.2% |
  | A7m | 4.6% |
  | A7h | 3.1% |
  | A7q | 2.9% |
  | A7i, A7k, A7r, A7s | ≤ .05% |
  | typed FINAL | 0% |

- **What the loose A7 matches are.**
  - A7g: a numeric field named "draft".
  - A7o and A7p: one-line, unattributed "Claim: record X has value Y" verification rows.
  - A7h and A7m: reported speech in natural text and NLI pairs.

  None is a role-attributed case conclusion inside a decision packet.
- **Refinement to section 3.** A7o and A7p do train short, unattributed claim verification (8–32% of
  their rows). That is the nearest relative of the planted-quote skill, but it comes in one- or
  two-line states, not inside long documents.

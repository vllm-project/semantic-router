# 9B Milestone 8 result: DEV2.0-27B distillation gives no finalist; DEV2.0-9B stands

Run under the [preregistration](lux9b-m8-prereg-2026-09-30.md) (`0d5534325`) and
[amendment 1](lux9b-m8-prereg-amendment-1-2026-09-30.md) (`2641b484e`, node A GPU3–4 only; KDX dropped), both frozen
before any GPU job. The released `llm-semantic-router/DEV2.0-9B` (K-a13 at T = 1) stays the 9B model. Development
readouts are never release scores. Nothing was uploaded.

**Verdict.**

- **D2** (A20r soft targets on the human-rated rows only, gold on the other rows) **stopped at its member-1 early
  rule**: Noul `rule_precedence` 245 against the control's 286 (limit −4).
- **D1** (A20r soft targets on every row, λ = 1.0) passed its early rule. It beat the control on typed DEV at every
  α. But **no point of its line is eligible under M7's α rule**, because each point's PN1-dev clean gold-no yes-rate
  is above K-a13's (⅓: +.0047; ½: +.059; ⅔: +.105, which also FLAGs on HT-DEV v2).
- **So there is no finalist, no formal run and no item-8 hand-off.** M7's control line C5, read under the same rule
  for the report, has no eligible point either.
- Resources: **3.21 of 24 GPU-h**.

## Teacher targets

- **Parity PASS:** the DEV2.0-27B scored runtime on node A reproduced 80 of 80 typed FINAL predictions of A20r's
  post-key run exactly (max |Δp| 0.0; `teacher-parity/parity.json`).
- **Collection:** three shards, 12,443 prompts, every question valid (identity `2e07451107a2…`, T = 1, 32K), about
  11 minutes per shard on one GPU. Build `m8/data/m8-kd/build` (manifest `8455dbb0…`): D1 teacher `cd531c3e…` (12,443
  rows), D2 teacher `b7985a44…` (3,153 rows). Both arms' train.jsonl is C's `eb55dbb2…` byte for byte.
- **Diagnostics on the top-up rows** (never selection; A20r was trained on overlapping mixtures, so these rows are
  partly in its training distribution):

  | Rows | n | accuracy vs gold, A20r / own Lux | argmax agreement | Noul yes-rate (gold / A20r / own Lux) |
  | --- | ---: | --- | ---: | --- |
  | all | 12,443 | .803 / .736 | .813 | .498 / .546 / .526 |
  | Choice | 4,310 | .899 / .853 | .864 | — |
  | Noul | 4,834 | .846 / .791 | .859 | .498 / .546 / .526 |
  | Score | 3,299 | .615 / .500 | .681 | — |
  | human-rated S | 3,153 | .619 / .581 | .780 | .489 / .524 / .535 |

- The matched-control recipe checks PASS for all six D continuations and both zero-step preflights: each D run's
  trainer contract equals C's same member except the teacher file and its row count.

## Early rules (member 1 = K-s1 continued with seed 20260931; α 1; T = 1)

| | typed T | C / N / S | `rule_precedence` | HT-DEV v2 H | hop yes | SELECT | clean gold-no yes (report) | Verdict |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| C-m1 (M7 control, M8 re-read) | .7913 | 678 / 286 / 302 | 286 | .5150 | .992 | .8586 | .714 | — |
| **D1-m1** | **.8338** | 685 / 309 / 340 | 309 | .5261 | .996 | .8730 | .735 | **continue** (`early-D1.json` `5e3ecd26…`) |
| D2-m1 | .7519 | 670 / 245 / 288 | 245 | .5295 | .996 | .8746 | .765 | **stop**: RP 245 < 286 − 4 (`early-D2.json` `9f354342…`) |

- **Incident (CPU only):** the two early scripts wrote the control's shared screen file
  (`screens/C-m1-e1/htdev2.json`) in the same second. D1's crashed on "file exists" and chain `m8-gpu3` ended. The
  deterministic rule was then completed on the unchanged inputs, and no training or readout was repeated. D1's
  members 2–5 then ran on GPU3–4 (chains `m8-gpu3b` / `m8-gpu4b`), because D2's stop had freed GPU4. The settings
  were exactly as preregistered.

## Lines and the rule (development; never v3)

Rules `m8/rules.sh 2641b484e rules-lines`, 12:02Z; readout `m8/rules-lines/readout.json` `d5ffcc13…`, runtime
`3277dec9d`. R = M7's re-read of K-a13 (T .9250, C / N / S 799 / 338 / 343, RP 338; floors C 775, N 326, S 331,
RP 334). The COORDINATION notes were re-read first (newest 20:05 UTC+8; no rule change for 9B M8).

| Point | T | G | C / N / S | RP | H3 | HT-DEV v2 H (Δ, verdict) | PN1 hop / clean gold-no / PAWS-X-6 | MLX-DEV-9B Noul-ML / Choice-ML | Eligible |
| --- | ---: | ---: | --- | ---: | ---: | --- | --- | --- | --- |
| R = K-a13 | .9250 | — | 799 / 338 / 343 | 338 | .5622 | .5636 (0, —) | .987 / .262 / .626 | — | anchor |
| **KD1 ⅓** | .9350 | +.0100 | 800 / 334 / 362 | 334 | .5599 | .5601 (−.004, TIE) | .987 / **.267** / .628 | +.001 [−.004, +.006] / +.007 [+.002, +.015] | **no**: clean gold-no +.0047 > 0 |
| KD1 ½ | .9506 | +.0256 | 800 / 360 / 361 | 360 | .5749 | .5465 (−.017, TIE) | .992 / .321 / .653 | +.012 [+.006, +.018] / +.018 [+.007, +.031] | no: clean gold-no +.059 |
| KD1 ⅔ | .9594 | +.0344 | 800 / 371 / 364 | 371 | .5742 | .5408 (−.023, FLAG) | .992 / .367 / .676 | +.017 / +.017 | no: FLAG; clean gold-no +.105 |
| C5 ⅓ (report) | .9287 | +.0037 | 800 / 330 / 356 | 330 | .5653 | .5571 (−.007, TIE) | .987 / .280 / .633 | −.001 / −.002 | no: RP 330 < 334; clean gold-no +.018 |
| C5 ½ (report) | .9481 | +.0231 | 800 / 361 / 356 | 361 | .5659 | .5463 (−.017, TIE) | .992 / .337 / .662 | +.015 / +.013 | no: clean gold-no +.074 |
| C5 ⅔ (report) | .9463 | +.0213 | 800 / 376 / 338 | 376 | .5714 | .5393 (−.024, FLAG) | .992 / .378 / .683 | +.017 / +.018 | no: FLAG; clean gold-no +.115 |
| K5-a12 (M6 finalist; report) | .9550 | +.0300 | 800 / 372 / 356 | 372 | .5751 | .5569 (−.007, TIE) | .987 / .318 / .652 | +.012 / +.013 | (formal: +1.62 [−0.19, +2.41], item 4 fail) |

- Rule outputs: `alpha-KD1.json` `6209c493…` (no eligible α), `alpha-C5.report.json` `e2436bae…`, `finalists.json`
  `2a150bbf…` (`finalists: []`).
- **The distillation effect (D1 − C at equal α, matched tokens and objective):**
  - typed T +.006 / +.003 / +.013 at ⅓ / ½ / ⅔, mostly Score (+6 / +5 / +26);
  - clean gold-no yes −.013 / −.015 / −.011 (less near-miss yes-bias);
  - HT-DEV v2 +.003 / .000 / +.001;
  - MLX-DEV-9B Choice-ML +.009 at ⅓.

  The A20r teacher is better than the own-Lux teacher on every development axis, but only by small margins.
- **Why no point clears the rule:** moving away from Lux 1.0 raises typed accuracy and the PAWS-X-style yes-bias
  together. That bias sank K5-a12's multilingual item.
  - KD1 ½ reads like K5-a12: T .951 vs .955, clean gold-no .321 vs .318, H_dev2 .5465 vs .5569. So it would most
    likely repeat K5-a12's formal outcome (item 1 borderline, item 4 fail).
  - KD1 ⅓ misses the guard by .005 (n = 850 rows; the paired 95% CI of the difference is [−.006, +.014]), but its
    typed gain (+.010) is too small to pass item 1. A successor needs about +2 v3 over K-a13, and K5-a12's dev +.030 gave formal +1.62
    [−0.19, +2.41].

## Lessons

- **At 9B, a teacher term must stay on every row.** D2, like M6's AutoJev-on-human-rows arm KA, lost Noul
  `rule_precedence` (−41 at member level) when the typed rows trained on gold only. D1's teacher on every row raised
  it (+23).
- **Our 27B teacher beats own Lux 1.0 as a distillation target, but by a development-sized margin** (typed +.003 to
  +.013, slightly less yes-bias). At M7's 6.2M-token dose, that does not move the 9B line past K5's typed-vs-multilingual
  trade-off.
- The same pattern appeared at 0.8B (20:05 note): KD beat the control in development but not in formal.
- **The binding constraint at 9B is the near-miss / paraphrase yes-bias** that grows with distance from Lux 1.0.
  Both the control and D1 fail on it at every α. M7's PN1-r2 arms show that the targeted lever is too strong at
  full dose.
  - Next levers: HR2 human-rated data (the program's main lever);
  - a yes-bias-balanced Noul block (hop-balanced, or PN1 at a much lower dose or learning rate) added to the D1 recipe;
  - or an α below ⅓, which needs a new preregistration.
- The MLX-DEV-9B screen reads every KD1 and C5 point as better than K-a13, while PN1 clean gold-no reads them as
  worse. Keeping both, as the preregistration argued, blocked candidates that would repeat K5-a12's item-4 failure.

## Resources and state

- **GPU-hours 3.21 of 24** (job wall-clock × GPUs from `m8_gpu_hours`):
  - teacher parity 0.03, shards 0.61;
  - the C-m1 re-read 0.10;
  - D1 / D2 member 1 with preflights 0.70, member-1 readouts 0.20;
  - D1 members 2–5 1.07;
  - the KD1 line's readouts 0.50.

  Soups, the teacher build and the rules were CPU.
- **Kept on node A** (no HF upload): `m8/data/m8-prompts`, `m8/data/m8-kd` (teacher files), `m8/teacher-*` (A20r
  predictions), the D1 / D2 member runs, `m8/KD1-*` (soup, interpolations, readouts), screens and rules.
- **Leases:** `owner.9b-m8` on node A GPU3–4 is released (12:06Z). The ~27B owner entries were never rewritten, and
  no M8 job ran on GPU2 after the 17:15 reclaim.
- **Tooling note:** `m8/screens.sh` is not safe for two concurrent writers of the same screen. Serialize the screens
  of a shared control, or add a lock, before reusing it with parallel arms.

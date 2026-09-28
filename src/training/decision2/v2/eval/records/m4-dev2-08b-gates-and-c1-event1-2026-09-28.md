# DEV2.0-0.8B release candidate: gate checks and JevArena-C1 scoring event 1 (2026-09-28)

**Candidate.** Recipe E8F: own Eos 1.0, full fine-tuning on the full A7 + v1 mixture, a
uniform three-seed soup, CAL698 calibration, and `qwen-full` packaging at 16,384 tokens.
Private `llm-semantic-router/dev2-dec-staging@16c0929a`, folder `m2/E8F-soup/`.

**Scored post-key run.** `/data/dev2/runs/dec/formal/m2/m2-E8F-soup-nodeA`: v3 50.236,
public 231 156.

Code: `v2/eval/gates.py` (`860b16bd0`) and `v2/eval/sealed/score.py` (`e0210bfb2`).
Everything ran on node A.

## 1. Gate checks (stored, sealed v3 predictions; post-key same-panel; CPU)

### (a) Human transfer vs the tier leaders

Joint paired bootstrap with per-axis intervals: 5,000 draws, same node.

| Candidate vs | Δ v3 [95% CI] | Δ H (human transfer) [95% CI] | Δ T (typed) [95% CI] |
| --- | --- | --- | --- |
| Intern-Decision 0.8B (tier leader, 43.535) | **+6.70 [+0.63, +10.35]** | +0.058 [−0.043, +0.119] | +0.078 [+0.047, +0.110] |
| Kev 0.8B (43.217) | **+7.02 [+1.33, +11.25]** | +0.050 [−0.041, +0.122] | +0.094 [+0.065, +0.123] |
| Decision 1.0 Eos (adopted, 42.547) | **+7.69 [+3.65, +13.32]** | −0.021 [−0.085, +0.088] | +0.181 [+0.149, +0.213] |

Human transfer is not significantly below the leader. Its point estimate is above
Intern and Kev, and its interval includes 0 against all three.

### (b) No decision type collapsed (typed FINAL)

A type counts as collapsed when one answer takes ≥ 90% of it, or when the lower bound of
its Wilson 95% interval is at or below chance.

| Type | Candidate accuracy [95% CI] | Chance | Top answer share | Verdict | Eos 1.0 accuracy [95% CI] | Eos top share | Eos verdict |
| --- | --- | ---: | ---: | --- | --- | ---: | --- |
| Choice (800) | .661 [.628, .693] | .267 | 22% | OK | .394 [.360, .428] | 31% | OK |
| Noul (800) | .764 [.733, .792] | .500 | 58% | OK | .512 [.478, .547] | 97% | **COLLAPSED** (97% false; at chance) |
| Score (400) | .268 [.226, .313] | .200 | 35% | OK | .300 [.257, .347] | 66% | OK |

- **Candidate Score.** It uses four of the five levels: it predicts levels 1–4 but never
  level 0 (35 gold items). Recall by level is 0 / .39 / .37 / .24 / .21.
- **Eos 1.0 Score.** Mostly level 4 (recall .77) and level 0 (.54).
- **Disclose.** Candidate Score accuracy is slightly below Eos 1.0 (.268 vs .300), and
  it never predicts level 0.

## 2. JevArena-C1 v1.1 — scoring event 1 (sealed independent set)

**Label: independent confirmation (JevArena-C1), not post-key v3.**

- **Source independence recheck** against what landed after `d8eae3e4`:
  - the A7 commit `0b47239e` (`dec10-v3`, `rec10-v2`, `enc10-v4`);
  - the AutoJev target wave `aj-m` at `3a99bf1c`;
  - the local AutoJev prompt pools (`aj-m`, `aj-a0s`).

  No name or provenance hit in 1.32M rows and 499 files. The content scan found 0
  OVERLAP and 6 weak text-level REVIEW hits, the same kinds as before. **No change, so
  there is no v1.2.**
- **Collection.** Each model was collected once on node A GPU5 with its recorded runtime.
  The adapter modules are byte-identical to the stored runs. The candidate used its
  16K package and persisted autotune cache.
- **Seals.** All predictions were sealed before the gold was decrypted. The prompts were
  on the panel root only while collection ran, and the gold only in a private temp dir.
  Both are gone. Every step is in node A's `ACCESS.log`.
- **Kev.** Its first launch failed before producing any prediction (the offline Hub
  cache was not mounted). It was collected once in a logged addendum.
- **GPU.** 0.153 GPU-hours.

### Scores

C1 is 100 × the mean task macro-F1 over 14 tasks and 2,874 items.

| Model | C1 | Choice | Noul | Score | Valid | Long-input accuracy | Non-English accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **DEV2.0-0.8B (E8F)** | **40.24** | 41.14 | 59.32 | 24.38 | 2,874 | .577 | .422 |
| Decision 1.0 Eos | 37.94 | 39.78 | 55.86 | 21.27 | 2,874 | .620 | .440 |
| Kev 0.8B | 39.06 | 39.49 | 59.62 | 22.89 | 2,872 | .523 | .457 |
| Intern-Decision 0.8B | 34.58 | 36.19 | 56.15 | 15.58 | 2,831 | .519 | .433 |
| JPT-0.8B (internal; NC licence) | 39.01 | 41.74 | 49.74 | 26.19 | 2,788 | .532 | .470 |

### Paired differences

Group bootstrap by source group within each task (5,000 draws).

| Comparison | Δ C1 [95% CI] | Δ Choice | Δ Noul | Δ Score |
| --- | --- | --- | --- | --- |
| Candidate − Eos 1.0 | **+2.31 [+0.29, +4.30]** | +1.36 [−1.30, +3.98] | +3.46 [−1.44, +8.46] | +3.10 [−0.74, +6.83] |
| Candidate − Intern-Decision | **+5.66 [+3.49, +7.78]** | **+4.94 [+1.79, +8.16]** | +3.17 [−0.62, +7.03] | **+8.79 [+4.79, +12.55]** |
| Candidate − Kev | +1.18 [−0.99, +3.37] | +1.64 [−1.64, +4.97] | −0.30 [−4.60, +3.92] | +1.48 [−2.38, +5.42] |
| Candidate − JPT-0.8B (internal) | +1.23 [−1.00, +3.43] | −0.60 [−3.98, +2.73] | **+9.58 [+5.45, +13.76]** | −1.82 [−5.79, +2.08] |
| Intern − Eos 1.0 (reference) | −3.36 [−5.30, −1.39] | −3.59 [−6.56, −0.65] | +0.29 [−4.54, +5.05] | −5.69 [−8.53, −2.67] |

### Per task (macro-F1 × 100)

| Task | Candidate | Eos 1.0 | Kev | Intern | JPT* |
| --- | ---: | ---: | ---: | ---: | ---: |
| delichess/communicative_function | 25.0 | 26.0 | 27.9 | 20.0 | 20.2 |
| delichess/epistemic_stance | 40.9 | 30.0 | 47.1 | 30.6 | 48.3 |
| hallutruthqa/find_truth (ar) | 27.7 | 30.6 | 36.8 | 41.2 | 39.7 |
| hallutruthqa/hallucination (ar) | 69.5 | 63.1 | 43.9 | 62.4 | 51.1 |
| implicaturex/likelihood | 14.7 | 3.6 | 6.5 | 6.2 | 3.6 |
| innoduel/preferred_idea (fi/en/sv/uk) | 51.7 | 55.1 | 35.4 | 44.0 | 46.6 |
| legal_case_law/argument_function | 43.8 | 45.2 | 31.7 | 24.9 | 42.3 |
| narrative_gold/event_causality | 33.8 | 25.1 | 40.8 | 27.3 | 45.1 |
| narrative_gold/setting_concreteness | 21.8 | 22.9 | 23.8 | 10.1 | 22.8 |
| narrative_gold/setting_temporal_grounding | 35.2 | 25.0 | 19.7 | 31.0 | 39.5 |
| narrative_gold/span_is_event | 54.9 | 37.8 | 65.4 | 48.4 | 55.0 |
| tutormoments/is_rapport | 53.6 | 66.7 | 69.5 | 57.6 | 43.2 |
| tutormoments/moment_type | 65.0 | 66.4 | 56.9 | 65.4 | 50.0 |
| wb_reviews/star_rating (ru) | 25.8 | 33.6 | 41.5 | 15.1 | 38.9 |

### Per language (accuracy)

| Language (items) | Candidate | Eos 1.0 | Kev | Intern | JPT* |
| --- | ---: | ---: | ---: | ---: | ---: |
| en (1,945) | .444 | .417 | .435 | .402 | .421 |
| ar (555) | .477 | .454 | .477 | .532 | .515 |
| ru (300) | .297 | .383 | .443 | .240 | .397 |
| fi (68) | .515 | .574 | .353 | .485 | .441 |
| sv (4) / uk (2) | .500 / .500 | .500 / .500 | — | .500 / .000 | .250 / .500 |

### Reading

- On the sealed independent set the candidate beats its own 1.0 model: +2.31
  [+0.29, +4.30]. The margin is much smaller than on post-key v3 (+7.69), because the v3
  gain is mostly typed reasoning, which C1 does not test.
- It clearly beats the v3 tier leader Intern-Decision, +5.66 [+3.49, +7.78].
- It is level with Kev and with JPT (the latter internal only).
- **Regressions to disclose** (all against Eos 1.0):
  - Russian star ratings, 25.8 vs 33.6;
  - tutoring-rapport Noul, 53.6 vs 66.7;
  - HalluTruthQA Choice, 27.7 vs 30.6;
  - long-input accuracy, .577 vs .620;
  - non-English accuracy, .422 vs .440, which matches the disclosed mlx-diag multilingual
    Noul loss.
- **Use budget.** 1 of the 3 C1 events is used. The C1 prompts, gold and salt were never
  exposed to any model or track outside this logged event.

Artifacts: private eval-artifacts under `m4/c1-event1/` and `m4/gates-08b/` (commit
`063ca642`: reports, seals, gold-free predictions, paired results, access log). They
contain no prompts or gold.

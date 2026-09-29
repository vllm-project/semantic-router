# JevBench public 231: is it worth keeping? (eval track, 2026-09-29)

Question (user, via the 16:05 directive 4): does JevBench have value, and why do some models (4B, 8B) look good on
JevArena but poor on JevBench? If it has no value, drop it from cards and the pipeline; if it has value, optimize
against it too.

**Decision: (a) keep it as a card metric and as a guarded optimization signal**, with the card and pipeline changes in
§6. The reproduction is faithful and the panel is reliable across model sizes. Its hard tier measures two skills that
JevArena v3 does not test, and our models are really weaker at them. But it is too small to separate models of the
same size unless the gap is at least about 10 items, so it is a non-regression guard and a skill target, never a
selection criterion.

Everything below is post-key same-panel on stored, sealed predictions. It is CPU only (0 GPU-hours), with no access
to the sealed C1 set: C1 numbers are the already-scored event 1 and 2 aggregates. Per-item data stays on node A
under `/data/dev2/private/eval/jevbench-value/`. The folder [`jevbench-value/`](jevbench-value/) holds aggregates
and the drivers as run:

| Part | Folder | Drivers |
| --- | --- | --- |
| Provenance and fidelity | [`provenance/`](jevbench-value/provenance/findings.md) | 4 |
| Reliability and validity | [`stats/`](jevbench-value/stats/summary.md) | 3 |
| 4B / 8B gap and gold audit | [`gap/`](jevbench-value/gap/summary.md) | 7 |
| Contamination | [`contamination/`](jevbench-value/contamination/summary.md) | 8 |
| Guard checks | [`guard/`](jevbench-value/guard/) | 1 |

## Plain-language answer

**English.** JevBench is useful, but small. Our 231 public questions are scored exactly as upstream scores them, and
they rank models of different sizes consistently. Between two models of the same size, though, gaps under about 10
questions are noise: 8B's 178 vs Lux 1.0's 183, and 4B's 171 vs Nox 1.0's 173, are statistically the same. Where a
gap is real (4B vs Decider 4B, −21; 27B vs Eikos-27B, −14), it sits in the hard tier. Those items apply long policy
documents, or they quote a person's plausible but wrong conclusion that the model must check instead of adopting.
JevArena never tests either skill and our training data almost never contains them. We inherited this weakness from
Decision 1.0, and our training deepened it slightly. It is not a scoring or format problem. Some peers are also
openly tuned toward JevBench.

**中文。** JevBench 有价值，但题量小。我们复现的 231 道公开题评分与上游完全一致，跨尺寸排名稳定。但同尺寸两个模型相差
约 10 题以内是噪声：8B 的 178 对 Lux 1.0 的 183、4B 的 171 对 Nox 1.0 的 173，统计上都分不开。真实的差距（4B 比
Decider 4B 少 21 题，27B 比 Eikos-27B 少 14 题）几乎全在 hard 档。这类题要么要求适用长篇政策文件，要么在题面里引用
某人看似合理但错误的结论，要求模型核对而不是照搬。JevArena 从不考这两项，我们的训练数据里也几乎没有。这个短板从
Decision 1.0 继承而来，我们的训练还略微加深了它。这不是评分或格式问题；另外有几个对手公开针对 JevBench 调优。建议
保留在卡片上（注明噪声范围、按档报告），在后继规则里加"显著退步"守卫，并用我们自己的数据补这两项能力。

## 1. What it measures

- **Source.** Upstream is `fstandhartinger/jevbench` (MIT), a one-person project. Our pin is `1bcc55eb`; upstream
  HEAD `9ec6f15a` has byte-identical public datasets and scorer. A rebuild from HEAD matches node A's prompts,
  targets and manifest byte for byte.
  - Easy 48: authored and reviewed before inference; the author is not named.
  - Standard 72: 36 hand-written paraphrase pairs from the build script.
  - Hard 111: written by two LLMs (54 by Claude Opus 5, 57 by GPT-5.6 Sol), each blind-reviewing the other's items.
    No human reviewed the gold.
- **Composition** (tier: items; types; families; input length):

  | Tier | Items | Choice / Noul / Score | Families | Median state |
  | --- | ---: | --- | --- | --- |
  | Easy | 48 | 36 / 12 / 0 | intent, fact, extraction, tool selection | 50 chars |
  | Standard | 72 | 36 / 24 / 12 (4-level) | policy, intent, ordinal, extraction, adequacy, routing | 72 chars |
  | Hard | 111 | 67 / 38 / 6 | long_policy 19, multi_hop 18, judge_hard 17, temporal_numeric 15, probability 10, trap 8, ambiguous 7, tradeoff 6, adversarial 6, routing_hard 5 | 1,611 chars |

  - 37 hard inputs exceed 4,000 characters (the longest is 15,161 characters, about 3,980 native tokens).
  - 35 states are dicts and 196 are strings; 230 of 231 prompts are English.
  - About 60 of the 111 hard items plant a person's conclusion (a note, draft recommendation or reviewer remark) that
    the evidence contradicts. A regex flags 46 of them, with precision about .93 and recall about .72.
- **Relation to the official JevBench.** The official suite has 842 decisions:

  | Part | Total | In our 231 |
  | --- | ---: | ---: |
  | Easy | 72 | 48 |
  | Standard | 96 | 72 |
  | Judge | 146 | 0 |
  | Hard | 220 | 111 |
  | Sealed (v1.4) | 308 | 0 |

  - Official Intelligence = 0.8 × chance-corrected tier score (weights hard .30, easy .14, standard .28, judge .28)
    - 0.2 × sealed. It is penalized when public accuracy exceeds sealed accuracy by more than 25 points. The composite
    is a harmonic mean of Intelligence, Calibration, Speed and Cost.
  - Our raw public accuracy equals the official "p" exactly. It feeds about 36% of the official Intelligence inputs.
    We measure no judge, held-out, sealed, calibration, speed or cost part. **Our number is not the official JevBench
    score and must never be called that.**
  - One-pass decision models collapse on the sealed set. Across 87 non-LLM rows the median sealed accuracy is 28.9%,
    and the median public-minus-sealed gap is 47 points. Public hard and held-out hard are equally difficult: upstream
    reports a field mean gap of −0.7 points, and our recomputation over 50 systems gives +0.4. So the sealed drop
    reflects a harder set, not public overfitting.
- **Relation to JevArena v3.** No v3 family or task derives from JevBench.
  - Typed FINAL states are short programmatic dicts (at most 462 characters); CSS15 is human-labelled social-science
    text.
  - JevBench-hard's long policy packets, planted conclusions, multi-hop records and exact time arithmetic have no
    counterpart in v3.
  - v3 has what JevBench lacks: oracle-verified counterfactual, order and relabel variants, 5-level Score at scale,
    real human labels, and 35 times more items.
- **Fidelity: faithful, 0 items affected.**
  - We re-scored all 86 stored runs with an upstream-faithful rule; 0 outcomes changed. Option keys, the 0.02
    tolerance, renormalization, alphabetical ties and Noul (a 0.5 tie resolves to "no") all match.
  - We have two looser checks (a missing `type`; the choice field is not validated), but neither occurs in any stored
    answer.
  - Every model receives the same `{state, questions}` object that upstream's native adapter sends.
  - 4-level Score and dict states do not single out DEV2.0: Score is 14–15 of 18, and dict states are 23 of 35, the
    same as Decider 4B.
  - DEV2.0 has 0 over-budget answers. Every invalid answer on the panel is a context overflow from a short-limit peer
    (Kai and Lex 44 each at 1,024 tokens, GLiNER2.5 56, This-That 37, Jebadiah 36), and upstream also counts these as
    wrong.
  - Where upstream measured the same systems, it agrees with us:

    | System | Ours | Upstream |
    | --- | ---: | ---: |
    | Decider 4B | 192 (48/71/73) | 193 (48/71/74), decider-4b-v2 |
    | Nimble | 185 (v2) | 184 (v1) |
    | Hopper | 194 (Hopper-G) | 190 (original) |

## 2. Reliability

(98 scored runs, 66 distinct models: [`stats/`](jevbench-value/stats/summary.md).)

- **Per-model 95% interval** (items bootstrapped within tier): the median half-width is ±11 items (range 7.5–13).
- **Informative items.**
  - Easy is at ceiling: 58 of 66 models score 48/48, and no easy item has a difficulty between .10 and .90.
  - Standard is near ceiling from 2B up (66–72 of 72).
  - About 105 items are informative, 86 of them hard. In effect it is a 111-item hard panel.
- **Split-half reliability.**
  - Across all models it is high: Pearson .93 (Spearman–Brown .96), KR-20 .96. That is mostly size spread.
  - Within tier, in pools that include peers, it is .59–.87.
  - **Among our own sibling arms it is absent:** KR-20 is −.35 to .30.
- **Test–retest.** Identical weights give 0–1 flipped items across processes, nodes and images. A context limit moves
  scores only through invalid answers (Kai 1K vs 8K: 13 items).
- **Minimum detectable difference** (exact McNemar, α .05, 80% power): the typical within-tier value is 11–20 items.
  Between 2.0 siblings it is 9–15, against a median sibling gap of 2 items; 2 of 178 sibling pairs are significant.
- **Is 178 vs 183 distinguishable? No.** DEV2.0-8B vs Lux 1.0:
  - 2 items only the candidate gets right, 7 only Lux 1.0 gets right (the same with either Lux run);
  - exact p .18, paired 95% CI −5 [−11, +1];
  - power to detect a true 5-item gap is .37, and 80% power would need about 629 items.
  - 4B vs Nox 1.0 (171 vs 173; 5 vs 7 discordant; p .77) is also noise.

## 3. Validity

(Spearman ρ with bootstrap-over-models CIs; [`stats/validity.json`](jevbench-value/stats/validity.json).)

| Criterion | n | ρ | Partial r, controlling log params | Within tier |
| --- | ---: | --- | ---: | ---: |
| v3 | 66 | .83 [.74, .89] | .29 | .27 |
| v3, family representatives | 29 | .85 [.67, .93] | — | .19 |
| v3, representatives without JPT / Hopper | 25 | .93 [.81, .97] | — | .54 |
| log10 loaded params | 66 | .91 | — | — |
| mlx-diag overall | 51 | .89 [.80, .93] | .59 | .56 |
| C1 (sealed; 0.6B/0.8B only) | 10 | .77 (r .86) | — | ρ −.10 inside the 0.8B event |

- **It is foremost a size meter.** Public 231 correlates with size (ρ .91) more than with v3 (ρ .83).
- **Given size, it tracks typed Choice.** The partial is .41–.57 for Choice and about 0 for Noul and Score. Over the
  29 representatives, T+H gives β_T .71 and β_H .17.
- **By tier.**
  - Standard is the most valid: C1 r .95, within-tier r .51 against v3.
  - Hard has no within-tier relation to v3 (r .06) and none to C1 beyond v3 (partial −.09). It measures something v3
    does not; whether that is C1-relevant is unproven.
- **Information beyond v3.**
  - Public ≈ v3 explains R² .70 (residual SD 12 items). The residual correlates with the mlx-diag residual: r .57
    [.34, .75], n 51. So public 231 and mlx-diag share size-independent signal that v3 misses.
  - With C1, the partial r(public, C1 | v3) is .77 (p .016). But it rests on Kai, Lex and GLiNER2.5, whose short
    context limits depress both scores. Without them (n 7) it is .30 (p .56), and adding public 231 worsens the
    leave-one-out fit. **No demonstrated C1 value beyond v3.**
- **Out-of-line models** (residual vs the v3 fit, all 66):
  - above the fit: JPT-4B +33, JPT-0.8B +28, 27B C0 +26; then Hopper, Intern and Decider 2B +15–16, JPT-9B +14,
    Eikos +12;
  - below it: GLiNER2.5 −32, DEV2.0-8B −19, DEV2.0-4B −17, Kai −19, Jebadiah −16.

## 4. The 4B and 8B question

([`gap/`](jevbench-value/gap/summary.md); b = items only the first model gets right, c = only the second.)

| Pair | Correct | b / c | Δ [95% CI] | Exact p | Verdict |
| --- | --- | --- | --- | --- | --- |
| DEV2.0-4B vs Nox 1.0 | 171 / 173 | 5 / 7 | −2 [−9, +5] | .77 | noise |
| DEV2.0-4B vs Decider 4B | 171 / 192 | 6 / 27 | −21 [−32, −10] | .0003 | real skill gap (23 of 27 losses hard) |
| DEV2.0-4B vs JPT-4B | 171 / 203 | 7 / 39 | −32 [−44, −20] | < .0001 | real (JPT-4B is off-card; CC BY-NC) |
| DEV2.0-8B vs Lux 1.0 | 178 / 183 | 2 / 7 | −5 [−11, +1] | .18 | noise, with a weak distribution-shift hint |
| DEV2.0-27B (F1) vs Eikos-27B | 198 / 212 | 2 / 16 | −14 [−22, −6] | .0013 | real skill gap (15 of 16 losses hard) |
| DEV2.0-27B (F1) vs AutoJev-27B | 198 / 201 | 6 / 9 | −3 [−11, +5] | .61 | noise |

- **Format and parsing artefacts: 0 items.**
  - Our candidates and the peers above have 0 invalid answers, 0 renormalized answers, 0 point-vs-argmax
    disagreements and 0 truncations.
  - Score levels match gold, and standard 4-level Score is 12/12 for everyone.
  - No model exploits the longest-option cue (at most about 2 items).
- **Named skills: the hard tier tests two things our training lacks.**
  1. **Checking a planted conclusion instead of adopting it.** About 60 of 111 hard items; scores on the 46 flagged:

     | Model | Correct of 46 |
     | --- | ---: |
     | DEV2.0-4B / Nox 1.0 | 18 / 22 |
     | Decider 4B / JPT-4B | 30 / 37 |
     | DEV2.0-8B / Lux 1.0 | 23 / 24 |
     | F1 / 27B C0 | 29 / 37 |
     | Eikos-27B / AutoJev-27B | 38 / 34 |

     - Such quotes appear in 0% of typed FINAL and at most 1% of any A7 arm's rows.
     - A7o and A7p do train short, unattributed claim-checking (8–32% of rows), but never inside long documents.
  2. **Applying long policy packets**: 8–13k-character documents with amendments, superseded editions and precedence.
     long_policy scores, out of 19:

     | Model | long_policy |
     | --- | ---: |
     | DEV2.0-4B | 3 |
     | Decider 4B | 10 |
     | JPT-4B | 14 |
     | DEV2.0-8B / Lux 1.0 | 9 / 8 |
     | F1 | 10 |
     | 27B C0 / Eikos-27B | 15 / 15 |

     Against Decider 4B, DEV2.0-4B wins 0 and loses 7 long_policy items (p .016). A7's long rows are programmatic
     tables and automata, not policy prose.
  3. **Smaller contributors:**
     - Noul yes-bias on unmet conditions: on standard + hard Noul, DEV2.0-4B gives 18 false "yes" answers, against
       Decider's 11 and Eikos's 1. It is not a threshold offset; the best threshold recovers 2 items.
     - Exact time and unit arithmetic (DST, rounding, pro-rata): every model scores 1–7 of 15.
     - For the 8B only, multi-hop chains through records (0 / 4 discordant).
- **Lineage: inherited from Decision 1.0, slightly deepened by our training.**
  - 4B: Nox 1.0 already had 59 hard and 5/19 long_policy. The m3 distillation soups took long_policy to 3–4 and
    planted-conclusion items from 22 to 18–20, and N4XF kept that, while v3 rose by 6.7.
  - 9B: multi_hop fell from 14 to 10 at the DW soup. The ⅔-Lux soup did not restore it.
  - 27B: from C0 to F1, hard fell 83 → 79, long_policy 15 → 10 and planted conclusions 37 → 29, while v3 went
    56.75 → 67.21. On planted items F1 vs C0 is 1 / 9 (Fisher p .012 against the rest). The shift is already in the
    shared m2 fine-tune mixture. **A7 is not the specific cause:** F1 (with A7) vs F2 (without) is 198 vs 194.
- **Gold audit** (the hard tier's gold was never human-reviewed).
  - We audited the 19 items fewer than 15% of models answer correctly, plus 1 item where 3 or more strong models share
    a confident wrong answer.
  - All 20 golds are clearly correct; each shared wrong answer traces to one missed step.
  - Dropping all 20 items still leaves the Decider 4B gap (−18, p .0014) and the Eikos gap (−8, p .039) significant.
- **Verdict.**
  - Against their own 1.0, the 4B and 8B gaps are **noise**.
  - Against the high-scoring peers, the gap is a **real skill our training doesn't cover**: checking planted
    conclusions and applying long policy packets. It is inherited and mildly **deepened by a distribution shift**
    (short typed states and the fine-tune mixtures), not a format artefact.
  - Part of the peers' lead is **benchmark-targeted training**:
    - Decider cards report public-hard gains between versions, and decider-4b-v2 trained on 8,000 rows generated
      from the sealed families' names;
    - Hopper's upstream row records heavy development against the public half;
    - JPT headlines its public accuracy.

## 5. Contamination

([`contamination/`](jevbench-value/contamination/summary.md).)

- **Our training rows vs public 231.** A fresh public-231-only screen covered every released model's final,
  hash-checked training file. It used `v2.data.overlap scan` (rules E/S/L/N) plus unsampled word 8/13-grams and exact
  leaves.

  | Model | File | Result |
  | --- | --- | --- |
  | 0.6B | `98e4e859` | nothing |
  | 0.8B | `d1dc33fc` | nothing |
  | 2B | `13804ac6` | nothing |
  | 27B | `de00df03` | nothing |
  | 4B | `c7d51219` | 1 new near-match group (3 HotpotQA rows vs 1 hard multi-hop item, S rule) |
  | 8B | `a66131b1` | 1 near-match group (2 generated calendar rows; the item the rescreen already flagged) |

  - **Exact overlap is 0 everywhere.** Both exposed models answer their item wrong, so the exposure changes nothing.
  - Boilerplate: one 6-word generic question wording is shared with 6 easy items; it is question text only.
  - Limits: base-model and Decision 1.0 training data are unscreened (Lux 1.0 is ⅔ of the 8B soup), and the screen is
    lexical, so paraphrases are not caught.
- **Peers.**
  - **Out-of-line scores:** only JPT-4B (+33 items, 2.5 SD) and JPT-0.8B (+30, 2.3 SD) are outliers against v3.
  - **C1 cross-check:** JPT-0.8B's sealed C1 is 8.1 points *above* its v3 prediction, the largest in the set. That is
    real strength, not memorization.
  - **Upstream board:** the public-hard vs held-out-hard check is Decider-2B +4.6, Decider-4B −1.2, Nimble −6.6 and
    Kev ≤ +6.5 points, all within noise. JPT, Eikos, AutoJev, Intern and Jet are not on the board.
  - **Model cards:** JPT, Kev and Decider say no JevBench item was used. Eikos says its eval suites were never trained
    on. Jet says "benchmarks informed training focus".
  - **Item signatures:** none. No excess on items that are hard for every other model, no excess confidence on public
    hard vs typed, and no shared idiosyncratic answers beyond our own lineage pairs.
  - **Judgement (suggestive):**

    | Level | Peers |
    | --- | --- |
    | Open, moderate | JPT-4B |
    | Low–moderate | Intern, Decider-2B |
    | Low | JPT-0.8B, Hopper, JPT-9B, Eikos |
    | No signal | Decider-4B, Nimble, AutoJev, Kev, Jet, Jebadiah |

- **Memorization probe: designed, not run.**
  - Design: an evidence-blanked rerun of the 183 standard + hard items for the suspects and our references, with a
    matched blanked typed-FINAL baseline.
  - Cost: about 0.3 GPU-h on node A.
  - It was not run for two reasons:
    - the collector accepts only registered panels, so it needs two panel-registry entries (a shared-module change);
    - no decision here depends on it: JPT is off-card, and the recommendation does not rely on peer scores being
      clean.
  - The design is in the contamination summary if the coordinator wants it.

## 6. Recommendation (a) and the exact changes

**Card changes** (release engineering, in the next card-only revision of each size, e.g., together with the 16:05
renames; `v2/release/card.py` text only, no chart code change needed):

1. Keep the score-table column "Public 231 (easy/standard/hard)" and the vs-own-1.0 per-tier rows unchanged.
2. Under the public-231 rank chart, replace the one-line note with: "JevBench public 231 is our rerun of the 231 public
   JevBench questions (about a third of the official Intelligence inputs; no sealed, judge, calibration, speed or
   cost part). It is not the official JevBench score. The easy tier is at ceiling for every model, and two totals
   within about 10 items are not distinguishable (paired 95% interval)."
3. In the disclosures, give the public-231 comparison with its paired interval from `v2.eval.gates public231`. Where a
   same-tier peer is significantly ahead, name the skills.
   - 4B: "171 vs Nox 1.0's 173 (−2 [−9, +5], within noise); below Decider 4B (−21 [−32, −10]), mostly hard-tier long
     policy documents (3 vs 10 of 19) and items that require checking a quoted person's conclusion".
   - 9B tier (DEV2.0-9B after the rename): "178 vs Lux 1.0's 183 (−5 [−11, +1], within noise)".
   - 27B: "198 vs AutoJev-27B's 201 (−3 [−11, +5]); below Eikos-27B (−14 [−22, −6]; hard 79 vs 92), in the same two
     skills".
4. Never place upstream board numbers or the official JevBench composite on our cards or charts; the Decision Index
   rule applies.

**Pipeline changes:**

1. **Successor rule item 7** (replaces "report without gating"): the successor's public 231 must not be significantly
   below the current HF revision's scored run.
   - Check: `python3 -m v2.eval.gates public231 --left <successor run> --right <current run> --left-name … --right-name
     … --output …` must not return REGRESSION (Δ < 0 items and exact two-sided McNemar p < .05). Report Δ by tier.
   - Added in `5551fb38e` with tests.
   - It reproduces the stored values, and it passes normal sibling noise:

     | Pair | Δ | p |
     | --- | ---: | ---: |
     | N5B vs N4XF | +1 | 1.0 |
     | U-a13 vs K-a13 | +2 | .63 |
     | m7-mx vs the 0.6B T soup | +11 | .07 |

2. **No selection on public 231.** Nobody selects among siblings or in development on public 231 (within-sibling
   reliability ≤ .30; median sibling gap 2 items vs MDD 9–15). Development proxy v2 is unchanged. Formal runs keep
   collecting public 231, and REPORT keeps per-tier counts.
3. **Optimize the skills, not the items.** Research & data owns the sources.
   - Add two families from independent material:
     - (i) *planted-conclusion check*: states quoting a person's note, draft or review whose conclusion the model must
       verify against the evidence. Include matched controls where the quote is right; use short and long states,
       Choice and Noul.
     - (ii) *long policy packets*: 4–15k-character policies with amendments, superseded editions and precedence or
       exception clauses.
   - Also add Noul "condition not met" hard negatives, and exact time/unit arithmetic with programmatic gold.
   - Rules for this material:
     - never JevBench items, paraphrases or upstream generator material;
     - never generate from JevBench family names (decider-4b-v2's route);
     - screen against the protected inventory (public 231 is in it);
     - no track reads public-231 item text to design data (this record's abstract skill descriptions are the brief).
   - Arms must gain on a held-out slice of the new families and pass the successor rule. Public-231 hard, especially
     long_policy and the planted-conclusion subset, is then read only formally, as the out-of-sample check.
4. **Mixture awareness.** The 27B m2 fine-tune mixture, the 4B m3 distillation soups and the 9B DW soup each moved
   long-document / multi-hop items down. Tracks should add long-prose rows to typed/A7 mixtures rather than only short
   programmatic states.
5. **Data hygiene.** Quarantine the new 4B near-match group (3 HotpotQA rows) in future mixtures; released models are
   unaffected. Keep public 231 in every protected inventory.
6. **Eval.** Gate records add the per-tier and hard-family table from `v2.eval.gates public231`. If a future claim
   needs within-tier resolution, a larger independent long-policy / planted-conclusion panel is needed: about 630
   items for 80% power on a 5-item gap.

## How it ran

- **Mirrors:** node A analysis on mirror `28f05ace7` (integration head); guard checks on mirror `5551fb38e`. Drivers
  were piped via stdin. They are committed black-formatted, and
  [`DRIVERS-AS-RUN.sha256`](jevbench-value/DRIVERS-AS-RUN.sha256) lists the SHA-256 of the bytes as run. CPU only,
  0 GPU-hours, no lease taken.
- **Local copies:** node-B 27B score/REPORT/prediction files were streamed to node A's private folder.
- **Upstream:** the MIT repo was read locally from a scratch clone at the pinned commit.
- **Privacy:** no item text, item id, gold or model answer appears in this folder; `check_no_private.sh` is clean.
- **Guard outputs** (aggregates) in [`guard/`](jevbench-value/guard/), by output SHA-256 prefix:

  | Check | SHA-256 prefix |
  | --- | --- |
  | 8B vs Lux 1.0 same-renderer | `84325125` |
  | 8B vs Lux 1.0 adopted | `8018cfc7` |
  | 4B vs Nox 1.0 | `a41518c8` |
  | 4B vs Decider 4B | `2ee16403` |
  | N5B vs N4XF | `48a5fe39` |
  | U-a13 vs K-a13 | `c1bc0f0b` |
  | m7-mx vs T soup | `96733997` |

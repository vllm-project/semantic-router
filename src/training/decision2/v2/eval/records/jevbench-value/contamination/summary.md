# JevBench public 231: contamination (eval & peers, 2026-09-29)

Scope: our reproduction of the 231 public JevBench items (`fstandhartinger/jevbench@1bcc55eb`; easy 48 / standard 72 /
hard 111; gold-free prompts `642d3fac…`, identical on both nodes). CPU only, 0 GPU-hours, no access to the sealed
directory. This file holds aggregates only. Per-item outputs and training-row ids are on node A and node B under
`/data/dev2/private/eval/jevbench-value/contamination/` (0700). The numbers are in
[`contamination.json`](contamination.json) and the drivers in [`drivers/`](drivers/).

**Bottom line.**

- None of the six released models' final training files contains an exact copy of a public-231 item.
- A fresh public-231-only screen finds two single-item near matches: 4B (3 rows) and 8B (2 rows). The exposed model
  answered each of those items wrong, so neither match can have inflated a released score.
- No peer shows a memorization signature. JPT-4B and JPT-0.8B sit far above the public-vs-v3 line. The item-level
  and upstream evidence does not support training on public items, but it cannot rule it out.

## A. Our training rows vs the public 231

### Training files of the released models (hash-checked on the node that stores each file)

| Model | Recipe | Final training file | Rows | Node |
| --- | --- | --- | --- | --- |
| DEV2.0-0.6B | 06b `m4-t-a7-soup` | `m4-mix-t.train.jsonl` `98e4e859` | 26,203 | A |
| DEV2.0-0.8B | E8F (A7 + v1) | `m2-full-a7-v1/train.jsonl` `d1dc33fc` | 162,777 | A |
| DEV2.0-2B | S2T | `m3-v2m-ret/train.jsonl` `13804ac6` | 56,198 | B |
| DEV2.0-4B | N4XF soup | `m4-xl-full-29m/train.jsonl` `c7d51219` | 58,742 | B |
| DEV2.0-8B | K-a13 soup (x60) | `m4-k-xl-r2-60m/build/train.jsonl` `a66131b1` | 122,651 | A |
| DEV2.0-27B | F1 = M3-A soup | `mixtures-m3-1/a7.train.jsonl` `de00df03` | 25,822 | B |

Teacher pools score the same rows: the 8B's `teacher.jsonl` holds own-Lux 1.0 soft targets for its `train.jsonl`
rows, and the 0.6B's Eos / Lux targets are on its own rows. Their prompts are therefore the training inputs screened
here.

### Earlier screens

- **Pool-level screens with JevBench quarantines.**
  - arms-v1 dropped 2 JevBench groups and quarantined 6 rows plus 1 group on 13-grams against JevBench public
    ([`arms-v1-2026-09-28.md`](../../../../data/records/arms-v1-2026-09-28.md)).
  - The H7 / H8 union screen removed 1 group for "JevBench-231"
    ([`m3b-gap-sources-2026-09-28.md`](../../../../data/records/m3b-gap-sources-2026-09-28.md)).
- **The r2 rescreen.** It excluded 305 groups (PI-v4, including the `jevbench_public231` roles); 2 G6 groups touch
  1 hard public item. The gate records show that no released training file contains any of those G6 groups:
  - 0.6B, 4B, 8B and 27B contain 0 of the 305 groups;
  - 0.8B contains 33 and 2B contains 24, none of them touching public 231
    ([`m5-overlap-effects-2026-09-29.md`](../../m5-overlap-effects-2026-09-29.md) and the 4B / 9B / 27B gate records).
- **What the rescreen covered.** It screened the source pools, not every final file. For example, the 4B and 8B H7 /
  H8 rows had only their build-time screen, and 101 of the 27B's A7g rows had only the A7 v3 screen.

### Fresh public-231-only screen of each final file

The screen ran `v2.data.overlap scan`, with rules E / S / L / N, against a one-role inventory built from the gold-free
prompts. A supplemental check used unsampled word 8- and 13-gram containment plus exact normalized-leaf equality;
grams that appear in 5 or more public items count as template text.

| Model | E | S | L | N | Quarantine groups (rows) | Boilerplate E-leaf groups | Items with any 13-gram | Max 8-gram containment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.6B | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.8B | 0 | 0 | 0 | 0 | 0 | 97 | 0 | 0 |
| 2B | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4B | 0 | **1** | 0 | 0 | **1 (3)** | 16 | 0 | 0 |
| 8B | 0 | 0 | 0 | **1** | **1 (2)** | 25 | 0 | 0.004 |
| 27B | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

- **4B, S rule.** One HotpotQA-distractor-train group (`hotpotqa_coverage`, Score; 1 of its 3 rows hit) nearly
  duplicates a short leaf of one hard `multi_hop` Choice item.
  - This group is not among the rescreen's 305.
  - DEV2.0-4B answers the item wrong, as do 0.6B–8B. Of the 29 models scored, 12 answer it, including DEV2.0-27B,
    which has no overlap with it.
- **8B, N rule.** Two generated `a2_calendar` Noul rows share one sampled 30-character gram with the
  rescreen-flagged hard item. That is the only link between them.
- **Boilerplate matches** (0.8B / 4B / 8B). One 6-word generic question wording appears in 6 easy items and in SGD
  Choice instructions. It is question text only (no state, no answer) and never quarantines.
- **Post-quarantine overlap.**
  - Exact overlap is 0 in every file.
  - Under the program's rules, 0.6B, 0.8B, 2B and 27B are 0. The 4B and the 8B each have 1 near-match group.
  - Neither exposed model got its item right, so the effect on any released public score is 0 items.

**The single flagged item.** No released model trained on its G6 groups. DEV2.0-0.6B, 0.8B, 2B, 4B and 8B all
miss it. DEV2.0-27B, whose training data has no overlap with it, answers it. Removing it moves every model's public
total by at most 1 item.

**Limitations.**

- The pretraining and Decision 1.0 training data are unscreened. Lux 1.0 is ⅔ of the 8B soup's weights; Nox, Sol
  and Eos start the 4B, 2B and 0.8B.
- Program generators (A2 verifiable, A7 Stage 1–4) are covered only through the rows they emitted.
- The screens are lexical and do not catch paraphrases.
- Mitigation for the Decision 1.0 gap: those models show no public over-performance. Their residuals against v3
  are −18.9 to +8.8 items, all within 1 SD.

## B. Do high-scoring peers look trained on public items?

**Public total vs post-key v3** (29 models, OLS: public = 60.7 + 2.01·v3, σ = 14.6 items, R² = .72; the
leave-one-out studentized residual is in parentheses). The table also shows item signatures from the stored
per-item scores:

- **Hard-for-everyone:** hard items that at most 20% of the other 28 models answer, observed hits vs a Rasch
  expectation (number of such items in parentheses).
- **Share of correct answers at confidence ≥ .95** (public hard / typed FINAL / CSS15), using the max option
  probability. Typed accuracy recomputed with the program's scorer equals `typed-final.score.json` for all 29 models.
- **Accuracy among ≥ .95 answers** (public hard / typed FINAL).

| Model | Public resid. | Hard resid. | Hard-for-everyone obs / exp | Correct at ≥ .95 | Acc at ≥ .95 |
| --- | --- | --- | --- | --- | --- |
| JPT-4B | +32.7 (+2.49) | +27.9 | 9 / 6.5 (20) | 0.10 / 0.56 / 0.11 | 0.90 / 0.96 |
| JPT-0.8B | +29.8 (+2.28) | +23.4 | 4 / 1.6 (19) | 0.00 / 0.00 / 0.02 | — / 1.00 |
| Hopper-G | +16.0 (+1.12) | +12.0 | 3 / 3.8 (18) | 0.12 / 0.24 / 0.07 | 1.00 / 1.00 |
| Intern | +15.9 (+1.12) | +15.0 | 3 / 1.4 (20) | 0.00 / 0.00 / 0.00 | — / — |
| Decider-2B | +14.9 (+1.04) | +10.4 | 5 / 2.1 (20) | 0.25 / 0.05 / 0.14 | 1.00 / 0.91 |
| JPT-9B | +13.8 (+0.97) | +13.7 | 4 / 4.7 (19) | 0.14 / 0.59 / 0.12 | 1.00 / 0.96 |
| Eikos-27B | +12.1 (+0.87) | +14.1 | 9 / 10.0 (21) | 0.20 / 0.31 / 0.00 | 1.00 / 1.00 |
| Sol1 | +8.8 (+0.61) | −0.6 | 2 / 1.1 (19) | 0.09 / 0.61 / 0.07 | 0.67 / 0.72 |
| Decider-4B | +7.0 (+0.49) | +4.5 | 3 / 3.9 (19) | 0.34 / 0.71 / 0.12 | 0.89 / 0.99 |
| Bosun-1.7B | +5.7 (+0.40) | −0.2 | 1 / 0.7 (19) | 0.23 / 0.11 / 0.16 | 0.83 / 0.83 |
| DEV2.0-2B | +3.0 (+0.20) | −0.6 | 0 / 1.5 (18) | 0.05 / 0.54 / 0.09 | 0.75 / 0.67 |
| DEV2.0-27B | +2.3 (+0.16) | +3.7 | 3 / 4.9 (19) | 0.23 / 0.68 / 0.02 | 1.00 / 0.97 |
| Nimble-v2 | −0.3 (−0.02) | +0.3 | 1 / 2.6 (18) | 0.09 / 0.42 / 0.02 | 1.00 / 1.00 |
| Kev | −0.5 (−0.03) | −3.6 | 3 / 0.7 (20) | 0.00 / 0.00 / 0.00 | — / — |
| Nox1 | −1.1 (−0.08) | −2.5 | 0 / 1.6 (18) | 0.24 / 0.66 / 0.16 | 0.88 / 0.94 |
| Eos1 | −4.2 (−0.29) | −4.7 | 1 / 0.5 (18) | 0.03 / 0.14 / 0.06 | 1.00 / 0.98 |
| AutoJev-27B | −4.6 (−0.33) | −0.6 | 5 / 6.0 (20) | 0.25 / 0.38 / 0.13 | 1.00 / 1.00 |
| Bosun-0.6B | −5.1 (−0.36) | −2.6 | 1 / 0.4 (19) | 0.03 / 0.00 / 0.13 | 0.25 / 1.00 |
| DEV2.0-0.8B | −5.6 (−0.38) | −8.6 | 2 / 0.8 (18) | 0.09 / 0.34 / 0.08 | 1.00 / 0.80 |
| DEV2.0-0.6B | −6.1 (−0.43) | −5.0 | 2 / 0.5 (19) | 0.00 / 0.00 / 0.02 | — / — |
| This-That | −6.3 (−0.43) | −13.3 | 0 / 0.6 (18) | 0.77 / 0.44 / 0.37 | 0.57 / 0.72 |
| Jet-v6.2 | −8.0 (−0.55) | −9.5 | 3 / 1.9 (19) | 0.07 / 0.24 / 0.09 | 0.50 / 1.00 |
| Lux1 | −9.9 (−0.70) | −5.5 | 0 / 2.4 (18) | 0.18 / 0.57 / 0.17 | 1.00 / 1.00 |
| Lex1 | −10.0 (−0.74) | +0.0 | 1 / 0.2 (18) | 0.00 / 0.00 / 0.00 | — / — |
| Jebadiah-27B | −16.2 (−1.16) | −15.0 | 6 / 2.2 (20) | 0.05 / 0.25 / 0.05 | 1.00 / 1.00 |
| DEV2.0-4B | −16.5 (−1.17) | −14.1 | 1 / 1.5 (18) | 0.12 / 0.56 / 0.07 | 0.88 / 0.97 |
| DEV2.0-8B | −18.7 (−1.37) | −12.0 | 1 / 2.0 (18) | 0.08 / 0.60 / 0.03 | 1.00 / 0.98 |
| Kai1 | −18.9 (−1.39) | −5.3 | 1 / 0.2 (18) | 0.00 / 0.21 / 0.01 | — / 0.55 |
| GLiNER2.5-Decide | −30.1 (−2.29) | −21.7 | 1 / 0.2 (18) | 0.00 / 0.13 / 0.03 | 0.00 / 0.79 |

- **Hard-for-everyone.** No model is far above its Rasch expectation. The largest z values (Kev 2.8, Jebadiah 2.7,
  Decider-2B 2.1, DEV2.0-0.6B 2.1) come from expected counts of 0.5–2 hits, which is noise over 29 tests.
- **Confidence.** No suspect shows memorization-like overconfidence on public hard items: for most peers, the share
  of correct hard answers at ≥ .95 is below their typed-FINAL share. The exceptions:
  - Decider-2B: .25 vs .05, a mild and suggestive signal;
  - Bosun-1.7B: .23 vs .11;
  - This-That: overconfident on every panel.
  - JPT-0.8B and Intern never reach .95.
- **Agreement.** The residual correlation is highest for same-lineage pairs:
  - DEV2.0-8B / Lux1 .64, DEV2.0-4B / Nox1 .60, JPT-4B / JPT-9B .48 (99.5th percentile of 406 pairs),
    AutoJev-27B / DEV2.0-27B .39.
  - JPT-4B / JPT-9B shares 13 rare correct answers against 9.2 expected (ratio 1.4), in line with our own
    lineage pairs.
  - JPT-0.8B does not share idiosyncratic answers with its siblings: .07 with JPT-4B and −.04 with JPT-9B.
- **C1 cross-check** (10 models with sealed C1 scores). JPT-0.8B's C1 is +8.1 above its v3 prediction, the
  largest positive residual. Against its public prediction it is −3.9, within the spread of Intern (−5.9) and
  Kai (−5.3). Its public lead over Eos (171 vs 142) comes with a small C1 lead (39.01 vs 37.94). A model that owed
  its public score to memorization would not be expected to be this strong on held-out human-labelled data.

**Upstream JevBench board** (@1bcc55eb):

- **Public vs sealed.** 74 of the 91 systems with sealed scores exceed the 25 pp public-to-sealed gap (median
  46 pp; untrained raw-logit controls 22–42 pp; frontier LLMs mostly 3–8 pp). Peers: Decider 4B v2 48.8, Decider 2B
  46.3, Hopper 48.2, Nimble 9B 50.8, Kev 43–50, Jev 49.9. The gap reflects the shift to the fresh sealed set, so it
  does not identify contamination.
- **Same-distribution check** (111 public vs 109 held-out hard items, v1.2 per-task, 50 valid systems): mean
  +0.4 pp, SD 7.3 pp.
  - Decider-2B +4.6 (z 0.7), Decider-4B v2 −1.2 (v1.4.2), Nimble 9B −6.6, Kev −10.8 to +6.5.
  - No peer's public hard accuracy exceeds its held-out hard accuracy beyond noise.
- **Not on the board:** JPT, Intern, Eikos, AutoJev, Jet, Bosun, This-That, GLiNER2.5-Decide, Jebadiah.

**Card declarations** (pinned revisions, fetched on node A):

- **JPT:** "no item from JevBench … held out"; the cards headline public accuracy.
- **Kev:** public items seen by "no training or selection step".
- **Decider 2B / 4B:** no JevBench item used for training, generators, selection or temperatures. Both cards still
  report version-over-version public-hard gains (+13 / +11 items).
- **Eikos:** its evaluation suites were "never used in training".
- **Hopper (G):** "not tuned for JevBench".
- **Jet:** "Benchmarks informed training focus"; says exclusions ≠ decontamination.
- **AutoJev:** its training corpus is not released.
- **Intern, Nimble, Jebadiah, Bosun, GLiNER2.5-Decide, This-That:** no statement either way.

**Judgement per suspect** (suggestive, not proof):

| Peer | Judgement | Evidence |
| --- | --- | --- |
| JPT-4B | Open, moderate | Largest public outlier; the card denies training on JevBench; no memorization signature; no sealed or C1 score. Public accuracy is the card's headline, so there is selection pressure. |
| JPT-0.8B | Low | Also strong on C1; never answers at ≥ .95. |
| Intern | Low–moderate | +16 on public but −5.9 vs its public prediction on C1; no declaration. |
| Decider-2B | Low–moderate | Mild confidence excess on public hard; upstream held-out hard is clean; public tracked across versions. |
| Hopper-G, JPT-9B, Eikos-27B | Low | Residuals within 1.2 SD. |
| Decider-4B, Nimble, AutoJev, Kev, Jet, Jebadiah | None | No signal on any check. |

## C. Cheap memorization probe (design only; not run)

**Panels.**

- `probe-p231-noev`: public standard + hard (183 items), with the state replaced by a fixed placeholder and
  questions / criteria / option order kept.
- `probe-typed-noev`: 183 typed FINAL items, seeded and stratified by family × type, with the same state
  replacement.

**Models.**

- Suspects: JPT-0.8B, JPT-4B, JPT-9B, Intern, Decider-4B, Hopper-G, Eikos-27B.
- Unexposed references: DEV2.0-0.8B, DEV2.0-4B, DEV2.0-8B.
- Decision 1.0: Eos1, Nox1, Lux1.
- Declared-clean control: Kev.
- All weights are on node A: `/data/dev2/models/*` for the JPT-0.8B / 4B, Intern, Hopper-G, Kev and Eikos-27B
  packages; `/data/decision20-20260926/{competitors/jpt-9b-r7114b0c3, models/decider-4b, models/Decision-1.0-*}`;
  DEV2 checkpoints as in their `COLLECT.json`.
- Not local: Decider-2B (would need an `hf download`) and AutoJev / Jebadiah (node B only).

**Readout.**

- For each model: evidence-free accuracy on each panel, and retention (the share of full-evidence-correct items
  still correct without evidence).
- Memorization index = (public noev − typed noev) minus the same difference for the unexposed DEV2 references,
  from a paired item bootstrap (5,000 draws). Flag a model when the lower bound exceeds +10 pp.

**Cost.** About 0.3 GPU-h for all 14 models, capped at 0.5 GPU-h. Prior walls were 18–46 s for public 231 per
≤ 9B model and 94 s for Eikos; typed FINAL took 52–129 s per 1,600 rows. The probe has 183 + 183 shorter rows plus
8-item smokes. Run on GPU0 / GPU1 under the eval entry (`--shared-lease eval`) with coordinator approval.

**Mechanism.** No existing mechanism supports a custom panel without code changes:

- `same_panel collect` rejects any panel whose prompts hash is not in `panels.ALL`.
- `run_same_panel.sh` mounts only the frozen gold-free root.

Needed: two `panels.ALL` entries (prompts, `prompts_sha256`, originals, gold path), committed, pushed and mirrored.
The commands would then be:

```bash
SRC=<probe-sha>-src_training_decision2; S=/data/dev2/src/$SRC/src/training/decision2; R=/data/dev2/runs/eval/jevbench-value/probe
jq .adapter <prior-run>/COLLECT.json > $R/<model>.adapter.json          # reuse the exact adapter block of the formal run
$S/v2/eval/run_same_panel.sh --gpu 0 --track eval --src $SRC --shared-lease eval --run-dir $R/<model> --model-dir <weights> \
  [--mount /data/dev2/hf-cache --env HF_HUB_CACHE=/data/dev2/hf-cache --env HF_HUB_OFFLINE=1] \
  -- --adapter-spec $R/<model>.adapter.json --model-path <weights> --revision <rev> [--extra source=<src>] \
     --panels probe-p231-noev,probe-typed-noev --max-items 8 --reason "jevbench-value memorization probe (smoke)"
# then the same command without --max-items; score both panels on CPU against the registry gold
```

Without the registry change, the only route is running each adapter module (for example `python3 -m inference.jpt
--input … --output …`) directly in the pinned image, which produces no collector receipts.

# Sealed confirmation set (JevArena-C1) — design only

Eval & peers track, Milestone 1 item 6. Status: **design, not built.** JevArena
v3 answers have been read by the project, so every v3 number is post-key
same-panel evidence. Any claim of *independent* validation needs a set whose
sources, items and labels no project member or pipeline has touched. Building
it requires paid, independent human annotation, which is not cheap, so this
milestone only fixes the protocol.

## 1. Sources never touched by the project

Eligible sources must be absent from every project artifact, checked before
any item is written:

- all v1–v3 panels (typed DEV/FINAL generators and seeds, the 18 CSS tasks,
  pressure panels, authored v3–v13 pools, JevBench public 231, Decision Bench
  v4, the Decision Index 0.2.1 corpora);
- every TRAIN/SELECT/CAL source, teacher corpus and data arm in the private
  training dataset and the research & data track's registries;
- documented training corpora of the peers we compare against (for example the
  public data lists of Kev, Decider and AutoJev), where published.

Admissible classes, in order of preference:

1. **Commissioned originals**: decisions written by independent contractors who
   have not seen project data, from a frozen brief (domains, types, length and
   language quotas), created after the freeze date.
2. **Time-split real records**: human-labelled records first published after the
   freeze date (new annotation releases, or a post-cutoff time window of an
   existing public source) whose licence permits private evaluation use.
3. **Licensed private partner records**: stay in the private environment and
   never enter HF, the gist or any card.

Every source gets a rights record (licence, redistribution scope, retention)
before collection; restricted text stays under `/data/dev2/private/`.

## 2. Size, types and strata

Sized so that a paired v3-style difference of about 2 points has a 95%
interval half-width near 2 points; the task-level resampling that dominates
v3's current ±4–5 point intervals needs more tasks, not only more items.

| Stratum | Items | Structure |
| --- | ---: | --- |
| Typed Choice / Noul / Score | 400 / 400 / 400 | at least 300 independent groups; Score with 3–7 ordered levels, level frequencies balanced within ±10% |
| Human-judgement transfer | 20 tasks × 180 = 3,600 | new tasks only; 2–6 labels each; macro-F1 per task, median across tasks |
| Multilingual | 4 languages × 300 = 1,200 | natively written zh, es, ja, de (no machine translation); mixed types |
| Long input | ≥ 20% of all items | complete input ≥ 4,000 characters, up to the longest native limit among compared models |
| Robustness pairs | 300 pairs | order, label-name and counterfactual pairs, reported separately, never counted as extra items |

About 6,400 original items in total, plus the paired variants.

## 3. Label protocol

- Rubrics and answer formats frozen and hashed before annotation.
- Two independent annotators per item, blind to each other and to any model
  output; a third adjudicates disagreements. Admission requires Cohen's κ ≥ 0.6
  (Choice, Noul) or quadratic-weighted κ ≥ 0.6 (Score) per task; items still
  ambiguous after adjudication are dropped before sealing.
- Annotators are not members of any Decision 2.0 track and do not see training
  data or model outputs.
- Gold-free audits before sealing: exact and MinHash (≥ 0.8) near-duplicate scan
  against all artifacts in §1; hypothesis-only / option-only / length baselines
  must stay within majority + 5 points per task; per-language and per-length
  answer-position balance.
- Jev, peer or Decision model outputs must never be used to draft, filter or
  label items.

## 4. Sealing

- A custodian outside the model tracks holds the labels encrypted at rest; only
  gold-free prompts are released to the runner.
- Before any model sees a prompt, the gist records SHA-256 commitments of the
  prompts, labels, rubrics, audit receipts and the scoring code revision.
- Access to labels is logged; any label access before a scoring event retires
  the affected items.

## 5. One-shot scoring

- Only frozen, hash-pinned release candidates (plus the same-tier 1.0 model and
  at most two qualified open peers per tier) are scored, using the existing
  runner (`v2/eval/same_panel.py`) with this panel registered as a new frozen
  panel. Predictions are sealed before the custodian decrypts labels.
- Each package is scored exactly once. No checkpoint selection, calibration,
  threshold or prompt change may use this set; results are reported whatever
  their direction.
- Use budget: at most one release candidate per size and three scoring events in
  total; afterwards the set is declared post-key and a successor set is built.
- Reported as "independent confirmation (JevArena-C1)" with paired intervals,
  separately from the post-key v3 table.

## 6. Cost and next step

Rough effort: about 6,400 items × 2 annotations × 2–4 minutes plus
adjudication of 15–25%, on the order of 500–900 annotator-hours, plus source
licensing. Decision needed from the user: budget and vendor for independent
annotators and a custodian. Until then no independent-validation claim can be
made; post-key v3 plus JevBench public 231 remain the evaluation basis.

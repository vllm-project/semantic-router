# JevArena: one development/release decision evaluation protocol

JevArena runs every eligible model through its **native** inference path on
the same frozen prompts. It records exact model and data revisions, keeps gold
out of inference, counts missing or invalid answers as wrong, and publishes
task/type breakdowns beside the rank. Native models that cannot answer one of
Choice, Noul or ordinal Score stay visible with explicit unsupported coverage;
they are not quietly assigned a better eligible-only score.

## Panels and roles

| Panel | Development | Release | Headline metric |
| --- | --- | --- | --- |
| Typed synthetic | frozen 1,600-item DEV | fresh independent 1,600-item FINAL | family-macro accuracy, Choice/Noul/Score and calibration breakdown |
| Human transfer | three 1,430-item CSS pilot tasks | 15 held-out CSS tasks, 6,547 items | median task macro-F1, task-level accuracy/calibration |
| Authored reasoning | pinned 231 public JevBench items | same public items | equal easy/standard/hard accuracy, plus micro, Brier and ECE |
| Perturbation | paired counterfactual, order and label variants in synthetic panel | independently seeded FINAL pairs | average joint-correct fraction across the three relations |

The RQ1/RQ2/RQ3 pressure panel is a development diagnostic shown separately:
its source has a failing text-blind shortcut gate, so it is not in the scalar.
Latency, cost, validity, Brier and ECE are mandatory side tables. Latency and
cost require matched hardware and service conditions to compare. Any 42
previously flagged near-neighbor CSS evaluation rows are also reported as a
separate 6,505-item clean-mask sensitivity view; neither its membership nor
the full-panel headline is changed after seeing final labels.

For each phase, `arena.py` gives all four axes **equal weight in a geometric
mean**, multiplied by 100. An absent or zero axis results in zero. This rule
was fixed before opening fresh FINAL gold. The geometric mean penalizes a
model that improves synthetic rules while losing human transfer or authored
reasoning. The rank includes only same-panel reruns; it does not paste
historical scores from different benchmark versions or hardware. Pareto uses
actual parameter count and this same JevArena score; unknown-size hosted
systems appear in the rank but not on the size frontier. Report each axis and
paired 2.0-versus-1.0 confidence intervals as well as the scalar.

## Public JevBench reproduction

`jevbench_public.py` rebuilds exactly 48 easy, 72 standard and 111 hard
public items from [`fstandhartinger/jevbench`](https://github.com/fstandhartinger/jevbench)
revision `1bcc55eb6c8cffde2306b3db03ede39b61c6152a`. Each source JSONL
has a pinned SHA-256. It writes gold-free `prompts.jsonl`, separate
`targets.jsonl`, and a manifest. Native collector receipts must bind every
item input hash, model ID (or the explicit absent-ID sentinel for native
Decider), and immutable model revision. When identity lives in a companion
manifest rather than every prediction row, pass it explicitly with
`--prediction-manifest`; the scorer verifies prompt/prediction digests and
each row's model and adapter hashes. Scoring follows JevBench's exact
option-key validation, finite probabilities, 0.02 rounding tolerance,
renormalization and alphabetical argmax ties; it also reports the original
0.001 strict-valid rate. Missing answers count wrong. The report separates
all three tiers, Brier, ECE and latency; the arena axis averages tier
accuracies so 111 hard questions do not silently dominate 48 easy questions.

This is an **independent public-only JevBench ranking**. Upstream v1.2 and
v1.4.2 full leaderboards include private or sealed questions and combine
intelligence with calibration, speed or cost. The public-only rank and Pareto
cannot be called official JevBench scores, and historical composite figures
cannot be placed on the same numerical axis. Public items are exposed and
may be inspected during development; FINAL synthetic and CSS labels remain
unseen until candidate selection is frozen.

Run the builder and scorer on an authorized GPU experiment host:

```bash
PYTHONPATH=/work/source python3 -m jev_arena.jevbench_public build \
  --upstream-root /work/external/jevbench --output-dir /work/bench/jevbench-public-231

PYTHONPATH=/work/source python3 -m jev_arena.jevbench_public score \
  --panel-dir /work/bench/jevbench-public-231 \
  --predictions /work/bench/jevbench-public-231/model.predictions.jsonl \
  --model-id llm-semantic-router/dev-2.0-4b \
  --model-revision IMMUTABLE_REVISION \
  --output /work/bench/jevbench-public-231/model.score.json
```

No artifact containing training text, benchmark gold, model weights or
credentials belongs in this source tree. The private Hugging Face dataset
stores the distributable training split and lineage/rights manifests.

## JevArena v2 release protocol

`arena_v2.py` freezes a six-axis, equal-weight geometric mean before any
sealed FINAL label is used. The axes are typed synthetic family-macro
accuracy, held-out human-transfer task-median macro-F1, public JevBench
tier-macro accuracy, Decision Bench v4 text-readable task-macro accuracy,
independently authored Choice/Noul/Score family-macro accuracy, and paired
robustness joint correctness. A zero axis gives a zero aggregate. Invalid or
missing answers count wrong in the upstream scorers. Calibration, validity,
latency, throughput and cost remain mandatory separate results. Paired
variants are not counted as additional independent questions.

The release roster requires the same 1,600-item typed FINAL, 6,547-item
15-task CSS evaluation, 231 public JevBench items, 1,041 eligible Decision
Bench v4 cases with 30 visual-only N/E, and 1,200–1,480 independently
authored sealed items for **every** model. That totals 10,619–10,899
effective text answers per model before paired variants. A smaller,
independently seeded 100–250-item authored DEV panel pairs with typed DEV,
CSS pilot and the two public panels. All panel prompt/target/gold hashes must
match across a rank roster. The authored axis is unavailable until its source,
oracle, ambiguity, human review and training-overlap quality gate passes.
The release aggregate cannot be produced from an incomplete roster.

The v1 four-axis development ranking above is an earlier protocol. It can
guide exploration but its scalar is not interchangeable with v2. The
authored builder/scorer, pretest freeze and full-panel release reports are
required before any v2 result is publishable.

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
The ranker carries individual Choice/Noul/Score accuracy and each human
transfer task's macro-F1 alongside its six scalar axes. `render.py` emits
rank, actual-parameter-count Pareto, six-axis matrix, and model-by-task
matrix SVGs from the same ranked report. The task matrix names the different
cell metrics and exposes every transfer task rather than only their median.

The v1 four-axis development ranking above is an earlier protocol. It can
guide exploration but its scalar is not interchangeable with v2. The
authored builder/scorer, pretest freeze and full-panel release reports are
required before any v2 result is publishable.

## Authored v9 editorial pilot

`authored_v9_pilot.py` builds a twelve-item **DEV-only** Choice/Noul/Score
packet from private hand-authored scenarios and a private salt. It renders
independent signed evidence documents, parses their attested fields back from
the visible prompt, and checks the current-rule answer against a separate
reference implementation. Its three long cases use different source joins:
an accepted revision, an entity alias, and an effective-time event log.
Every claimed essential document is removed in a gold-free ablation; the
remaining evidence must leave a policy field unproved, and a domain-valid
alternative completion must change the answer. This mechanical test cannot
certify that prose lacks shortcuts or that a dossier is editorially strong.

The builder writes `prompts.jsonl` and `ablations.gold-free.jsonl` outside the
source tree, plus `private/targets.jsonl`, `private/proof_traces.jsonl` and a
private audit receipt. The public-safe receipt records hashes, type and
mechanism counts, answer-position counts, and long-context lengths. It never
marks a packet release qualified. Source scenarios, salt, targets and proofs
stay in the authorized private experiment workspace. No v9 item may enter a
FINAL benchmark or training set from automated proof alone. An independent
reviewer must seal answers to prompts, then seal a separate ablation review,
before any post-key comparison. Failed frozen versions remain as evidence of
the design revisions rather than being overwritten.

```bash
PYTHONPATH=/work/source python3 -m jev_arena.authored_v9_pilot \
  --specs /work/private/authored-specs.json \
  --private-salt /work/private/authored-salt.bin \
  --output-dir /work/private/authored-v9-dev12
```

## Authored release v2 blind human review

After a mechanically valid v2 packet is sealed, assign its independently
salted `original-review-a`, `original-review-b` and `paired-review` files to
three different human reviewers. Keep all files in private storage. Reviewers
see only their own gold-free packet and editable form, never the oracle,
parent joins, proofs, model output or another review. The form requires a
native typed answer, evidence citations from both sources, paragraph-level
notes, and explicit ambiguity, necessity, realism, shortcut and rights
judgments. A negative quality judgment is recorded, not converted into a
passing review. For the long case, the reviewer must inspect every paragraph.

`authored_release_scale_v2_review.py` creates a private form and seals each
completed form with the packet hash, reviewer identity digest and UTC time.
It validates native answer types and complete row coverage without opening a
key. A separate adjudicator must verify that the three reviewers are human,
independent of the author and one another, and that their reviews were sealed
before any key was opened. No automatic seal constitutes release approval.

```bash
PYTHONPATH=/work/source python3 -m jev_arena.authored_release_scale_v2_review template \
  --packet /work/private/r4-packets/original-review-a.private.jsonl \
  --packet-receipt /work/private/r4-packets/receipt.private.json \
  --role original-review-a --output /work/private/review-a-form.private.jsonl

PYTHONPATH=/work/source python3 -m jev_arena.authored_release_scale_v2_review seal \
  --packet /work/private/r4-packets/original-review-a.private.jsonl \
  --packet-receipt /work/private/r4-packets/receipt.private.json \
  --role original-review-a \
  --answers /work/private/review-a-form.private.jsonl \
  --reviewer-id-file /work/private/reviewer-a-id.private.txt \
  --output /work/private/review-a-sealed
```

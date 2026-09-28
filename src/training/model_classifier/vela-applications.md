# Vela application classifier recipes

This guide covers Feedback, Guard, Safety, and Hazard. A recipe is a
reproducible experiment; it does not certify a particular checkpoint. Model cards introduce the task, intended use, and place in the Vela family.
Keep dataset revisions, training lineage, and detailed evaluations with the
release evidence. Model cards summarize measured capabilities and material
limitations without embedding experiment logs.

## Output contracts

| Task | Ordered labels | Output |
|---|---|---|
| Feedback | `SAT`, `NEED_CLARIFICATION`, `WRONG_ANSWER`, `WANT_DIFFERENT`, `NO_FEEDBACK` | Softmax |
| Guard | `benign`, `jailbreak` | Softmax |
| Safety | `safe`, `unsafe` | Softmax |
| Hazard | `violence`, `criminal_activity`, `sexual_content`, `child_exploitation`, `hate`, `harassment_abuse`, `regulated_substances`, `weapons`, `self_harm`, `privacy`, `specialized_advice`, `misinformation` | Independent sigmoid |

All four use standard `ModernBertForSequenceClassification`. Hazard sets
`problem_type=multi_label_classification`; the other three set
`single_label_classification`. Every config must have an exact inverse
`id2label`/`label2id` mapping. A Hazard request matching any selected category
uses independent scores; adding probabilities from separate labels is invalid.

Feedback receives the current user follow-up, not an inferred conversation.
Satisfaction must describe that user's reaction to a preceding answer. An
assistant explanation or a neutral factual statement is not positive feedback.
Clarification asks to understand the existing answer; a different format or
approach is a revision request. Ambiguous utterances cannot be made unambiguous
by assigning a confident training label.

Vela adds `NO_FEEDBACK` at ID 4 while preserving historical IDs 0–3. An ordinary
new task or supplied material is not satisfaction. The legacy four-way
checkpoints stay unchanged. New Vela training initializes all five head rows from the qualified Vela
base and trains them jointly with its encoder. Legacy adapter migration is a
separate compatibility tool; it does not establish the lineage of a new release.
`NO_FEEDBACK` does not match any of the four feedback rules. A low-confidence
Vela result is an abstention, never a fabricated `SAT` prediction. The checkpoint
still cannot recover context absent from its input, so mixed intentions and
ambiguous references need separate reporting.

Guard detects instruction attacks, independently of content risk. A
harmful request without an instruction attack can be `benign` for Guard
and `unsafe` for Safety. A quoted attack discussed as data can also be benign.
The same attack classifier should handle an override requesting harmless
content. Evaluate all four combinations of content risk and instruction attack.

Safety and Hazard are separate checkpoints. Hazard is unconditional and has
explicit safe negatives. It is not the historical nine-way head that must
choose a risk even for safe input. The twelve-label taxonomy does not cover
all possible risks: unsupported source categories include manipulation,
copyright, high-risk government decisions, vague caution, and general ethical
judgments. Their absence is not a safety endorsement.

## Data preparation

The builders verify fixed source revisions or file hashes, retain IDs and
source groups, and refuse to overwrite materialized corpora. Run commands from
the repository root. Download only the source files required by a builder.

- [Feedback builder](user_feedback_classifier/vela_data.py) uses
  [WildFeedback](https://huggingface.co/datasets/microsoft/WildFeedback) user turns
  and explicit weak projections. It excludes assistant turns, ambiguous flags,
  and detected secret-like strings. Conversation groups stay together. The
  resulting four-way labels remain weak supervision, even when a model fits
  them well. Report those scores separately from author or human judgments.
  The reserved [FeedbackQA diagnostic](user_feedback_classifier/vela_feedbackqa.py)
  uses human excellent ratings and an explicit positive-answer filter from the
  [official test source](https://huggingface.co/datasets/McGill-NLP/feedbackQA).
  It measures SAT recall only; no FeedbackQA examples enter this training recipe.
  The versioned `vela_weak_supervision_v2` training projection uses the source
  turn state and user prose to exclude ordinary continuations, quoted/code
  keywords and mixed intentions; the original weak development results remain
  available as a separate diagnostic. Its explicit `structured-v3` quote policy
  covers curly/Chinese quotations and Markdown block quotes, with reviewed
  ID/text-hash exceptions; the default `ascii-v2` corpus remains reproducible.
  `vela_no_feedback` materializes reviewed
  source user turns from an explicit ID/text-hash receipt. This is a single
  model-assisted review, not independent human annotation; source `NEWTOPIC`
  alone is insufficient, and excluded ambiguous cases are recorded.
- [Guard builder](prompt_guard_fine_tuning_lora/vela_data.py) combines
  [LLMail](https://huggingface.co/datasets/microsoft/llmail-inject-challenge)
  attack annotations with explicitly reviewed
  [SALAD](https://huggingface.co/datasets/OpenSafetyLab/Salad-Data) augmented and
  original requests. SALAD requires a source/text-hash-bound prompt-attack review
  sidecar; wording changes and source attack flags do not supply gold labels.
  Unreviewed, uncertain or conflicting question families and their transitive
  text aliases are excluded. Submitter teams and question families stay in one
  partition. LLMail labels concern untrusted email context and remain weak annotations.
  ToxicChat's noncommercial data is excluded from these Vela candidates.
- [Safety/Hazard builder](safety_classifier/vela_data.py) reads
  [AEGIS 2](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0).
  It reproduces a historical **weak source crosswalk**, not reviewed Vela gold.
  Its binary target is the source prompt label. Dialogue categories are attributed to
  the prompt only for prompt-only examples or unsafe prompts with a safe
  response. Source-safe prompts have zero projected risk targets. These
  attribution rules and full masks do not establish that the visible text
  satisfies Vela's twelve definitions; a source category can describe a topic
  or quoted material. Unsupported source categories remain in the audit.
- [CultureGuard extension](safety_classifier/vela_cultureguard.py) uses the
  [official twelve-language source](https://huggingface.co/datasets/nvidia/Nemotron-Safety-Guard-Dataset-v3).
  Original AEGIS IDs bind translations and cultural adaptations to the source
  group. Repeated prompt families are reserved for test before validation and
  training. Disagreeing category annotations become unknown dimensions. The
  data is synthetic translation/adaptation, not twelve independent natural
  benchmarks. Its jailbreak tag is never converted into a Guard target.

Use the [content-risk rubric](safety_classifier/configs/hazard-rubric-v1.json)
for separate, explicitly reviewed development. It distinguishes harmful action
and current crisis from education, prevention and support; insufficient context
remains unknown. Review a source-stratified selection before looking at new
candidate predictions and preserve every raw-to-reviewed change. Low-support
classes cannot be certified or calibrated by a small reviewed subset.
For such development data, set Hazard's explicit
`--selection-minimum-support 10` to exclude classes with fewer than ten positive
or negative observations from the checkpoint selection category average. All raw per-class metrics
remain recorded. The default of one preserves existing selection behavior.
To materialize the pinned source projections:

```bash
python -m src.training.model_classifier.safety_classifier.vela_data \
  --source /data/aegis2 --output /artifacts/aegis
python -m src.training.model_classifier.safety_classifier.vela_cultureguard \
  --source /data/cultureguard --aegis /data/aegis2 \
  --output /artifacts/cultureguard
```

Supply admitted training rows and their source manifests explicitly. Authored
contrasts, repeated backgrounds and translations remain related training
examples; preserve `group_id` across every variant. Source annotations must
retain their original provenance and license.

Hazard rows add `targets` and `label_mask`, both in config label order. Missing
annotation is not a negative label. The loss averages observed binary losses
within each example, then averages examples; unknown dimensions have zero
loss and gradient. A partially annotated positive can supervise one hazard
without asserting that every other hazard is absent.

### Admit reviewed Hazard supervision

For new Hazard training and semantic evaluation, use
[vela_hazard_admission](safety_classifier/vela_hazard_admission.py) after source
selection, split isolation and full-text review. Broad source labels may help
stratify review sampling; they never supply the reviewed targets or negatives.
Keep historical raw evaluation labels and results as separate source-policy
diagnostics. An incompatible crosswalk does not excuse errors on correctly
labelled safe examples.

The input JSONL must retain `id`, `text`, `group_id`, `source`, `source_revision`
and `source_split`. The review JSON binds the exact input-file SHA, rubric-file
SHA and ordered labels. Every input ID needs an item with matching source
identity and visible-text SHA, a reason, and three explicit, disjoint lists:
`positive_labels`, `negative_labels`, `unknown_labels`. Together they must cover
all twelve labels. `unknown_reasons` gives a reason for each unknown label.
Record the actual reviewer and whether review was AI-assisted or human; hashes
establish identity, not annotation accuracy.

This runnable example illustrates the format using one authored documentation
sentence; it is not a training corpus or quality benchmark:

```bash
export VELA_ADMISSION_DEMO="$(mktemp -d)"
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

out = Path(os.environ["VELA_ADMISSION_DEMO"])
rubric = Path("src/training/model_classifier/safety_classifier/configs/hazard-rubric-v1.json")
rubric_bytes = rubric.read_bytes()
labels = list(json.loads(rubric_bytes)["labels"])
sha = lambda value: hashlib.sha256(value).hexdigest()
row = {
    "id": "example-1", "group_id": "example-family-1",
    "source": "documentation_example", "source_revision": "authored-example-v1",
    "source_split": "train", "text": "Explain what personal data means.",
    "source_categories": ["PII/Privacy"],
}
input_bytes = (json.dumps(row) + "\n").encode()
(out / "input.jsonl").write_bytes(input_bytes)
item = {key: row[key] for key in ("id", "group_id", "source", "source_revision", "source_split")}
item.update({
    "text_sha256": sha(row["text"].encode()),
    "positive_labels": [], "negative_labels": labels, "unknown_labels": [],
    "unknown_reasons": {}, "binary_label": "safe",
    "reason": "A general definition requests no harmful action or personal disclosure.",
})
review = {
    "protocol": "documentation-review-example-v1",
    "reviewer": "illustrative AI review", "reviewer_kind": "ai",
    "input_sha256": sha(input_bytes), "rubric_sha256": sha(rubric_bytes),
    "labels": labels, "items": [item],
}
(out / "review.json").write_text(json.dumps(review, indent=2) + "\n")
PY
python -m src.training.model_classifier.safety_classifier.vela_hazard_admission \
  --input "$VELA_ADMISSION_DEMO/input.jsonl" \
  --review "$VELA_ADMISSION_DEMO/review.json" \
  --rubric src/training/model_classifier/safety_classifier/configs/hazard-rubric-v1.json \
  --output "$VELA_ADMISSION_DEMO/admitted"
```

Use the resulting `admitted/rows.jsonl` as the new Hazard training input; admit
the independently reviewed development partition separately. A known positive
produces `unsafe`. Only twelve observed negatives produce `safe`; partial
negative supervision stays `unknown` with `binary_observed=false`, outside the
safe false-positive denominator. Fully unknown rows are excluded from loss,
while their group IDs and unknown counts remain in the manifest. Identical
visible text with conflicting targets or masks requires explicit adjudication.

`source_annotations` preserves the previous input metadata, including weak
targets when present; `current_supervision` identifies the actual review.
For an existing reviewed corpus, recover its exact review and parent lineage
and add `--preserve-existing` to reject unexpected target or mask changes.
This flag checks equality; it does not certify an old review by source name.
Translations and context variants need verified semantic parent provenance or
their own text review. Reserve all related source groups across splits before
creating variants. Never overwrite a frozen input, review or admission output.

## Initialization and training

The validated environment uses Transformers 4.57.6, PEFT 0.18.1, datasets 4.1.1,
scikit-learn 1.7.2 and a platform-appropriate PyTorch 2.10 installation. Record the
actual Torch accelerator build and all installed dependency versions with each
candidate; installing a CPU or CUDA wheel does not enable ROCm.

New Vela runs start from `llm-semantic-router/Vela-1.0-Encoder-307M` at
revision `fe9ccc074b781bc0e2e13c2c8d26f2640410636a`. Download that revision
to `/models/encoder` and retain its file hashes in the training receipt.
Initialize a fresh task head and train the complete encoder. Use the actual
parent ID and immutable revision; changing metadata on an old adapter does not
transfer it to a new base.

The [shared trainer](sequence_repair/README.md) handles softmax tasks:

```bash
python -m src.training.model_classifier.sequence_repair.train \
  --base /models/encoder --base-id llm-semantic-router/Vela-1.0-Encoder-307M \
  --base-revision fe9ccc074b781bc0e2e13c2c8d26f2640410636a \
  --method full --fresh-head --contract /artifacts/task/contract.json \
  --train /artifacts/task/train.jsonl \
  --dev /artifacts/task/development.jsonl \
  --output /artifacts/run --steps 1200 --batch-size 16 --accumulate 1 \
  --max-length 32768 --microbatch-token-budget 32768 \
  --learning-rate 0.00001 --head-learning-rate 0.0001 \
  --evaluation-dtype float32 --eval-every 1200 \
  --balanced-sampling --source-balanced-sampling --selection source-macro-f1
```

These are example experiment settings. Freeze task-specific sampling, budgets,
and development gates before training; they are not universal release criteria.
An optional `--head-learning-rate` separates the fresh head's rate from the
encoder rate while sharing the schedule. Full runs save complete `best-model`
and `last-model` checkpoints with their `training-origin.json`. Declare which
checkpoint may be selected before reading development results.

Hazard instead uses `safety_classifier.train_vela_hazard` with the same base,
full-training, contract and budget arguments. Pass the admitted training
`rows.jsonl` and separately admitted development rows described above, keeping
raw source crosswalks for historical reproduction. Its selection choices are
`macro-ap`, `source-macro-ap`, `fp-budget-macro-f1`, and
`joint-fp-budget-macro-f1`; it implements masked BCE and samples safe negatives
plus positive categories. Do not pass Hazard data through the single-label
trainer or decode its logits with softmax.

Both trainers retain explicit LoRA continuation for compatible existing
experiments. Such a run must use its actual original base and adapter lineage;
it cannot substitute for training a new Vela release from the qualified base.

For deployment with a false-alarm budget, use `--selection fp-budget-macro-f1
--selection-false-positive-budget 0.05`. At each development checkpoint it fits
one shared threshold to maximize supported-category macro-F1 while at most 5%
of explicitly safe development rows trigger any label. All labels, including
those with insufficient positive support for selection, count toward this
false-alarm budget. The saved operating point also reports correct known-positive
category hits, all-label metrics, support and feasibility. A prediction of an
unrelated hazard does not count as a correct category hit. Fixed 0.5 metrics and
AP remain available at every checkpoint. This fits an empirical development
operating point; it does not guarantee the same population false-alarm rate.
The historical AP selection modes and default remain unchanged.

Hazard also accepts `--source-weights` as a JSON object naming every training
source with a positive weight. This replaces equal-source selection while
preserving optional length/category balancing within each source. For example,
large external corpora need not receive the same draw probability as a small
set of authored contrasts. Existing runs without this option preserve their
sampling sequence. Each evaluation checkpoint records actual source draws,
unique-row coverage, language draws and positive-label exposures. Select total
steps using these counts and loss curves; repeated sampling is not an epoch.
Pass the actual `--base-id` as well as `--base-revision` when using a new encoder.
For partial supervision, `--supervision-diagnostics` additionally records
observed-label-count distributions, positive/negative/unknown exposures and
exact loss gradients with respect to each logit, broken down by source. It also
records actual final-head row gradient norms before clipping. Source-wise
absolute logit gradients are not signed encoder gradients, and source draw
weights are not the effective positive/negative loss ratio: one observed label
gets more weight per example than one of twelve observed labels.

Both loops use FP32 parameters and loss, BF16 GPU autocast, and explicit
normalization over the global batch. This avoids a Transformers loss-
kwargs path that can over-scale ModernBERT updates during gradient accumulation.
Dataset, group, ID and normalized-text checks run before training. Oversize
training examples are recorded as rejected; evaluation fails instead of
silently truncating. Short and long partitions should be materialized and
reported explicitly, so rejected examples do not disappear from the task.
The partition command preserves `over-budget.jsonl` for separate long-context
evaluation, together with counts and checksums for both files.

## Long-context development

`vela_context_curriculum_v2` places supplied training payloads inside explicit
background paragraphs while retaining their labels and source groups. Its
Hazard strata count each observed positive category; unknown categories are
not treated as positives. Supply at least two reviewed backgrounds per language:

```bash
python -m src.training.model_classifier.vela_context_curriculum_v2 \
  --input artifacts/vela/guard/train.jsonl --task sequence \
  --backgrounds artifacts/vela/training-backgrounds.json \
  --tokenizer artifacts/vela/base --output artifacts/vela/guard/context-train.jsonl
```

Backgrounds are a JSON object mapping language codes to lists of text. The
builder creates training variants, not a natural benchmark. Independently
prepare development and final backgrounds and semantic families, including
benign quotations and negations. Report language coverage explicitly.

Measure actual tokens, not characters or padding length. Evaluate 4K, 8K, 16K
and 32K inputs at the head, middle and tail. A 32K encoder config states position
capacity; training at 2K does not establish long-task accuracy. Mixed-length
curricula can use `--length-balanced-sampling` with a 32768 budget and small
microbatches. Both the shared sequence loop and the Hazard loop support the explicit
`--microbatch-token-budget 32768`: each samples the same global batch first,
then groups examples by actual length so that each microbatch fits the padded
token budget. Each example retains its global-batch loss weight. This changes
dropout random-number consumption, so record a new recipe rather than claim
an identical continuation of a previous run. Keep short development retention
in checkpoint selection.
Hazard defaults to `--loss-normalization observed`: it averages BCE over each example's observed labels, then weights
that example by the complete optimizer-step sample count. Examples with more
observed categories do not receive extra weight when microbatch sizes differ.
The explicit `--loss-normalization taxonomy` alternative divides each masked
sum by the fixed output dimension. It reduces partially annotated examples'
relative influence without supplying any negative label for an unknown category.
Treat this as a different training objective and compare it with identical data,
sampling and initialization. Supervision diagnostics record the actual selected
objective. Every development checkpoint retains per-row probabilities so that
AP selection cannot hide a different recall/false-positive operating point.
Report full-context and chunked/windowed policies as separate systems.

## Freeze, evaluate, and export

Select checkpoints using development data only. Use source-balanced metrics
when a large weak corpus would hide regressions on smaller independent
families. Preserve per-class, per-source, per-language, per-length and position
breakdowns. Test files are not accepted by the training selection loop.

The shared `sequence_repair.export` command freezes either a selected full
checkpoint (`--method full`, `--base` pointing to that checkpoint) or a LoRA
adapter (`--method lora`, with its original base and `--adapter`). Both paths
verify every exported tensor and reloaded logits, preserve the trained label
contract, and record parent, parameter counts, label order and file hashes.
The LoRA path additionally checks numerical equivalence across merging. Its run manifest must explicitly
record that test data did not select the checkpoint. Run independent final
inference only after freezing that candidate. A newly trained checkpoint needs
new inference graphs; an ONNX file from previous weights is not a valid export.

Hazard's `safety_classifier.vela_hazard` CLI evaluates independent sigmoid
outputs and preserves unknown-label masks. `--calibrate-development` fits
per-class maximum-F1 operating thresholds only where development support is
sufficient. These are operating points, not calibrated posterior probabilities.
Freeze the threshold file before using `--thresholds` on final data, and report
safe-input false positives as well as risk recall. Never tune thresholds on the
final test and describe that fitted score as independent performance.
Always retain results for the fixed 0.5 threshold alongside fitted results.
The router applies one configured threshold per hazard rule to the maximum
selected category score. It does not automatically read a per-category
calibration file; use separate explicit rules when categories need different
operating thresholds.

Legacy Feedback's four labels classify an applicable follow-up; none represents
an ordinary new question. Its `vela_applicability` result is a forced-label
diagnostic. For the five-class Vela contract, report `NO_FEEDBACK` precision and
recall, errors into any of the original four classes including SAT, and original
four-class retention. Do not relabel ordinary queries as satisfied users.
Conversation gating remains required, and ambiguous replies may still need
context unavailable to the classifier.

Final cards must distinguish source labels from human judgments, duplicated
translations from independent groups, encoder capacity from measured task
quality, and native full-context evidence from serving-time chunking. Include
failure cases and regressions alongside gains. Publication is a separate step
and must use new Vela repositories rather than overwrite historical models.

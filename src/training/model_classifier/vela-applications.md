# Vela application classifier recipes

This guide covers Feedback, PromptGuard, Safety, and Hazard. A recipe is a
reproducible experiment; it does not certify a particular checkpoint. Published
model cards must identify the selected weights, actual encoder parent, dataset
revisions, separate development and final results, and measured limitations.

## Output contracts

| Task | Ordered labels | Output |
|---|---|---|
| Feedback | `SAT`, `NEED_CLARIFICATION`, `WRONG_ANSWER`, `WANT_DIFFERENT`, `NO_FEEDBACK` | Softmax |
| PromptGuard | `benign`, `jailbreak` | Softmax |
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
checkpoints stay unchanged. The five-way recipe preserves their trained head
rows when initializing its new class, then trains all five jointly; adding a
softmax term changes probabilities and requires new development checks.
`NO_FEEDBACK` does not match any of the four feedback rules. A low-confidence
Vela result is an abstention, never a fabricated `SAT` prediction. The checkpoint
still cannot recover context absent from its input, so mixed intentions and
ambiguous references need separate reporting.

PromptGuard detects instruction attacks, independently of content risk. A
harmful request without an instruction attack can be `benign` for PromptGuard
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
- [PromptGuard builder](prompt_guard_fine_tuning_lora/vela_data.py) combines
  [LLMail](https://huggingface.co/datasets/microsoft/llmail-inject-challenge)
  attack annotations with paired
  [SALAD](https://huggingface.co/datasets/OpenSafetyLab/Salad-Data) attacks and
  original questions. Normalized annotation conflicts are removed across
  phases; submitter teams and paired questions stay in one partition. The
  LLMail labels concern untrusted email context and remain weak annotations.
  ToxicChat's noncommercial data is excluded from these Vela candidates.
- [Safety/Hazard builder](safety_classifier/vela_data.py) reads
  [AEGIS 2](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0).
  Its binary target is the prompt label. Dialogue categories are attributed to
  the prompt only for prompt-only examples or unsafe prompts with a safe
  response. Safe prompts have zero risk targets. Unsupported or ambiguous
  category attribution is recorded rather than guessed.
- [CultureGuard extension](safety_classifier/vela_cultureguard.py) uses the
  [official twelve-language source](https://huggingface.co/datasets/nvidia/Nemotron-Safety-Guard-Dataset-v3).
  Original AEGIS IDs bind translations and cultural adaptations to the source
  group. Repeated prompt families are reserved for test before validation and
  training. Disagreeing category annotations become unknown dimensions. The
  data is synthetic translation/adaptation, not twelve independent natural
  benchmarks. Its jailbreak tag is never converted into a PromptGuard target.

Hazard's historical `safety_classifier.vela_hazard_supervision` v3 projection
remains reproducible. The subsequent `safety_classifier.vela_hazard_partial` v4
projection quarantines all generated `jailbreaking` rows and unsafe `adapted`
rows because the generation process does not establish reliable binary and
exhaustive category annotations. Generic unsafe rows retain only positive
categories also supported by the original AEGIS training prompt; all other
categories are unknown. Generic/adapted safe rows retain full negative masks.
These and unchanged AEGIS labels remain weak source supervision, not individually
verified clean negatives. Both versions refuse development/test input and record
exclusions. Never overwrite the original raw development/test labels.

Use the [content-risk rubric](safety_classifier/configs/hazard-rubric-v1.json)
for separate, explicitly reviewed development. It distinguishes harmful action
and current crisis from education, prevention and support; insufficient context
remains unknown. Review a source-stratified selection before looking at new
candidate predictions and preserve every raw-to-reviewed change. Low-support
classes cannot be certified or calibrated by a small reviewed subset.
For such development data, set Hazard's explicit
`--selection-minimum-support 10` to exclude classes with fewer than ten positive
or negative observations from checkpoint selection AP. All raw per-class metrics
remain recorded. The default of one preserves existing selection behavior.
`safety_classifier.vela_hard_negatives` supplies a small original contrast corpus
with fixed train/development semantic families and paired translations. These
authored examples supplement source supervision; they are not independent
natural-user evidence.

For example:

```bash
python -m src.training.model_classifier.safety_classifier.vela_data \
  --source /data/aegis2 --output /artifacts/aegis
python -m src.training.model_classifier.safety_classifier.vela_cultureguard \
  --source /data/cultureguard --aegis /data/aegis2 \
  --output /artifacts/cultureguard
```

Each task has a `vela_authored` module for original training contrasts.
These examples and their repeated or translated variants are training data,
not independent evidence. Preserve `group_id` across every variant.

Hazard rows add `targets` and `label_mask`, both in config label order. Missing
annotation is not a negative label. The loss averages observed binary losses
within each example, then averages examples; unknown dimensions have zero
loss and gradient. A partially annotated positive can supervise one hazard
without asserting that every other hazard is absent.

## Initialization and training

The validated environment uses Transformers 4.57.6, PEFT 0.18.1, datasets 4.1.1,
scikit-learn 1.7.2 and a platform-appropriate PyTorch 2.10 installation. Record the
actual Torch accelerator build and all installed dependency versions with each
candidate; installing a CPU or CUDA wheel does not enable ROCm.

Initialize a fresh adapter and trainable prediction head:

```bash
python -m src.training.model_classifier.sequence_repair.initialize \
  --base /models/encoder --base-id llm-semantic-router/mmbert-32k-yarn \
  --base-revision 72a23a6640489471eb4ff7ad3ec5bc80af8a27de \
  --contract /artifacts/task/contract.json --output /artifacts/initial
```

Use the actual parent ID and revision when the encoder changes. A reused
adapter must also retain its previous training lineage; sharing a model name
is not evidence that adapters from different parents are equivalent.

The [shared trainer](sequence_repair/README.md) handles softmax tasks:

```bash
python -m src.training.model_classifier.vela_partition \
  --input /artifacts/task/validation.jsonl --tokenizer /models/encoder \
  --max-length 2048 --output /artifacts/validation-2048
python -m src.training.model_classifier.sequence_repair.train \
  --base /models/encoder --base-revision 72a23a6640489471eb4ff7ad3ec5bc80af8a27de \
  --adapter /artifacts/initial --contract /artifacts/task/contract.json \
  --train /artifacts/task/train.jsonl \
  --dev /artifacts/validation-2048/within-budget.jsonl \
  --output /artifacts/run --steps 1200 --batch-size 8 --accumulate 4 \
  --max-length 2048 --learning-rate 0.00003 --eval-every 200 \
  --balanced-sampling --source-balanced-sampling --selection source-macro-f1
```

Hazard instead uses `safety_classifier.train_vela_hazard` with the same base,
adapter, contract, data and budget arguments. Its selection choices are
`macro-ap` and `source-macro-ap`; it implements masked BCE and samples safe
negatives plus positive categories. Do not pass Hazard data through the
single-label trainer or use softmax to decode its logits.

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

`vela_context_curriculum` places training payloads inside original background
paragraphs while retaining their labels and groups. `vela_context_probes`
provides separate authored development stress cases. Neither generator
creates a natural benchmark. Use independent backgrounds and semantic families
for final evaluation, include benign quotations and negations, and report
English and Chinese separately before claiming multilingual robustness.

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
Hazard first averages BCE over each example's observed labels, then weights
that example by the complete optimizer-step sample count. Examples with more
observed categories do not receive extra weight when microbatch sizes differ.
Report full-context and chunked/windowed policies as separate systems.

## Freeze, evaluate, and export

Select checkpoints using development data only. Use source-balanced metrics
when a large weak corpus would hide regressions on smaller independent
families. Preserve per-class, per-source, per-language, per-length and position
breakdowns. Test files are not accepted by the training selection loop.

The shared `sequence_repair.export` command merges a selected adapter into a
standard safetensors checkpoint, checks before/after logits, and records parent,
parameter counts, label order and file hashes. Its run manifest must explicitly
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

---
title: Train Vela Classifiers
sidebar_label: Classifiers
---

# Train Vela classifiers

Vela classifiers turn a request into a routing signal. Adapt them when your
application needs better coverage of a language, domain, or request pattern.
All tasks use the shared Vela Encoder foundation.

To use the published models, start with
[Vela runtime configuration](../tutorials/global/vela-models.md).

## Choose the task and labels

| Model | Output | Example use |
| --- | --- | --- |
| Domain | 14 subject areas | Route a legal question to a specialist |
| Guard | `benign`, `jailbreak` | Detect attempts to override instructions |
| Feedback | Four feedback types plus `NO_FEEDBACK` | Handle an unsatisfactory answer |
| Modality | `AR`, `DIFFUSION`, `BOTH` | Choose text, image, or combined output |
| FactCheck | `FACT_CHECK_NEEDED`, `NO_FACT_CHECK_NEEDED` | Select an answer-verification path |
| PII | Token labels for 17 entity types | Find personal information for redaction |

[Safety and Hazard](./mmbert-safety-classifier) have a separate guide for
content-risk detection. Guard's task is prompt-attack detection; ordinary
harmful content belongs to Safety.

### Domain

Domain predicts biology, business, chemistry, computer science, economics,
engineering, health, history, law, math, other, philosophy, physics, or psychology.
Include requests outside the specialist domains so the model learns a useful
`other` class.

### Feedback

| Label | Meaning |
| --- | --- |
| `SAT` | The user is satisfied with the answer |
| `NEED_CLARIFICATION` | The user needs an explanation of the answer |
| `WRONG_ANSWER` | The user reports an incorrect answer |
| `WANT_DIFFERENT` | The user requests a different format or approach |
| `NO_FEEDBACK` | The message does not express feedback |

The input is the current user follow-up. Include ordinary new questions as
`NO_FEEDBACK`; a positive statement unrelated to the answer is not satisfaction.
Keep ambiguous replies separate when the missing conversation prevents a
reliable label.

### FactCheck and Modality

FactCheck decides whether an answer needs verification; it does not determine
whether a claim is true. Include factual questions alongside creative and
non-factual requests.

Modality predicts the requested output from text. `AR` means text,
`DIFFUSION` means an image, and `BOTH` means a combined response. It is a text
classifier and does not inspect uploaded images.

## Prepare your data

The shared sequence trainer expects JSONL rows with `id`, `text`, `label`,
and `group_id`. Keep related examples in one partition and preserve
`source`, `language`, `length_bucket`, and `position` when you need those
evaluation slices. A separate `contract.json` defines the ordered labels.

Use the [sequence training reference](https://github.com/vllm-project/semantic-router/tree/main/src/training/model_classifier/sequence_repair)
for file formats and source preparation. The
[application recipes](https://github.com/vllm-project/semantic-router/blob/main/src/training/model_classifier/vela-applications.md)
provide Feedback and Guard dataset builders. Review their source labels against
your task, especially quoted attacks, benign instructions, and neutral follow-ups.

## Train a sequence classifier

Download a fixed revision of Vela Encoder to `/models/vela-base` and set
`VELA_BASE_REVISION` to that revision. The example below assumes your
FactCheck contract and training/development files are already prepared.

```bash
python -m src.training.model_classifier.sequence_repair.train \
  --method full --fresh-head \
  --base /models/vela-base \
  --base-id llm-semantic-router/Vela-1.0-Encoder-307M \
  --base-revision "${VELA_BASE_REVISION:?Set the downloaded revision}" \
  --contract /data/factcheck/contract.json \
  --train /data/factcheck/train.jsonl --dev /data/factcheck/dev.jsonl \
  --output /data/factcheck/run \
  --steps 600 --batch-size 4 --accumulate 4 \
  --max-length 32768 --microbatch-token-budget 32768 \
  --learning-rate 0.00001 --head-learning-rate 0.0001 \
  --eval-every 200 --evaluation-dtype float32 --selection source-macro-f1
```

`--fresh-head` initializes the classifier and trains it with the complete
encoder. For deliberate continuation, supply a compatible Vela task checkpoint
and omit that flag. Choose step count and sampling for your data; the example
settings are a starting point.

The input limit includes special tokens. Oversize training examples are
reported as rejected, and evaluation rejects overflow. Include real long
examples when increasing the limit.

### Continue a model while preserving existing behavior

When adapting a published classifier, start from that task checkpoint and omit
`--fresh-head`. The optional `--trainable-last-layers` setting updates only the
last encoder blocks and classification head. Mark old training examples with
`retention_replay: true` and use `--retention-targets` to constrain changes to
their predictions while learning from new labels.

Follow the [continuation workflow](https://github.com/vllm-project/semantic-router/tree/main/src/training/model_classifier/sequence_repair#preserve-behavior-during-continued-training)
to generate targets and configure training. Compare both attack recall and false
alarms on separate development requests before replacing a deployed Guard model.

## PII detector

PII requires entity-span training rather than one label per request. The model
uses BIO labels: `B-TYPE` starts an entity, `I-TYPE` continues it, and `O`
marks other tokens.

Use the [PII training workflow](https://github.com/vllm-project/semantic-router/tree/main/src/training/model_classifier/pii_model_fine_tuning_lora),
including `train_repair.py`, to align character spans with tokenizer outputs
and train the token classifier. Evaluate entity-level precision, recall, and F1.
Token accuracy can hide missed entities because most tokens are `O`.

## Evaluate and deploy

Compare the original and trained checkpoints on the same held-out requests.
Inspect per-class errors, languages, short and long inputs, and requests that
should produce no match. Choose thresholds using development data.

Export the selected model with
[the sequence exporter](https://github.com/vllm-project/semantic-router/tree/main/src/training/model_classifier/sequence_repair#freeze-then-evaluate-the-independent-test).
It includes the trained weights, tokenizer, and task-specific label mappings.
PII uses its own export workflow.

Finally, configure [local model bindings](../installation/runtime/in-process.md)
and send representative requests through
[route preview](../installation/runtime/lifecycle-diagnostics.md). Check the
actual signal and decision as well as model confidence.

The [artifact index](https://github.com/vllm-project/semantic-router/blob/main/src/training/model_artifacts.json)
retains entrypoints for earlier mmBERT adapters and merged releases.

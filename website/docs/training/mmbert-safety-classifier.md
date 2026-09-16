---
title: Train Vela Safety and Hazard
sidebar_label: Safety and Hazard
---

# Train Vela Safety and Hazard

Adapt Vela Safety and Hazard to your application's content policy.
**Safety** predicts `safe` or `unsafe`. **Hazard** identifies the categories
of risk, so you can choose a suitable response. Prompt injection and jailbreak
detection use the separate Guard model.

To use the published models, follow
[Safety models](../installation/runtime/safety.md). This guide covers training
your own compatible checkpoints.

## Define the outputs

Both models use the Vela Encoder foundation and a sequence-classification head.

| Model | Output | Training objective |
| --- | --- | --- |
| Safety | Two-class softmax: `safe`, `unsafe` | Cross-entropy |
| Hazard | Twelve independent sigmoid scores | Masked binary cross-entropy |

Hazard uses this ordered label set:

```text
violence, criminal_activity, sexual_content, child_exploitation,
hate, harassment_abuse, regulated_substances, weapons,
self_harm, privacy, specialized_advice, misinformation
```

A request may have multiple hazards or none. Keep safe negative examples in
Hazard training. If a category has not been reviewed, mark it unknown with
`label_mask: 0` for that category instead of treating it as absent.

## Prepare training and evaluation data

Start with the
[Safety and Hazard data builders](https://github.com/vllm-project/semantic-router/blob/main/src/training/model_classifier/vela-applications.md#data-preparation)
for AEGIS and CultureGuard, or use your own labeled requests. Review source
labels against your policy: mentioning a sensitive topic, quoting a threat,
and requesting harmful action need different judgments.

Include safe educational, preventive, and supportive requests alongside
unsafe examples. Keep related documents, conversations, and translations in
one split. Prepare training, development, and final test partitions separately.

Each task needs a `contract.json` with its label maps. Safety rows contain a
single `label`; Hazard rows contain ordered `targets` and `label_mask` arrays.
The [recipe reference](https://github.com/vllm-project/semantic-router/blob/main/src/training/model_classifier/vela-applications.md)
documents the full formats and reviewed-data admission command.

## Train from Vela Encoder

Use an isolated environment with Transformers 4.57.6 and a platform-appropriate
PyTorch build. The [training reference](https://github.com/vllm-project/semantic-router/blob/main/src/training/model_classifier/vela-applications.md#initialization-and-training)
lists dependencies. ROCm training uses PyTorch's `cuda` device API.

Download a fixed Vela Encoder revision to `/models/vela-base` and set
`VELA_BASE_REVISION` to that revision. The following commands assume your
contract and data files are ready.

Train Safety:

```bash
python -m src.training.model_classifier.sequence_repair.train \
  --method full --fresh-head \
  --base /models/vela-base --base-id llm-semantic-router/Vela-1.0-Encoder-307M \
  --base-revision "${VELA_BASE_REVISION:?Set the downloaded revision}" \
  --contract /data/safety/contract.json \
  --train /data/safety/train.jsonl --dev /data/safety/dev.jsonl \
  --output /data/safety/run \
  --steps 600 --batch-size 4 --accumulate 4 \
  --max-length 32768 --microbatch-token-budget 32768 \
  --learning-rate 0.00001 --head-learning-rate 0.0001 \
  --eval-every 200 --evaluation-dtype float32 \
  --selection binary-fp-budget-recall --positive-label unsafe \
  --selection-false-positive-budget 0.1
```

Train Hazard with the multi-label trainer:

```bash
python -m src.training.model_classifier.safety_classifier.train_vela_hazard \
  --method full --fresh-head \
  --base /models/vela-base --base-id llm-semantic-router/Vela-1.0-Encoder-307M \
  --base-revision "${VELA_BASE_REVISION:?Set the downloaded revision}" \
  --contract /data/hazard/contract.json \
  --train /data/hazard/train.jsonl --dev /data/hazard/dev.jsonl \
  --output /data/hazard/run \
  --steps 600 --batch-size 4 --accumulate 4 \
  --max-length 32768 --microbatch-token-budget 32768 \
  --learning-rate 0.00001 --head-learning-rate 0.0001 \
  --eval-every 200 --evaluation-dtype float32 \
  --selection fp-budget-macro-f1 --selection-false-positive-budget 0.05
```

These are example budgets and learning rates. Choose the allowed false-positive
rate for your application before selecting a checkpoint. Full training saves
complete `best-model` and `last-model` directories. To continue a compatible
Vela task checkpoint, supply it as the base and omit `--fresh-head`.

## Evaluate risks and long inputs

For Safety, measure unsafe recall and false positives on safe requests.
For Hazard, measure each category's precision and recall, average precision,
and how often safe requests trigger any category.

Choose thresholds on development data, then keep them fixed for the final
test. Report languages and categories separately so a large source cannot hide
a weak one.

Test short requests alongside 8K, 16K, and 32K documents. Place relevant content
at different positions and include benign quotations. The input budget includes
special tokens; oversize training rows are reported, and evaluation rejects
overflow. Full-context inference and window scanning need separate evaluation
because they expose different context to the model.

## Export and connect the models

Use [the sequence exporter](https://github.com/vllm-project/semantic-router/tree/main/src/training/model_classifier/sequence_repair#freeze-then-evaluate-the-independent-test)
with `--method full`, the selected checkpoint, and `--runtime-task safety`
or `--runtime-task hazard`. It preserves the trained label order and writes
the weights, tokenizer, and runtime mappings. An ONNX deployment requires
graphs exported from those same weights.

Configure the models through
[local bindings](../installation/runtime/in-process.md) or a supported
[external service](../installation/runtime/external.md).
The [Safety signal guide](/docs/tutorials/signal/learned/safety) explains how
to use a binary risk rule or a category condition. In a category-specific
Safety rule, the router runs Hazard after Safety passes its threshold.

Use [route preview](../installation/runtime/lifecycle-diagnostics.md) to verify
scores, decisions, errors, and latency with your actual serving configuration.

## Earlier Safety models

The earlier mmBERT Safety adapters and nine-class Hazard head remain in the
[legacy training workflow](https://github.com/vllm-project/semantic-router/tree/main/src/training/model_classifier/safety_classifier).
Their label contracts differ from Vela's twelve independent Hazard categories.

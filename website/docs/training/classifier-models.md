---
title: mmBERT-32K Classifier Models
sidebar_label: Classifier Models
---

# mmBERT-32K classifier models

The current classifier family adapts the same multilingual, long-context
ModernBERT encoder to several routing decisions. Sharing a foundation keeps
tokenization and encoder behavior consistent, while each task has its own
labels, data preparation, training loss, and output head.

## Shared architecture and training pattern

Intent, jailbreak, feedback, modality, and fact-check models use sequence
classification:

```text
request -> mmBERT-32K encoder -> pooled representation -> task classifier -> label
```

The PII model uses token classification:

```text
request -> mmBERT-32K encoder -> one classifier output per token -> BIO entities
```

The training scripts apply LoRA updates to ModernBERT attention and MLP
projections while training the task head. A run can retain the adapter or merge
it into the base weights. The `-lora` and `-merged` artifacts therefore share
the same logical architecture and label contract.

For the five standard sequence/token workflows, the repository also provides
convenience targets:

```bash
make train-mmbert32k-intent
make train-mmbert32k-jailbreak
make train-mmbert32k-feedback
make train-mmbert32k-factcheck
make train-mmbert32k-pii
```

Run the selected Python entrypoint with `--help` before overriding the target's
defaults. Dataset downloads, output checkpoints, and caches should live
outside Git.

## Intent classifier

The intent model predicts one of 14 subject areas: biology, business,
chemistry, computer science, economics, engineering, health, history, law,
math, other, philosophy, physics, or psychology.

It is a 14-way sequence classifier trained on MMLU-Pro questions plus
supplementary examples that improve the `other` fallback. The standard target
uses LoRA rank 32 and alpha 64, five epochs, batch 16, and learning rate
`2e-5`. Evaluation reports accuracy and weighted F1; deployment evaluation
should also inspect per-class recall and confusion with `other`.

MMLU-Pro publishes only `validation` (70 rows) and `test` (12032 rows), so the
training pool has to come out of `test`. The trainer reserves a stratified 20%
of `test` before it samples anything, keeps those rows off the gradient path,
and writes their row indices together with the metrics measured on them to
`heldout_eval.json` beside the checkpoint. Quote that number: accuracy over the
whole `test` split covers rows the model trained on, so it is not held-out
evidence.

Artifacts:
[`merged`](https://huggingface.co/llm-semantic-router/mmbert32k-intent-classifier-merged),
[`LoRA`](https://huggingface.co/llm-semantic-router/mmbert32k-intent-classifier-lora).

## Jailbreak detector

The jailbreak model is a binary sequence classifier for `benign` versus
`jailbreak`. Training combines ToxicChat, Salad-Data attack examples, and
explicit short and long attack-pattern augmentation. The pipeline trains the
classification head together with LoRA adapters and selects a checkpoint using
held-out classification metrics.

The released model card records LoRA rank 48 and alpha 96. The current standard
target defaults to rank 32, alpha 64, five epochs, batch 16, and learning rate
`2e-5`; override rank and alpha when reproducing the released configuration.
Evaluate benign false positives separately from missed attacks and include
multilingual, obfuscated, long-context, and indirect-prompt slices; aggregate
accuracy alone is not a safety threshold.

Artifacts:
[`merged`](https://huggingface.co/llm-semantic-router/mmbert32k-jailbreak-detector-merged),
[`LoRA`](https://huggingface.co/llm-semantic-router/mmbert32k-jailbreak-detector-lora).

## Feedback detector

The feedback model predicts four states from a user's follow-up message:

| Label | Meaning |
| --- | --- |
| `SAT` | The answer satisfied the user |
| `NEED_CLARIFICATION` | The user needs clarification |
| `WRONG_ANSWER` | The answer appears incorrect |
| `WANT_DIFFERENT` | The user wants a different result or approach |

It uses weighted cross-entropy to compensate for class imbalance. The standard
LoRA run uses rank 64 and alpha 128, up to 10 epochs, batch 16, learning rate
`2e-5`, early checkpoint selection by macro F1, and a 512-token training limit.

Artifacts:
[`merged`](https://huggingface.co/llm-semantic-router/mmbert32k-feedback-detector-merged),
[`LoRA`](https://huggingface.co/llm-semantic-router/mmbert32k-feedback-detector-lora).

## Modality router

The modality router predicts how a downstream response should be produced:

| Label | Route |
| --- | --- |
| `AR` | Autoregressive text model |
| `DIFFUSION` | Image-generation model |
| `BOTH` | A text explanation plus visual output |

Training assembles text requests, image-generation prompts, and mixed-modality
examples. The `BOTH` class can include reviewed seed/template examples and
optional examples synthesized through an OpenAI-compatible endpoint. The
trainer auto-selects LoRA rank from dataset size unless it is explicitly set,
uses focal loss and class weights, oversamples severe minority classes, and
selects by validation F1. The released model card records rank 16, alpha 32,
10 epochs, batch 32, and learning rate `2e-5`. The current script defaults to
eight epochs and chooses rank 16 for its default 6,000-example dataset.

```bash
python src/training/model_classifier/modality_routing_classifier/\
modality_routing_bert_finetuning_lora.py --help
```

Artifacts:
[`merged`](https://huggingface.co/llm-semantic-router/mmbert32k-modality-router-merged),
[`LoRA`](https://huggingface.co/llm-semantic-router/mmbert32k-modality-router-lora).

## Fact-check classifier

The fact-check model predicts `FACT_CHECK_NEEDED` or
`NO_FACT_CHECK_NEEDED`. It is a routing model: it decides whether a request
should enter a verification path; it does not determine whether a claim is
true.

Positive examples are information-seeking questions from sources such as
QASPER and Natural Questions. Negative examples include creative-writing,
code, and other non-information-seeking requests. The builder balances the two
classes and creates stratified train, validation, and test splits. The standard
target uses LoRA rank 32 and alpha 64, five epochs, batch 16, and learning rate
`2e-5`; the script selects by validation F1.

Artifacts:
[`merged`](https://huggingface.co/llm-semantic-router/mmbert32k-factcheck-classifier-merged),
[`LoRA`](https://huggingface.co/llm-semantic-router/mmbert32k-factcheck-classifier-lora).

## PII detector

The PII model is the exception to the sequence-classification pattern. It uses
a token-classification head and BIO encoding: `B-TYPE` marks the first token
of an entity, `I-TYPE` continues it, and `O` marks non-entity tokens. The
released model exposes 17 entity types as 35 labels (`O` plus two labels per
entity type).

The released model card records Presidio training, LoRA rank 32, five epochs,
batch 16, and learning rate `1e-4`. The current standard target expands that
method with a 70/30 AI4Privacy/Presidio mix, character-span alignment to
mmBERT subword tokens, rank 48, alpha 96, and eight epochs. Select and report
entity-level F1 rather than token accuracy, which is dominated by `O` tokens.

Artifacts:
[`merged`](https://huggingface.co/llm-semantic-router/mmbert32k-pii-detector-merged),
[`LoRA`](https://huggingface.co/llm-semantic-router/mmbert32k-pii-detector-lora).

## Validate the artifact contract

Before publishing or configuring a classifier, verify all of the following:

- the tokenizer and base-model revision match the training run;
- `id2label` and `label2id` preserve the documented order;
- the adapter includes the task head, or the merged model contains full model
  weights;
- the runtime uses the same truncation and normalization rules;
- adapter and merged logits agree on fixed examples;
- held-out metrics and failure slices are stored with the artifact.

The training entrypoints and artifact mapping are under
[`src/training/model_classifier`](https://github.com/vllm-project/semantic-router/tree/main/src/training/model_classifier).
For the hierarchical content-safety models, continue with
[Train the safety classifiers](./mmbert-safety-classifier).

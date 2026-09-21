---
title: Train Vela Router Models
sidebar_label: Overview
---

# Train Vela router models

Adapt Vela to the languages, topics, and routing policies in your application.
The family includes a shared encoder, task classifiers, an embedding model,
and a reranker. Start with a published task model when you need inference;
train a model when your evaluation shows a gap.

For deployment, follow [Use Vela models](../tutorials/global/vela-models.md).
This section covers creating and evaluating your own task models.

## Choose a training workflow

| Your goal | Workflow | What you train |
| --- | --- | --- |
| Improve semantic search or request similarity | [Embedding](./mmbert-32k-models#embedding-model-bi-encoder) | Vectors for queries and documents |
| Improve the order of retrieved candidates | [Reranking](./mmbert-32k-models#reranking-model-cross-encoder) | A relevance score for each query-document pair |
| Adapt domain, feedback, prompt-attack, fact-check or output-modality detection | [Classifiers](./classifier-models) | A request label |
| Detect personal information | [PII](./classifier-models#pii-detector) | Entity spans |
| Apply a content-risk policy | [Safety and Hazard](./mmbert-safety-classifier) | Safe/unsafe and risk categories |

[Vela's model catalog](./model-catalog) lists all eleven releases. Vela 1.0
accepts text; [multimodal embeddings](./multimodal-embeddings) are a separate
family. [Provider model evaluation](./model-performance-eval) and
[learned model selection](./ml-model-selection) help choose the downstream LLM.

## Choose your starting checkpoint {#record-the-base-and-task-lineage}

Use [Vela Encoder](https://huggingface.co/llm-semantic-router/Vela-1.0-Encoder-307M)
to train a new task. It is the common base for Vela's classifiers, embeddings,
and reranker. Download an explicit Hub revision and keep its tokenizer and
configuration together with the weights.

To improve an existing task, continue its complete Vela checkpoint, including
its trained head. Use the same labels and preprocessing unless you deliberately
train a new output contract. Each workflow explains fresh initialization and
continuation.

## Prepare examples from your application

Start with real requests and the behavior you expect. Include languages and
input lengths your application will receive, ordinary requests that should
not trigger a signal, and examples close to the decision boundary.

Keep related conversations, documents, and translations in the same split.
Use training data to update weights, development data to choose checkpoints
and thresholds, and a separate test set for the final comparison. The task
guides link to data formats and existing dataset builders.

## Train and compare

Run a small training job first to check the data and output labels. Then
compare your trained model with the published task model using the same inputs
and inference settings.

| Task | Measure |
| --- | --- |
| Classifier | Per-class precision, recall, F1, and false positives |
| PII | Entity-level precision, recall, and F1 |
| Embedding | Retrieval and semantic-similarity quality |
| Reranker | Ranking quality on fixed candidate lists |
| Safety and Hazard | Missed risks, safe-request false positives, and category coverage |

Check language and length slices separately. For long-context applications,
include short requests and actual long documents with relevant content near
the beginning, middle, and end. Measure latency at the input lengths you plan
to serve.

## Use the trained model in the router

Export a complete checkpoint with its tokenizer and labels. Choose a supported
engine and input limit in [Run models locally](../installation/runtime/in-process.md),
then validate your configuration:

```bash
vllm-sr config validate --config config.yaml
```

Use [route preview](../installation/runtime/lifecycle-diagnostics.md) to check
the signal, selected decision, and latency on representative requests before
deploying the model to traffic.

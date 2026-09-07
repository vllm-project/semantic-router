---
title: Train and Evaluate Router Models
sidebar_label: Overview
---

# Train and evaluate router models

Semantic Router uses small, task-specific models before it sends a request to
an LLM. These models create routing signals: they embed a request, rank a
candidate, classify an intent or risk, or predict which provider model should
answer. They do not generate the final response.

Use this section in three steps:

1. Choose the routing decision you need.
2. Learn the architecture and training objective for that model family.
3. Train, evaluate, and export an artifact with the same input and output
   contract that the router will use.

## Choose the model by routing decision

| You need to | Start with | Output |
| --- | --- | --- |
| Compare queries and documents efficiently | [mmBERT-32K embedder](./mmbert-32k-models#embedding-model-bi-encoder) | One normalized vector per input |
| Re-score a short list with higher accuracy | [mmBERT-32K reranker](./mmbert-32k-models#reranking-model-cross-encoder) | A relevance score for each query-document pair |
| Place text, images, and audio in one vector space | [Multimodal embeddings](./multimodal-embeddings) | A normalized cross-modal vector |
| Detect intent, jailbreaks, feedback, modality, fact-check needs, or PII | [Classifier models](./classifier-models) | A class, probability distribution, or token labels |
| Apply hierarchical prompt-safety policy | [Safety classifiers](./mmbert-safety-classifier) | `safe`/`unsafe`, followed by a hazard class |
| Learn which provider model should answer | [ML-based model selection](./ml-model-selection) | A provider-model choice |
| Compare models already in a provider pool | [Model performance evaluation](./model-performance-eval) | Per-model and per-category scores |

The [model catalog](./model-catalog) lists every artifact in the current MoM
multilingual embedding and classifier collections and maps release variants to
their training workflow.

## Understand the three common architectures

Most router models in this section use one of these patterns:

| Pattern | How it processes input | Best fit |
| --- | --- | --- |
| Bi-encoder | Encodes each input independently, then compares vectors | Large-scale retrieval and semantic cache lookup |
| Cross-encoder | Encodes a pair jointly and predicts one score | Accurate reranking of a small candidate set |
| Encoder plus task head | Encodes one request, then predicts sequence or token labels | Online routing and policy signals |

Multimodal models extend the bi-encoder pattern with separate text, image, and
audio towers whose outputs are projected into a shared space. The catalog and
family pages explain the exact towers, dimensions, labels, and objectives.

## Adapter versus merged model

Several classifier entries have both `-lora` and `-merged` artifacts. They are
two release shapes of the same logical model:

- A **LoRA adapter** stores the trained low-rank update and classification
  head. It is small, but inference also needs the compatible base model.
- A **merged model** folds the adapter into the base weights. It is larger and
  can be loaded as a standalone classifier by supported runtimes.

Choose the shape your inference backend supports. Do not compare the two names
as if they represented independently trained architectures.

## Follow the training lifecycle

### 1. Define the routing contract

Specify the labels or score, how that output changes routing, supported
languages and request lengths, latency budget, and fallback behavior. A label
is useful only when it maps to an observable router decision or policy.

### 2. Prepare versioned data

Keep training, validation, and test splits separate. Record dataset revisions,
licenses, preprocessing, label definitions, and synthetic-data rules.
Deduplicate before splitting so near-identical examples do not leak into
evaluation.

### 3. Start with a smoke run

Use the checked configuration or the script's `--help` output as the source of
truth. Resolve paths explicitly, run a small sample, and inspect label counts,
loss, and validation output before allocating a full training run.

### 4. Evaluate routing behavior

Match metrics to the decision:

| Task | Minimum useful evaluation |
| --- | --- |
| Sequence classification | Per-class precision, recall, F1, and confusion matrix |
| PII token classification | Entity-level precision, recall, and F1 |
| Safety detection | False-negative and false-positive rates plus per-hazard F1 |
| Embedding retrieval | Recall@k, ranking quality, language/domain slices, and latency |
| Model selection | End-to-end answer quality, cost, latency, and regret against an oracle |

Always retain a held-out test set. Slice results by language, domain, input
length, and the failure modes that matter to your deployment.

### 5. Export and integrate

Export the tokenizer, model or adapter, label mapping, and any architecture
metadata required by the runtime. Then validate the complete router
configuration:

```bash
vllm-sr validate --config config.yaml
```

Finally, send representative requests through the full router path. This
catches mismatched label order, preprocessing, dimensions, or artifact shape
that an offline trainer cannot detect.

## Recommended reading path

If you are new to these models, read the pages in this order:

1. [Current model catalog](./model-catalog)
2. [mmBERT-32K foundation, embedder, and reranker](./mmbert-32k-models)
3. [Small and large multimodal embeddings](./multimodal-embeddings)
4. [mmBERT-32K classifier models](./classifier-models)
5. [Two-level safety classifiers](./mmbert-safety-classifier)
6. [Model performance evaluation](./model-performance-eval)

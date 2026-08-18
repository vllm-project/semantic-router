---
title: Training Router Models
sidebar_label: Training Overview
---

# Training Router Models

Semantic Router uses small, task-specific models to understand a request before
it chooses a generative model. These router-owned models are different from the
LLMs in the provider pool: they classify, embed, score, or verify a request;
they do not produce the final answer.

This page helps you choose the right training workflow and explains what a
reproducible result should contain. Each workflow has its own README beside the
training code for exact commands and input formats.

## Choose a workflow

| Goal | Training area | Typical output |
|------|---------------|----------------|
| Classify domain, intent, feedback, modality, PII, or jailbreak risk | [`model_classifier`](https://github.com/vllm-project/semantic-router/tree/main/src/training/model_classifier) | A classifier or LoRA adapter used by a signal |
| Adapt embeddings for cache or retrieval | [`model_embeddings`](https://github.com/vllm-project/semantic-router/tree/main/src/training/model_embeddings) | An embedding model and evaluation report |
| Learn which provider model should answer a request | [`model_selection`](https://github.com/vllm-project/semantic-router/tree/main/src/training/model_selection) | KNN, KMeans, SVM, MLP, or reinforcement-learning selector artifacts |
| Compare provider models or produce routing scores | [`model_eval`](https://github.com/vllm-project/semantic-router/tree/main/src/training/model_eval) | Per-model and per-category evaluation data |
| Explore research ideas that are not part of the supported runtime contract | [`model_experiment`](https://github.com/vllm-project/semantic-router/tree/main/src/training/model_experiment) | Experimental code and local artifacts |

If your goal is to route among several generative models, start with
[ML-Based Model Selection](./ml-model-selection). If you need to measure the
models in an existing pool, start with
[Model Performance Evaluation](./model-performance-eval).

## The training lifecycle

### 1. Define the routing decision

Write down what the model must distinguish and how its output changes routing.
For example, a domain classifier may choose a decision, while a PII signal may
trigger a policy rule. A label is useful only when it maps to an observable
router behavior.

Decide before training:

- the label or score contract
- the languages and request types in scope
- acceptable false-positive and false-negative costs
- the latency and memory budget for online inference
- what happens when the model is unavailable or uncertain

### 2. Build and document the dataset

Keep training, validation, and test splits separate. Record the dataset source,
license, revision, preprocessing steps, label definitions, and any synthetic
data generation. Deduplicate before splitting so near-identical examples do not
leak into evaluation.

Security and privacy datasets need additional care. Remove credentials and
personal data that are not required for the task, restrict access to sensitive
examples, and document whether generated samples resemble production traffic.

### 3. Train from a pinned environment

Use the requirements and commands in the selected workflow directory. Record
the source commit, dependency versions, base model revision, random seed,
hyperparameters, and hardware class. Store large checkpoints and datasets in an
artifact registry rather than committing them to the repository.

LoRA is available for several classifier workflows when full fine-tuning is not
necessary. It trains a small adapter over a frozen base model, which can reduce
the amount of compute and storage required. Whether LoRA is the right choice
still depends on measured quality and the runtime's supported model format.

### 4. Evaluate the behavior that matters

Do not promote a model from training loss alone. Evaluate it on a held-out test
set and report the metrics that match the routing consequence:

| Task | Useful measurements |
|------|---------------------|
| Single-label classification | Per-class precision, recall, F1, confusion matrix |
| Multi-label or token classification | Per-label and entity-level precision, recall, F1 |
| Safety or privacy detection | False-negative and false-positive rates at the chosen threshold |
| Embedding retrieval | Recall@k, ranking quality, domain slices, latency |
| Model selection | End-to-end answer quality, selected-model distribution, cost, latency, regret against an oracle |

Slice results by language, domain, request length, and other conditions that are
important to your deployment. Measure online inference latency on the hardware
you intend to use.

### 5. Export and integrate

The exported artifact must match a runtime-supported format and the dimensions,
labels, and preprocessing used during training. Configure the corresponding
signal or selector with the artifact path, then validate the full router config
before deployment.

```bash
vllm-sr validate --config config.yaml
```

Run representative requests through the complete data path. This catches
integration errors that an offline notebook cannot see, such as label-order
mismatches, missing files, unsupported native backends, or a different embedding
model at inference time.

## Reproducible evaluation records

Long-lived documentation should explain how to reproduce a result, not paste a
single terminal session. Publish measured results as a versioned report or
artifact that includes:

- repository commit and model/dataset revisions
- evaluation command and configuration
- hardware and software environment
- sample counts and exclusions
- raw metrics plus aggregation method
- known limitations and failed slices

Without that context, accuracy, latency, cost, and training-time numbers are not
portable across models or machines and should be treated only as local
observations.

## Operational guidance

- Train with data representative of the traffic the router will actually see.
- Keep an explicit fallback when a learned model is unavailable or uncertain.
- Re-evaluate after changing the base model, labels, embedding model, provider
  pool, prompt format, or preprocessing.
- Monitor routing distribution and downstream quality after rollout; offline
  accuracy does not guarantee production behavior.
- Roll out new router models gradually and keep the previous artifact available
  for rollback.

## Next steps

- [ML-Based Model Selection](./ml-model-selection)
- [Model Performance Evaluation](./model-performance-eval)
- [Signals](/docs/tutorials/signal/overview)
- [Routing Pipeline](/docs/overview/signal-driven-decisions)

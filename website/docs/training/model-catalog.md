---
title: Vela Model Catalog
sidebar_label: Model Catalog
---

# Vela model catalog

[Vela 1.0](https://huggingface.co/collections/llm-semantic-router/vela-10-router-models-6aa555ba70cc6997d6d67798)
is the model family for intelligent routing. Its eleven releases share the
307M-parameter Vela Encoder foundation and cover request understanding,
safety, retrieval, and reranking.

## Choose a model

| Model | Use it to |
| --- | --- |
| [Encoder](https://huggingface.co/llm-semantic-router/Vela-1.0-Encoder-307M) | Train a new task on the shared foundation |
| [Domain](https://huggingface.co/llm-semantic-router/Vela-1.0-Encoder-307M-Domain) | Classify requests into 14 subject areas |
| [Guard](https://huggingface.co/llm-semantic-router/Vela-1.0-Encoder-307M-Guard) | Detect prompt injection and jailbreak attacks |
| [Safety](https://huggingface.co/llm-semantic-router/Vela-1.0-Encoder-307M-Safety) | Detect unsafe content |
| [Hazard](https://huggingface.co/llm-semantic-router/Vela-1.0-Encoder-307M-Hazard) | Identify 12 content-risk categories |
| [PII](https://huggingface.co/llm-semantic-router/Vela-1.0-Encoder-307M-PII) | Locate 17 types of personal information |
| [FactCheck](https://huggingface.co/llm-semantic-router/Vela-1.0-Encoder-307M-FactCheck) | Decide when an answer needs factual verification |
| [Feedback](https://huggingface.co/llm-semantic-router/Vela-1.0-Encoder-307M-Feedback) | Recognize satisfaction, clarification, correction, revision, or no feedback |
| [Modality](https://huggingface.co/llm-semantic-router/Vela-1.0-Encoder-307M-Modality) | Choose text, image, or combined output |
| [Embedding](https://huggingface.co/llm-semantic-router/Vela-1.0-Encoder-307M-Embedding) | Compare requests and retrieve relevant documents |
| [Reranker](https://huggingface.co/llm-semantic-router/Vela-1.0-Encoder-307M-Reranker) | Reorder retrieved candidates by relevance |

Guard detects attempts to redirect instructions. Safety detects content risk;
Hazard identifies its category. Use these models together when your application
needs both prompt-attack protection and content policies.

Embedding and Reranker offer four encoder depths and five dimensions so you
can balance quality, latency, and memory. See
[embedding and reranking](./mmbert-32k-models) for how to choose.

## Deploy or customize

To run the published models, follow [Use Vela models](../tutorials/global/vela-models.md).
That guide covers default downloads and router configuration. Model cards
provide standalone quickstarts and evaluation results.

To adapt a model to your own data, start with the
[training overview](./training-overview). The shared Encoder is a training
foundation; use a task checkpoint for a routing signal.

## Earlier releases and multimodal models

The earlier [mmBERT classifier collection](https://huggingface.co/collections/llm-semantic-router/mom-multilingual-class)
and [mmBERT embedding collection](https://huggingface.co/collections/llm-semantic-router/mom-multilingual-embed)
remain available for existing integrations and comparisons. Their original
training entrypoints are listed in
[the artifact index](https://github.com/vllm-project/semantic-router/blob/main/src/training/model_artifacts.json).

Vela 1.0 contains text models. Image and audio embedding models have their own
[multimodal training guide](./multimodal-embeddings).

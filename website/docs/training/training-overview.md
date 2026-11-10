---
title: Train Vela Router Models
sidebar_label: Overview
---

# Train the Router's models

The Router asks two kinds of model, and only one of them is worth training for
most applications.

- **The decision model** answers every routing judgment zero-shot: the
  application signals, your own questions, the System One selector, guards and
  judges. A new routing question needs no training: write it as a
  [`systemone` rule](../tutorials/signal/systemone) and tune its predicate
  ([Write custom questions](../system-one/use/custom-questions)). A larger
  [decision model](../system-one/deploy/deployments) is the next step when a
  question needs more accuracy.
- **The embedding model** serves recall only: the response cache, memory and
  RAG find their candidates with it, and no decision reads it. Train an
  embedding model when recall misses relevant items on your data.

## Choose a training workflow

| Your goal | Workflow | What you train |
| --- | --- | --- |
| Improve semantic search, cache recall or memory recall | [Embedding](./mmbert-32k-models#embedding-model-bi-encoder) | Vectors for queries and documents |
| Adapt recall to your domain and publish the model to Hugging Face | [Router embeddings and the Hub](./router-embeddings-huggingface) | LoRA adapters and domain-adapted embedding models |
| Choose the generation models a route calls | [Provider model evaluation](./model-performance-eval) | Nothing: an evaluation of your candidate models |

## Choose your starting checkpoint {#record-the-base-and-task-lineage}

Use [Vela Encoder](https://huggingface.co/vllm-sr/Vela-1.0-Encoder-307M) or
the published [Vela Embedding](https://huggingface.co/vllm-sr/Vela-1.0-Encoder-307M-Embedding)
to start. Download an explicit Hub revision and keep its tokenizer and
configuration together with the weights.

## Prepare examples from your application

Start with real requests and the items you expect them to recall. Include the
languages and input lengths your application will receive, and near-miss
pairs that should not match.

Keep related conversations, documents and translations in the same split.
Use training data to update weights, development data to choose checkpoints,
and a separate test set for the final comparison.

## Train and compare

Run a small training job first to check the data. Then compare your trained
model with the published embedding model using the same inputs and inference
settings: retrieval and semantic-similarity quality, on language and length
slices separately. Measure latency at the input lengths you plan to serve.

## Use the trained model in the router

Export a complete checkpoint with its tokenizer. Declare it as an embedding
deployment and name it where recall uses it
([Embeddings](../model-runtime/guides/embeddings.md)). A store indexed with
another model must be re-indexed, or keep naming the model that indexed it.
Then validate your configuration:

```bash
vllm-sr config validate --config config.yaml
```

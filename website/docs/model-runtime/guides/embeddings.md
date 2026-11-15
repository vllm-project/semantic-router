---
title: Embeddings
description: Embeddings for recall in the response cache, memory, RAG and vector stores.
sidebar_position: 1
---

# Embeddings

An embedding turns text into a vector so the router can recall by meaning.
Embeddings serve recall only, never a routing decision:

- the [response cache](tutorials/plugin/response-cache.md): find the cached
  requests a request may reuse;
- the memory and RAG sources of the [context plugin](tutorials/plugin/context.md)
  and vector stores: find stored facts and documents (`match.recall`).

The default model is Vela 1.0 Embedding. It reads up to 32,768 tokens and can
return smaller vectors from earlier layers when you need speed: layers 3, 6,
11 and 22, and 64, 128, 256, 512 or 768 dimensions.

## Turn it on

Enabling a feature that recalls is enough: the router runs Vela Embedding on
the CPU. `global.model_catalog.embeddings.default` names the deployment that
embeds, the way `system.decision_model` names the decision model's: the
pre-declared `vela-embedding` unless you change it. The deployment starts only
when the response cache, memory or a vector store recalls with it; a
configuration without recall never starts it. No decision reads embeddings: to
route on what an image shows, ask the decision model a question that reads it
([media inputs](../../system-one/get-started/concepts#media-inputs)).

## Choose where it runs

An embedding model is a deployment like any other: a local `model_runtime`
deployment, or a remote OpenAI-compatible embeddings API. Every recall feature
embeds with the deployment `embeddings.default` names. To change its device or
input budget, declare the deployment under the same name:

```yaml
global:
  model_catalog:
    deployments:
      vela-embedding:
        provider: model_runtime
        artifact: vllm-sr/Vela-1.0-Encoder-307M-Embedding
        device: cpu
        input:
          max_tokens: 8192
          overflow: truncate
    embeddings:
      default: {deployment: vela-embedding}
```

Features may read different views of the same model, so existing indexes stay
valid: with Vela Embedding, memory stores 256-dimension vectors and the
in-memory response cache 256-dimension vectors of layer 6, while vector stores
and RAG sources use all 768 dimensions. They share one deployment; each view is
its own vector space and is stored separately.

## Stores indexed with another model

A store or collection whose vectors another model made names that model's
deployment, and the dimension its vectors carry, in an `embedding` block: the
response cache, memory and vector store under `global.stores`, and a RAG
source over an external collection in its `backend_config`:

```yaml
global:
  model_catalog:
    deployments:
      omni-nano:
        provider: model_runtime
        artifact: vllm-sr/Vela-1.0-Omni-Nano
        revision: 2ff2d66385dbdd661a560ec3e8bcb45a0527d92e
        device: cpu
        input: {overflow: truncate}
  stores:
    vector_store:
      enabled: true
      backend_type: memory
      embedding:
        deployment: omni-nano
        dimension: 384
```

A store without an `embedding` block embeds with `embeddings.default`. The
Router checks every deployment when it loads the configuration: it must be
declared and embed text, and it must answer with the width of the store's
index, the block's `dimension` or the vector field of a persistent backend.
[Images and audio](./multimodal.md) covers the Omni models, which also embed
images and audio on the model runtime.

## Use an external embedding service

Any OpenAI-compatible `/embeddings` endpoint, for example `vllm serve` with an
embedding model, can replace the local model. Declare it as an
`openai_embeddings` deployment, with the model's facts in `capabilities`, and
name it in `embeddings.default`; the Router posts to `<endpoint>/embeddings`:

```yaml alternative
global:
  model_catalog:
    deployments:
      remote-embed:
        provider: openai_embeddings
        endpoint: https://embedding.example.com/v1
        model: BAAI/bge-m3
        auth: {api_key_env: EMBEDDING_API_KEY}
        timeout_ms: 5000
        max_retries: 2
        capabilities: {dimensions: 1024, modalities: [text]}
    embeddings:
      default: {deployment: remote-embed}
```

`capabilities.dimensions` is the width of its vectors and
`capabilities.modalities` what it embeds; a deployment that declares
modalities without `text` cannot serve recall. The service receives the text
being embedded. Layer views and windows of long text need a local model.

## Check it

These worker-level examples run inside an environment containing `vllm-srun`
(such as the Router image). Classify, embeddings, rerank and bundle are worker
APIs; the Router has no API that returns embeddings, and the instance frontend
publishes System One and decision requests.

```bash
vllm-srun serve vllm-sr/Vela-1.0-Encoder-307M-Embedding --device cpu --port 8100
curl -s localhost:8100/v1/embeddings -H 'content-type: application/json' \
  -d '{"input": ["How do I reset my password?", "I forgot my login password."], "dimensions": 256}'
```

The response is the OpenAI embeddings format. Add
`"options": {"return_meta": true}` to also get `meta.representation`: the
model, layer and dimension that identify the vector space.

## Change the model without mixing vectors

Vectors of different models, layers or sizes are never compared with each
other. After you change any of them, re-index vector stores and re-add the
memories you need, and expect the semantic cache to fill again. See
[Re-embed when the embedding model changes](../deploy/migrate.md#re-embed-when-the-embedding-model-changes).

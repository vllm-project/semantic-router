---
title: Embeddings
description: Embeddings for recall, the response cache, memory, RAG, vector stores and the embeddings API.
---

# Embeddings

An embedding turns text into a vector so the router can recall by meaning.
Embeddings serve recall only, never a routing decision:

- the [response cache](tutorials/plugin/response-cache.md): answer a
  repeated question from the cache;
- the memory and RAG sources of the [context plugin](tutorials/plugin/context.md)
  and vector stores: find stored facts and documents (`match.recall`);
- `/v1/embeddings`.

The default model is Vela 1.0 Embedding. It reads up to 32,768 tokens and can
return smaller vectors from earlier layers when you need speed: layers 3, 6,
11 and 22, and 64, 128, 256, 512 or 768 dimensions.

## Turn it on

Enabling a feature that recalls is enough: the router runs Vela Embedding on
the CPU. `global.model_catalog.embeddings.default` names the deployment that
embeds text, the pre-declared `vela-embedding` unless you change it.
`embeddings.multimodal` names the deployment that embeds images and audio; it
is off until you set it (see [Images and audio](./multimodal.md)).

## Choose where it runs

Every recall feature (the response cache, memory, vector stores, RAG sources
and `/v1/embeddings`) embeds with the deployment `embeddings.default` names.
To change its device or input budget, declare the deployment under the same
name:

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
      default: vela-embedding
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
Router checks every block when it loads the configuration: the deployment must
be declared and embed text, and it must answer with the dimension, its
`capabilities.dimensions` for a remote deployment.

## Use an external embedding service

Any OpenAI-compatible `/embeddings` endpoint, for example `vllm serve` with an
embedding model, can replace the local model for text. Declare it as an
`openai_embeddings` deployment and name it in `embeddings.default`; the Router
posts to `<endpoint>/embeddings`:

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
        capabilities: {dimensions: 1024}
    embeddings:
      default: remote-embed
```

The service receives the text being embedded. Features that need images,
audio, layer views or windows of long text need a local model.

## Check it

These worker-level examples run inside an environment containing `vllm-srun`
(such as the Router image). Classify, embeddings, rerank and bundle are worker
APIs; the instance frontend publishes System One and decision requests.

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
[Re-embed when the embedding model changes](model-runtime/migrate.md#re-embed-when-the-embedding-model-changes).

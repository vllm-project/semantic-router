---
title: Embeddings
description: Embeddings for semantic routing, the semantic cache, memory, RAG, tool selection and knowledge bases.
---

# Embeddings

An embedding turns text into a vector so the router can compare meaning. One
embedding model serves every feature that needs one:

- [embedding signals](tutorials/signal/learned/embedding.md) and
  [knowledge bases](tutorials/signal/learned/kb.md): route on similarity
  to example texts;
- the [semantic cache](tutorials/plugin/response-cache.md): answer a
  repeated question from the cache;
- [memory](tutorials/plugin/memory.md) and [RAG](tutorials/plugin/rag.md):
  find stored facts and documents;
- [tool selection](tutorials/plugin/tool-selection.md): offer the tools
  that fit the request.

The default model is Vela 1.0 Embedding. It reads up to 32,768 tokens and can
return smaller vectors from earlier layers when you need speed: layers 3, 6,
11 and 22, and 64, 128, 256, 512 or 768 dimensions.

## Turn it on

Enabling a feature that needs embeddings is enough; the router runs Vela
Embedding on the CPU. To set the vector size used for routing:

```yaml
global:
  model_catalog:
    embeddings:
      semantic:
        embedding_config:
          model_type: mmbert
          target_layer: 22
          target_dimension: 768
```

`mmbert` names the Vela Embedding slot. Use `qwen3` for Qwen3-Embedding-0.6B
(1,024 dimensions) and `multimodal` for Vela Omni (see
[Images and audio](./multimodal.md)).

## Choose where it runs

The binding is `embedding` and reads vectors (`embedding.v1`). Every service
that needs embeddings (caches, memory, vector stores, tools) uses the global
binding, so declare it once:

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
    bindings:
      embedding:
        deployment: vela-embedding
        contract: embedding.v1
```

Different features may read different views of the same model, for example
the semantic cache a 256-dimension vector of layer 6 and routing a
768-dimension vector of layer 22. They share one deployment; each view is its
own vector space and is stored separately.

## Use an external embedding service

Any OpenAI-compatible `/embeddings` endpoint, for example `vllm serve` with an
embedding model, can replace the local model for text. It replaces the
`embedding` binding above, so leave that binding out:

```yaml alternative
global:
  model_catalog:
    embeddings:
      semantic:
        embedding_config:
          backend: openai_compatible
          model_type: remote
          target_dimension: 1024
        endpoint:
          base_url: https://embedding.example.com/v1
          model: BAAI/bge-m3
          api_key_env: EMBEDDING_API_KEY
          dimensions: 1024
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

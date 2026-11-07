---
title: Embeddings
description: Embeddings for semantic routing, the semantic cache, memory, RAG, tool selection and knowledge bases.
---

# Embeddings

An embedding turns text into a vector so the router can compare meaning. One
embedding model serves every feature that needs one:

- the [semantic cache](tutorials/plugin/response-cache.md): answer a
  repeated question from the cache;
- the memory and RAG sources of the [context plugin](tutorials/plugin/context.md):
  find stored facts and documents.

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

`global.model_catalog.embeddings.default` names the deployment that embeds
text, the pre-declared `vela-embedding` unless you change it. Every service
that needs embeddings (caches, memory, vector stores, tools) uses it. To change
its device or input budget, declare the deployment under the same name:

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

Different features may read different views of the same model, for example
the semantic cache a 256-dimension vector of layer 6 and routing a
768-dimension vector of layer 22. They share one deployment; each view is its
own vector space and is stored separately.

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

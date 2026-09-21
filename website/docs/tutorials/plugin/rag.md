# RAG

## Overview

`rag` retrieves external context for a matched route before generation. Choose
Milvus or Qdrant for direct vector-store retrieval, or use an external HTTP
API, MCP tools, OpenAI file search, the Router's vector-store service, or a
primary/fallback hybrid.

## Key Advantages

- Keeps retrieval local to routes that actually need it.
- Supports backend-specific retrieval settings in one place.
- Avoids forcing every route to inject documents or tool context.

## What Problem Does It Solve?

Some routes need external document retrieval before answering, while most do not. `rag` lets the matched route perform retrieval and injection without globalizing that behavior.

## When to Use

- a route should fetch documents or facts before the final model call
- retrieval should use Milvus, Qdrant, or another explicit backend
- different routes need different retrieval settings

## Configuration

Choose one backend:

| Backend | Use it for | Required backend fields |
| --- | --- | --- |
| `milvus` | Direct retrieval from a Milvus collection | `collection`; optionally reuse the response-cache connection |
| `qdrant` | Direct retrieval from a Qdrant collection | `collection`; optionally reuse the response-cache connection |
| `external_api` | A service with a custom HTTP request contract | `endpoint`, `request_format` |
| `mcp` | Retrieval exposed as an MCP tool | `server_name`, `tool_name` |
| `openai` | OpenAI file search | `vector_store_id`, `api_key` |
| `vectorstore` | The Router-managed vector-store service | `vector_store_id` |
| `hybrid` | A primary backend with an optional fallback | `primary`, plus backend-specific nested configuration |

For `external_api`, `max_response_bytes` caps each response body; omitted or
`0` uses 4 MiB.

For OpenAI `direct_search`, `max_response_bytes` applies the same 4 MiB default
to each vector-store search response.

The examples below show the two direct-store options and the external HTTP
API. For the other backends, start from the field names above and validate the
complete config before deployment.

Add the plugin under `routing.decisions[].plugins`:

**Milvus backend:**

```yaml
plugins:
  - type: rag
    configuration:
      enabled: true
      backend: milvus
      top_k: 5
      similarity_threshold: 0.78
      injection_mode: tool_role
      on_failure: warn
      backend_config:
        collection: docs
        reuse_cache_connection: true
        content_field: content
```

**Qdrant backend:**

```yaml
plugins:
  - type: rag
    configuration:
      enabled: true
      backend: qdrant
      top_k: 5
      similarity_threshold: 0.78
      injection_mode: tool_role
      on_failure: warn
      backend_config:
        collection: docs
        reuse_cache_connection: true
        content_field: content
```

**External API backend:**

```yaml
plugins:
  - type: rag
    configuration:
      enabled: true
      backend: external_api
      top_k: 5
      similarity_threshold: 0.78
      injection_mode: tool_role
      on_failure: warn
      backend_config:
        endpoint: https://search.example.com/query
        request_format: custom
        request_template: '{"query":"${user_content}","top_k":${top_k},"threshold":${threshold}}'
        timeout_seconds: 15
        max_response_bytes: 4194304
```

Retrieved documents become provider-bound context. Apply collection-level
access control and avoid mixing tenants in one unrestricted search scope.
Similarity thresholds are embedding-model specific. See complete examples:
[`milvus.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/rag/milvus.yaml),
[`qdrant.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/rag/qdrant.yaml),
and
[`external-api.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/rag/external-api.yaml).

## Neural reranking

The `vectorstore` backend can rerank its structured search hits with a local
Vela pair scorer before formatting the context. Declare the deployment and
the recipe-local `rag.reranker` binding, then opt the route into `rerank`:

```yaml
global:
  model_catalog:
    deployments:
      document-ranker:
        artifact: models/Vela-1.0-Encoder-307M-Reranker
        provider: candle
        device: cpu
        precision: native
        input:
          max_tokens: 4096
          overflow: reject
routing:
  model_bindings:
    rag.reranker:
      deployment: document-ranker
      contract: relevance_scores.v1
      adapter: vela_reranker
      pair_scorer:
        layer: 22
        dimension: 768
```

Add this plugin to a decision in the same recipe:

```yaml
plugins:
  - type: rag
    configuration:
      enabled: true
      backend: vectorstore
      backend_config:
        vector_store_id: vs-your-documents
      top_k: 10
      rerank:
        top_k: 3
      on_failure: block
```

`top_k` retrieves candidates; `rerank.top_k` limits the reordered hits injected
into the prompt. Omit the latter to retain every candidate. Higher raw relevance
logits rank first, with equal scores retaining retrieval order. Document IDs,
chunk IDs and retrieval similarity scores stay intact. Reranker logits are
uncalibrated and do not replace the embedding similarity threshold.

The scorer uses the tokenizer's query/document pair template. The token budget
includes both texts and special tokens; overflow is rejected without truncating
either text. Loading validates the selected trained layer and dimension; zero
selects the artifact's actual full depth or width. CPU cost grows with candidate
count and pair length, so choose an explicit deployment budget.

Candle artifacts must include encoder weights, `config.json`, `tokenizer.json`,
`matryoshka_config.json` and `classification_heads.safetensors`. An ORT deployment
selects a complete graph through the binding's `head` field. Its embedded
`semantic_router.pair_scorer` metadata must declare the actual exit and
`relevance_logit` contract; the graph filename is not proof of its semantics.

Only reachable recipes with an enabled `rerank` plugin load a scorer. Missing
models, invalid scores and input-limit failures follow the existing RAG
`on_failure` policy. Cached context is isolated by recipe, embedding identity
and scorer identity. Runtime tracing records actual rerank latency and scores;
route preview does not execute retrieval or invent a reranker timing. Other RAG
backends currently reject `rerank` until they expose structured candidates.

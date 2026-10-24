# RAG

## Overview

`rag` retrieves external context for a matched route before generation. Choose
Milvus or Qdrant for direct vector-store retrieval, or use an external HTTP
API, OpenAI file search, the Router's vector-store service, or a
primary/fallback hybrid. `mcp` is rejected at startup until the router has an
MCP tool invoker.

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
| `mcp` | Rejected at startup until an MCP tool invoker exists | `server_name`, `tool_name` |
| `openai` | OpenAI file search | `vector_store_id`, `api_key` |
| `vectorstore` | The Router-managed vector-store service | `vector_store_id` |
| `hybrid` | A primary backend with an optional fallback | `primary`, plus backend-specific nested configuration |

`hybrid.strategy` defaults to `sequential`: the fallback runs only after the primary fails or returns no context. `parallel` starts both lookups and returns as soon as the primary has context. It does not rank the two results. The fallback is used only when the primary fails or is empty.

For `external_api`, `max_response_bytes` caps each response body; omitted or
`0` uses 4 MiB.

For OpenAI `direct_search`, `max_response_bytes` applies the same 4 MiB default
to each vector-store search response.

The examples below show the two direct-store options. For the
other backends, start from the field names above and validate the complete
config before deployment.

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

Retrieved documents become provider-bound context. Apply collection-level
access control and avoid mixing tenants in one unrestricted search scope.
Similarity thresholds are embedding-model specific. See complete examples:
[`milvus.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/rag/milvus.yaml)
and
[`qdrant.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/rag/qdrant.yaml).

## Neural reranking

The `vectorstore` backend can rerank its structured search hits with a local
Vela pair scorer before formatting the context. Declare the deployment and
the recipe-local `rag.reranker` binding, then opt the route into `rerank`:

```yaml
global:
  model_catalog:
    deployments:
      document-ranker:
        provider: model_runtime
        artifact: vllm-sr/Vela-1.0-Encoder-307M-Reranker
        device: cpu
        input:
          max_tokens: 4096
          overflow: reject
routing:
  model_bindings:
    rag.reranker:
      deployment: document-ranker
      contract: relevance_scores.v1
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

The reranker runs in the [model runtime](../../model-runtime/guides/rerank.md).
At startup the router checks that the model declares the selected exit, so a
`pair_scorer` the model was not trained for is refused before traffic arrives.

Only reachable recipes with an enabled `rerank` plugin load a scorer. Missing
models, invalid scores and input-limit failures follow the existing RAG
`on_failure` policy. Cached context is isolated by recipe, embedding identity
and scorer identity. Runtime tracing records actual rerank latency and scores;
route preview does not execute retrieval or invent a reranker timing. Other RAG
backends currently reject `rerank` until they expose structured candidates.

### SystemOne relevance scoring

To score retrieved documents with a Decision Model, explicitly select
`decision.v1` for the same recipe-local binding:

```yaml
global:
  model_catalog:
    deployments:
      relevance-judge:
        provider: model_runtime
        artifact: vllm-sr/Vela-2.0-0.3B
        device: cpu
routing:
  model_bindings:
    rag.reranker:
      deployment: relevance-judge
      contract: decision.v1
```

Keep the vectorstore RAG plugin and its `rerank.top_k` configuration above.
Omit `adapter`, `head`, `mapping_path` and `pair_scorer`: this binding asks a
Score question through the shared runtime. Readiness and Score capability are
checked at preparation, including a full-input warmup. A task-specific training
label is not required. Existing `relevance_scores.v1` bindings retain their
cross-encoder behavior.

Each candidate is judged against the query, with four ordered levels: unrelated,
related but not answering, partially answering, and directly answering. Scores
are in `[0, 3]` and are reported as `decision_relevance`. They are uncalibrated
ordinal scores, not probabilities, embedding similarities or cross-encoder
logits. Equal scores preserve retrieval order and all file/chunk metadata.

Scoring starts after retrieval and before context formatting. Each document is
a separate state; compatible calls use the existing runtime bundle and fusion
implementation. A bundle does not imply one model forward. The first slice
bounds a batch to 32 candidates, 1 MiB of query/document UTF-8 bytes (counting the
query once per candidate), and 30 seconds of inference, or the caller's earlier
deadline. Runtime token limits still apply. Input is never silently truncated.

Every answer must prove `input_coverage: complete`. Missing proof, invalid
scores, unavailable models and timeouts follow the configured RAG `on_failure`
policy. Select `block` when a failed scorer must prevent retrieval context from
being used. Candidate text remains untrusted evidence, never a tool command.

Retrieval cache identity includes model content/revision, question wording and
levels, score contract and read policy. A deployment without a model-content
digest disables this retrieval cache path. Diagnostics report coverage and
runtime-reported input tokens separately from native paired-token usage.

Use the [RAG relevance comparison driver](https://github.com/vllm-project/semantic-router/tree/main/bench/rag_relevance)
to compare identical labeled candidates against retrieval order and the existing
cross-encoder. Evaluate quality and latency for your documents before selecting
this backend; capability support alone establishes neither.

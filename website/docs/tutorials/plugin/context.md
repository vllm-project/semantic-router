# Context

## Overview

`context` is the route-local plugin that assembles the request's context. Its
`sources` retrieve documents (`rag`) and the user's memories (`memory`), each
kept by its `match` block. `compress` shrinks the request's own segments,
judged by its `match` block, until the request fits the `budget`. `injection`
places what the sources kept. A memory source also writes the turn to memory
after the response.

## Key Advantages

- One plugin owns retrieval, memory and compression, in a fixed order.
- One decision-model call per stage: the relevance judge keeps a RAG source's
  documents and ranks the segments compression keeps.
- The budget is one number, the most input tokens the request keeps, or the
  model's context window less the reserved output.
- Routing signals still evaluate the original request: the plugin changes only
  the conversation sent to the selected model.

## What Problem Does It Solve?

Documents, memories, long histories and tool outputs compete for the same
context window. Retrieving them in separate plugins left each with its own
thresholds, its own reranker and its own idea of the budget. The `context`
plugin retrieves the sources, lets the decision model judge what is relevant
and compresses what remains within one budget.

## When to Use

- a route answers from indexed documents
- a route personalizes answers with the user's memories
- a route's tool outputs or history are larger than the answer needs
- a route should turn memory off, with `enabled: false`

## Configuration

Add the plugin under `recipes[].decisions[].plugins`:

```yaml
plugins:
  - type: context
    configuration:
      enabled: true
      sources:                          # retrieved in order, at most one of each type
        - type: rag
          backend: milvus
          backend_config: {collection: docs, reuse_cache_connection: true}
          match:                        # the default, written out
            recall: {method: embedding, top_k: 20, min_similarity: 0.3}
            judge: {preset: relevance, top_k: 6, min_score: 0.5}
          max_context_length: 10000     # characters
          on_failure: skip              # skip | warn | block
        - type: memory
          store: {auto: true}           # write the turn after the response
      compress:
        mode: auto                      # auto | always
        targets:
          tool_outputs: {mode: extractive, min_tokens: 2000, target_tokens: 1000}
        failure_mode: fail_open         # fail_open | fail_closed
      budget:
        max_tokens: auto                # or a number of input tokens
        reserve_output_tokens: auto
      injection:
        mode: tool_role                 # tool_role | system_prompt
```

An enabled plugin needs `sources` or `compress`; `budget` needs `compress`.
`enabled: false` turns context off for the decision, global memory included.
A decision without a `context` plugin reads global memory when
`global.stores.memory.enabled` is true; a decision with one reads only the
sources it lists.

### RAG sources

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

`hybrid.strategy` defaults to `sequential`: the fallback runs only after the
primary fails or returns no documents. `parallel` starts both lookups and
returns as soon as the primary has documents. For `external_api`,
`max_response_bytes` caps each response body; omitted or `0` uses 4 MiB. For
OpenAI `direct_search`, the same default applies to each vector-store search
response.

The source's `match` block keeps the documents. By default the backend
recalls up to 20 documents with an embedding similarity of at least 0.3, and
one call to the deployment that answers the judge (the decision model unless
`match.judge.deployment` names another) asks the `relevance` question about
each, one batch entry per document: the 6 most relevant documents with a
score of at least 0.5 are kept, in the judge's order. A written block replaces
the default whole: a block with only `recall` keeps the recalled documents in
the backend's order. A local knowledge base, a `vectorstore` source over the
in-memory vector store, can also recall with `method: bm25`, which embeds no
query, or be judged whole by a block with only a `judge`: the judge reads its
newest chunks, at most `max_candidates` (64 by default). A source over an
external vector database recalls by embedding. External
API documents carry the score the service returns (Pinecone `score`, Weaviate
`1 - distance`, Elasticsearch `_score`, a custom `score`); a document without
one counts as similarity 1.

`on_failure` is what a failed retrieval does: `skip` (the default) goes on
without the documents, `warn` also logs a warning, `block` refuses the request
with `503`. A retrieval that finds nothing is not a failure. `cache_results`
keeps the kept documents for `cache_ttl_seconds` (default 3600) per query,
source and match block.

Retrieved documents become provider-bound context. Apply collection-level
access control and avoid mixing tenants in one unrestricted search scope.
Similarity thresholds are embedding-model specific.

### Memory sources

A memory source reads the store configured under `global.stores.memory`:
Milvus (the default), Valkey or Qdrant. See the
[Stores and Tools](../global/stores-and-tools) tutorial for the store, the
[Valkey Memory deployment guide](../../installation/valkey-memory) and the
[Qdrant deployment guide](../../installation/qdrant).

Its default `match` block recalls only: the store's
`default_retrieval_limit` and `default_similarity_threshold`, else 5 memories
with a similarity of at least 0.7. A written block replaces the default, and a
`judge` asks the decision model which memories are relevant. A block with only
a `judge` suits a small per-user memory: the judge reads the user's newest
memories, at most `max_candidates` (64 by default).

- `store.auto` writes the turn to memory after the response; unset follows
  `global.stores.memory.auto_store`. A request may turn the write off, and on
  only where `store.auto` is unset.
- `hybrid_search` fuses the store's vector and lexical scores by `hybrid_mode`
  (`weighted` or `rrf`).
- `reflection` overrides the store's reflection gate for the decision. When
  reflection is enabled (the default heuristic gate), retrieved memories are
  trimmed to `max_inject_tokens`. The gate estimates size with a script-aware
  heuristic, about 1.3 tokens per Latin word and 1.5 tokens per Han, Hiragana,
  Katakana or Hangul character, rounded up.

Memory persists request-derived content and sends retrieved memories to the
selected model. Choose user and tenant isolation, retention, authentication
and transport security appropriate for that data. A memory failure never
refuses the request: the Router records it in the replay record and goes on.

### Compression and budget

`compress` shrinks the request's own segments: tool outputs, history and the
documents and memories the sources injected. Its `match` block judges which
segments matter. Each segment is a candidate about the latest user message: by
default BM25 ranks the segments and one `relevance` call scores the best 64
of them (`judge.max_candidates`), one batch entry each; the least relevant
segments, and those the judge did not read, shrink first, until the request
fits the budget. `match: {recall: {method: bm25}}` ranks them lexically
instead, with no call. Without a judge answer the segments are ranked
lexically.

- `mode: auto` (the default) compresses a request over the budget, or with
  segments over their target's `min_tokens`; `always` compresses every request.
- `targets` sets each segment kind's mode (`preserve`, `extractive`,
  `truncate`, `recoverable`) and its `min_tokens` and `target_tokens`.
  `target_tokens` must be lower than `min_tokens`. History, RAG, memory and
  current-user text are preserved unless their target is enabled.
- `budget.max_tokens` is the most input tokens the request keeps: a request
  over it is compressed down to it. `auto` (the default) is the model's context
  window less `budget.reserve_output_tokens`, by default the request's output
  limit. The reserve can grow but never drops below the requested output.
- `failure_mode: fail_open` (the default) sends the request uncompressed when a
  rewrite fails; `fail_closed` refuses it. Neither bypasses the budget check:
  a request whose protected content cannot fit gets `400` with
  `context_length_exceeded`.

The budget uses a conservative UTF-8 byte estimate, not an exact tokenizer
count. System and developer instructions, safety text, tool calls, schemas,
media, citations and JSON user payloads are never truncated, and tool call and
result IDs stay paired. With `current_user: {mode: truncate}` the latest
plain-text user message keeps its beginning and end, separated by
`[... context omitted by route compression ...]`; the middle is omitted, not
summarized.

JSON strings compress only through their string leaves, OpenAI array content
compresses text blocks and keeps images, and Anthropic `tool_result` content
keeps `tool_use_id`, `is_error`, images and cache-control metadata.
`recoverable` targets store the original content in a shared Redis or Valkey
store, inject the reserved `vsr_context_retrieve` tool and make the
non-streaming follow-up call in process. Recovery is request- and
trusted-user-scoped, bounded by TTL, bytes and retrieval count.

`request_controls` lets a caller send `bypass` (skip compression) or
`target=N` (the tool-output target, clamped by `max_target_tokens`) in the
`x-vsr-compression-control` header; they are ignored unless the route enables
them.

### Injection

`injection.mode: tool_role` (the default) adds one tool call and its result per
source after the last user message: `vsr_rag_context` for documents and
`vsr_memory_context` for memories, in source order. `system_prompt` adds them
to the system prompt instead. Compression recognizes both as retrieved
context.

## Management and preview

- `GET /api/v1/plugins/context/compression/capabilities`
- `GET /api/v1/plugins/context/compression/health`
- `GET /api/v1/observability/plugins/context/compression/stats`
- `POST /api/v1/plugins/context/compression/preview`
- `POST /api/v1/plugins/context/preview` (a RAG source's retrieval)
- `POST /api/v1/storage/context-recovery/invalidate`

The compression preview ranks segments lexically and returns only plans,
target indexes, token counts, scores, warnings and skip reasons; it never
returns source or omitted content and requires `compression.preview`. Scoped
recovery invalidation requires `compression.manage`.

## Observability

Each stage's judge is counted in `vsr_match_judge_total{feature="rag"}`,
`{feature="memory"}` or `{feature="compress"}`. The request's
[router replay](./router-replay) record keeps the RAG backend and the memory
outcome, including a memory failure the request went on without. Memory
retrieval is counted by
`llm_memory_retrieval_total` with `backend` and `status` labels (`hit`, `miss`,
`error`). Compression metrics report whether it ran, its strategy, the
before and after counts and the omissions; these are estimates, not measured
billing-token savings. Raw omitted content and recovery keys are not included.

## Migrating

The `rag`, `memory` and `context_compression` plugins are sources, `compress`
and `budget` of the `context` plugin. A configuration that still uses them, or
their removed fields, fails at load with a one-line hint, and
`vllm-sr config migrate` rewrites it:

- RAG `top_k` and `similarity_threshold` become `match.recall`, and
  `rerank.top_k` becomes `match.judge.top_k`: the Vela cross-encoder reranker,
  its `rag.reranker` binding and `provider: rerank` deployments are removed,
  and the relevance judge keeps the documents.
- RAG `min_confidence_threshold` retrieved only when a fact-check rule
  matched: migrate ANDs a `fact_check` condition into the decision's rules.
- Memory `retrieval_limit` and `similarity_threshold` become `match.recall`,
  and `auto_store` becomes `store.auto`.
- Compression `scoring` is removed: `bm25` becomes
  `match: {recall: {method: bm25}}`, and the embedding scoring is replaced by
  the relevance judge. `budget.target_tokens` becomes `budget.max_tokens`.

Complete examples:
[`config/fragments/plugin/context/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/plugin/context).

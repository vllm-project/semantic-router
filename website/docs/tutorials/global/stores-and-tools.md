# Stores and Tools

## Overview

This page covers the shared storage and tool blocks inside `global:`.

These settings back route-local plugins and router-wide tool behavior.

## Key Advantages

- Centralizes shared backing stores instead of repeating them per route.
- Keeps response cache, memory, retrieval, and tool catalogs consistent.
- Lets route-local plugins stay small and focused.
- Makes shared infrastructure dependencies explicit.

## What Problem Does It Solve?

Route-local plugins often depend on shared storage or tool state. If those dependencies are configured ad hoc inside each route, the system becomes inconsistent and harder to operate.

These `global:` blocks solve that by defining shared backing services once.

## When to Use

Use these blocks when:

- multiple routes depend on the same response cache or memory backend
- retrieval features need one shared vector store
- the router should expose one shared tool catalog
- backing-store configuration belongs to the whole router rather than one route

## Configuration

### Response Cache

```yaml
global:
  stores:
    response_cache:
      enabled: true
      backend_type: memory
      similarity_threshold: 0.8
      polarity_guard:
        mode: lexical          # lexical | nli | lexical+nli
        nli:
          contradiction_threshold: 0.5
```

#### Negation guard

Bi-encoder similarity cannot tell *"turn on dark mode"* from *"turn off dark
mode"*: opposite-meaning queries often score above `similarity_threshold`
while genuine paraphrases score below it, so raising the threshold does not
fix the false hit. `polarity_guard` verifies the winning candidate before the
in-memory backend serves it:

- `lexical` (default): the model-free tier that catches negation cues and
  known antonym swaps. It is always on and needs no model.
- `nli` / `lexical+nli`: additionally runs the router's NLI model once per
  lookup on the single best candidate and rejects the hit when the
  contradiction probability exceeds `nli.contradiction_threshold`. The tier
  reuses the hallucination explainer
  (`global.model_catalog.modules.hallucination_mitigation.explainer`, by default
  `tasksource/ModernBERT-base-nli`); the native binding holds one NLI model, so
  the guard cannot bind a different one. Config loading fails when an NLI mode
  is selected without that model. Expect roughly 70 ms per verified hit on CPU;
  a cache hit still saves a full generation. If the model errors at lookup
  time the guard fails open: the hit is served and a
  `cache_polarity_nli_skipped` warning is logged.

Rejections are logged as `cache_negation_reject` with `tier: nli`, count as
misses, and still surface the rejected score on `x-vsr-cache-similarity`. Remote
and hybrid cache backends do not run the guard.

### Memory

The memory store supports three backends: `milvus` (default), `valkey`, and `qdrant`.

**Milvus backend** (default):

```yaml
global:
  stores:
    memory:
      enabled: true
      milvus:
        address: milvus:19530
        collection: agentic_memory
        dimension: 384
```

**Valkey backend** (requires Valkey with Search module):

```yaml
global:
  stores:
    memory:
      enabled: true
      backend: valkey
      valkey:
        host: valkey
        port: 6379
        dimension: 384
        collection_prefix: "mem:"
        index_name: mem_idx
        metric_type: COSINE
```

**Qdrant backend**:

```yaml
global:
  stores:
    memory:
      enabled: true
      backend: qdrant
      qdrant:
        host: qdrant
        port: 6334
        collection: agentic_memory
        dimension: 384
      embedding_model: bert
      default_retrieval_limit: 5
      default_similarity_threshold: 0.70
```

For full deployment instructions, see:

- [Valkey Agentic Memory](../../installation/valkey-memory) — Docker, Kubernetes, config reference, tuning, and troubleshooting
- [Qdrant](../../installation/qdrant) — Docker, Kubernetes, config reference, tuning, and troubleshooting
- `config/runtime/memory/` for backend-specific configuration references

When an external model with `model_role: memory_rewrite` is configured, its
`max_response_bytes` limits each query-rewrite response. An omitted or
non-positive value uses the 1 MiB default.

#### Write path bounds

Response handling does not wait for Memory persistence to complete. Identity
checks and capacity reservation precede bounded history snapshots; encoding and
writes run in the background. Other response-path Replay operations remain
synchronous.

Configure `global.stores.memory.persistence`:

| Field | Meaning | Default |
| --- | --- | --- |
| `timeout_seconds` | Seconds from reservation to timeout, including preparation, queue wait, and writing | 30 |
| `concurrency` | Worker slots, including preparation and writes | 8 |
| `queue` | Reserved attempts waiting for a worker | 64 |
| `shutdown_grace_seconds` | Seconds to drain writes on reload or shutdown before cancellation | 5 |

Omit a field or set it to `0` to take the default.

History is limited to 256 messages/items across request and retained Responses
history, 1 MiB of payload, 4096 structural nodes, and 32 nested content levels.
Exceeding a limit skips persistence with `skipped` / `history_too_large` and
`fail_open=true`, preserving the model response without truncating history.
Missing user identity skips preparation. Background contexts retain only span
context and tracestate.

For requests with a Router Replay record, accepted attempts reserve capacity for
`scheduled` and one terminal receipt, protecting both from queue saturation.
Storage errors, shutdown drain expiry, or process crashes can still lose receipts.
Exhausted persistence or receipt capacity rejects new writes with `queue_full` or
`receipt_queue_full`; retired pools use `shutting_down`. These remain fail-open
and are logged by request ID. Monitor
`llm_plugin_execution_total{plugin_type="memory_persistence", status="rejected"}`.

Timeout and cancellation report one terminal outcome even while queued; cancelled
jobs do not start. Native embedding calls cannot be interrupted, so active work
retains its worker slot and resources until exit. Cancellation does not undo
writes already accepted by a backend.

### Vector Store

```yaml
global:
  stores:
    vector_store:
      enabled: true
      backend_type: milvus
      metadata_store: postgres
```

Supported backends: `memory`, `milvus`, `llama_stack`, `valkey`, `qdrant`.

`metadata_store` controls the registry for vector-store and uploaded-file
metadata. Use `postgres` for restart-safe local or production-like stacks; the
CLI local runtime will provision Postgres and fill `metadata_postgres` connection
defaults when `metadata_store: postgres` is set. Use `memory` only for ephemeral
local experiments because store and file metadata is lost on router restart.

### Tools

```yaml
global:
  integrations:
    tools:
      enabled: true
      top_k: 3
      tools_db_path: config/runtime/tools/tools_db.json
```

## Data and Security

- Cache, memory, and vector stores can contain prompts, responses, embeddings,
  retrieved documents, or extracted memories. Configure authentication,
  encryption, retention, and tenant/user scope for the selected backend.
- Embedding dimensions must match existing collections. Rebuild or migrate an
  index when the embedding model or dimension changes.
- Tool retrieval controls what is shown to a model; it does not authorize tool
  execution. Enforce permissions at the tool service.
- See
  [complete backend examples](https://github.com/vllm-project/semantic-router/tree/main/config/runtime)
  and the full configuration contract in
  [`config/config.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml).

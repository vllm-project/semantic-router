# Memory

## Overview

`memory` is a route-local plugin for retrieving and storing conversation memory.

## Key Advantages

- Keeps memory behavior local to the routes that benefit from it.
- Supports retrieval and auto-store in one plugin.
- Separates route-local memory policy from shared backing-store config.

## What Problem Does It Solve?

Not every route should pay the complexity or privacy cost of retrieval memory. `memory` lets one matched route retrieve and store conversation context while the shared store remains configured under `global.stores.memory`. Session-aware model stability is a separate Router Learning adaptation configured under `global.router.learning`.

## When to Use

- a route should retrieve prior conversation context
- the route should automatically store useful new turns
- memory settings should stay local to one route family

## Configuration

The memory plugin requires a backing store configured under `global.stores.memory`. The router supports three backends:

- **Milvus** (default) — distributed vector database, best for large-scale production
- **Valkey** — lightweight single-binary option using the Search module, best for dev/test or existing Valkey infra
- **Qdrant** — single-binary with gRPC, simpler ops than Milvus, good for small-to-large workloads

See the [Stores and Tools](../global/stores-and-tools) tutorial for global memory configuration, the [Valkey Memory deployment guide](../../installation/valkey-memory) for Valkey-specific setup, or the [Qdrant deployment guide](../../installation/qdrant) for Qdrant-specific setup.

Add the plugin under `routing.decisions[].plugins`:

```yaml
plugins:
  - type: memory
    configuration:
      enabled: true
      retrieval_limit: 5
      auto_store: true
```

Memory can persist request-derived content and send retrieved memories to the
selected model. Choose user/tenant isolation, retention, authentication, and
transport security appropriate for that data. The omitted per-decision
threshold inherits the global setting; calibrate that value for the selected
embedding model and search mode before adding an override. See a complete example:
[`config/fragments/plugin/memory/session-memory.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/memory/session-memory.yaml).

### Reflection and `max_inject_tokens`

When reflection is enabled under `global.stores.memory.reflection` (the default
heuristic gate), retrieved memories are trimmed to `max_inject_tokens` before
injection. The gate estimates size with a script-aware heuristic, not the
upstream model tokenizer: about **1.3 tokens per Latin word unit** and about
**1.5 tokens per Han, Hiragana, Katakana, or Hangul character**, with the
combined estimate rounded **up** (`ceil`). Space-free CJK text therefore counts
many characters toward the budget instead of collapsing to a single whitespace
word. Operators tuning `max_inject_tokens` for Chinese, Japanese, or Korean
traffic should expect tighter trimming than the old English-only estimate.
Thai and other scripts without spaces are not fully segmented here. Exact
tokenizer accounting for routing budgets is tracked separately ([#3050](https://github.com/vllm-project/semantic-router/issues/3050)).

## Observability

Router Memory exposes bounded Prometheus metrics on the Router scrape endpoint.
Retrieval volume is tracked by `llm_memory_retrieval_total` with `backend` and
`status` labels (`hit`, `miss`, `error`). Latency and result counts use separate
histograms without per-user labels. If you upgrade from a release that labeled
memory metrics with `user_id`, follow the [Router Memory Prometheus label release
note](../../release-notes/router-memory-prometheus-labels).

## In-memory Store capabilities and lifecycle

The Go `InMemoryStore` implements the memory `Store` interface for tests and
proofs of concept. It is not a selectable `global.stores.memory` backend; the
three configured backends above remain the runtime choices. The following
inventory covers this implementation only, as an initial slice of
[the backend capability and lifecycle work](https://github.com/vllm-project/semantic-router/issues/4393).

| Operation | Enabled behavior and supported options | Observed behavior after `Close` |
| --- | --- | --- |
| `Store` | Insert by unique ID; reject duplicates. Use a supplied embedding or generate one through the configured provider. Assign `CreatedAt` if absent. | No data is stored and no embedding is requested. Currently returns `nil` for an uncanceled context; see the unresolved error contract below. |
| `Retrieve` | Cosine similarity, score descending; optional `UserID`, `ProjectID`, and `Types` filters. Apply the supplied `Threshold`; truncate only for a positive `Limit`. Hybrid search and adaptive threshold options are not implemented. | No results and no embedding work; currently returns `nil, nil`. |
| `Get` | Look up an ID; missing IDs return an error. | Disabled error, no memory. |
| `Update` | Require an existing ID. Update content, type, and update time; regenerate the embedding when content changes. Apply nonempty project and source values. | Disabled error. |
| `List` | Require `UserID`; optional `Types` filter. Use the shared limit/offset and ordering contract from [#4325](https://github.com/vllm-project/semantic-router/issues/4325). | Disabled error, no page. |
| `Forget` | Delete an existing ID; missing IDs return an error. | Disabled error. |
| `ForgetByScope` | Match `UserID` exactly, plus optional `ProjectID` and `Types`; no matches is a successful no-op. Callers must supply a user ID; this implementation does not reject an empty one. | Disabled error. |
| `IsEnabled` | `true` immediately after construction; no background initialization. | `false`. |
| `CheckConnection` | `nil`; there is no external connection or remote health probe. | Disabled error. |
| `Close` | Clear retained memories and disable the instance. | Repeated calls succeed. |

Each instance has its own process-local map. Data is visible to subsequent
operations on that instance, but a new instance starts empty; there is no disk
persistence or reconnect/restart recovery. `Store`, `Get`, `List`, and retrieval results
retain or return memory pointers rather than copies, so callers must not mutate
shared records concurrently. The store owns its map, while the embedding
provider belongs to the caller and can be reused by another store after close.
Drain active callers before closing: this slice verifies sequential shutdown,
not a concurrent close/drain or router reload guarantee.

The closed-state error behavior is inconsistent: `Store` and `Retrieve` currently
return success-shaped no-ops, while the other data operations return disabled
errors. Whether to preserve these no-ops or align the errors is an open contract
question in #4393. The lifecycle tests assert no retained data or embedding work
after close without establishing the current `nil` errors as a write/retrieval
guarantee.

### Verification coverage

`src/semantic-router/pkg/memory/inmemory_store_lifecycle_test.go` uses small
precomputed vectors and an injected function provider, with no database or model
downloads. It checks resource release, repeated close, disabled operations,
post-close writes, fresh-instance isolation, and provider reuse. Cancellation
coverage stays in `embedding_cancellation_test.go`, including
`TestCanceledStoreDoesNotPersistPrecomputedEmbedding`; this slice does not add a
new cancellation or pagination contract. After building the native libraries
required by the Router module, run the focused checks from its directory:

```bash
go test ./pkg/memory \
  -run '^(TestInMemoryStore(CloseReleasesMemories|ClosedOperations|ClosedWritesDoNotPersist|LifetimeAndEmbeddingOwnership)|TestCanceledStoreDoesNotPersistPrecomputedEmbedding)$' \
  -count=1
```

The fixtures do not use native inference, but compiling the whole `memory`
package still requires its other backends' native dependencies, including
Valkey GLIDE.

The other backends' lifecycle matrices and Router reload/drain integration remain
part of the parent issue.

## Upgrading the embedding model

Restart the model runtime after changing embedding weights. For embeddings from
the model runtime, including Vela Embedding, the router binds memory to
the loaded model, tokenizer, inference settings, and vector dimension. Changing
these creates a separate physical collection or index and a separate Redis hot
cache. Restarting with the same representation reuses its existing storage.
Your configured logical names remain unchanged.

Earlier untagged collections are preserved, but are not adopted automatically:
equal vector dimensions do not prove that two models produce compatible
embeddings. The management API has no import or bulk export endpoint for
memories, so the collection for the new model starts empty and repopulates
from new traffic. No old collection is deleted during
startup or model migration. This automatic identity binding covers every
embedding the model runtime serves.

A remote embedding endpoint cannot prove which model produced its vectors, so
memory keeps the configured collection or index, and the router logs a startup
warning. After you change `endpoint.model`, or the provider changes the model
behind the endpoint, point memory at a new collection or index. The new one
starts empty, and the old one is left as it was.

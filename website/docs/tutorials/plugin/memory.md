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

## Corrections

Router Memory stores each turn as it was said, so a fact and its later
correction can both be retrieved for the same request. The default memory
filter (`reflection.algorithm: heuristic`) then drops the older turn when a
sentence in the newer one says the user changed something, as in "I moved",
"we've switched", "I'm no longer" or "I don't ... anymore", and the part about
the change shares a word pair with the older message, such as "I live" or
"work as". A question, a change made by someone else, or one that is negated,
hypothetical or only planned doesn't count. A session-window memory loses only
the corrected turn.

Stored records don't change. An old turn is hidden only when its correction is
injected in the same request, and only if it states a single fact. A turn with
several sentences or clauses, such as "I live in Boston and work as a nurse",
is kept because it may hold other facts. A session window carries the time of
its newest turn only, so its earlier turns can correct turns before them in
that window but not other memories. Corrections phrased another way or written
in another language aren't recognized yet. Setting
`reflection.algorithm: noop` turns this off along with the rest of the filter.

## Upgrading the embedding model

Restart the model runtime after changing embedding weights. For local `mmbert`
models, including Vela Embedding, the router binds memory to
the loaded model, tokenizer, inference settings, and vector dimension. Changing
these creates a separate physical collection or index and a separate Redis hot
cache. Restarting with the same representation reuses its existing storage.
Your configured logical names remain unchanged.

Earlier untagged collections are preserved, but are not adopted automatically:
equal vector dimensions do not prove that two models produce compatible
embeddings. The management API has no import or bulk export endpoint for
memories, so the collection for the new model starts empty and repopulates
from new traffic. No old collection is deleted during
startup or model migration. This automatic identity binding currently covers
local `mmbert`; other embedding providers keep their existing behavior, except
Candle `bert` as described below.

Candle `bert` models have no content descriptor, so the router keys their memory
by an encoder version that changes whenever Candle BERT vectors change, as they
did when padding tokens stopped counting toward the average. After upgrading
across such a change, BERT memory opens a new collection or index and a new
Redis hot cache. Entries stored before the upgrade stay in the old collection
and are no longer recalled, so memory fills again from new conversations. BERT
served by another runtime keeps its existing storage.

A remote embedding endpoint cannot prove which model produced its vectors, so
memory keeps the configured collection or index, and the router logs a startup
warning. After you change `endpoint.model`, or the provider changes the model
behind the endpoint, point memory at a new collection or index. The new one
starts empty, and the old one is left as it was.

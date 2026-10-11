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

## Corrections

Router Memory stores each turn as it was said, so a fact and its later
correction can both be retrieved for the same request. The default memory
filter (`reflection.algorithm: heuristic`) then drops the older turn when a
sentence in the newer one says the user changed something, as in "I moved",
"we've switched", "I'm no longer" or "I don't ... anymore", and the part about
the change shares a word pair with the older message, such as "work as" or "I
live" followed by a place. A shared verb must keep its complement and a shared
noun its purpose, so "I work as a paramedic now" doesn't correct "I work out
every morning" and "my budget for groceries" doesn't correct "my budget for the
Japan trip". The destination in "I moved to Denver" doesn't match other facts
about Denver either. A question, a change made by someone else, or one that is
negated, hypothetical or only planned doesn't count. A session-window memory
loses only the corrected turn. When a message quotes a `---` line followed by a
`Q:` line, the stored copy escapes that `Q:`, so the quote can't start a new
turn. Windows stored before that escaping can still lose a corrected turn, but
their turns never correct other memories.

Stored records don't change. An old turn is hidden only when its correction is
injected in the same request, and only if it states a single fact. A turn with
several sentences or clauses, such as "I live in Boston and work as a nurse" or
"I live in Boston, my dog is Biscuit", is kept because it may hold other facts.
When a turn is hidden, a reply sentence about something else stays in the
prompt, such as "Your dog Biscuit is a Boston terrier" after a move away from
Boston. A session window carries the time of
its newest turn only, so its earlier turns can correct turns before them in
that window but not other memories. Corrections phrased another way or written
in another language aren't recognized yet. Setting
`reflection.algorithm: noop` turns this off along with the rest of the filter.

## Observability

Router Memory exposes bounded Prometheus metrics on the Router scrape endpoint.
Retrieval volume is tracked by `llm_memory_retrieval_total` with `backend` and
`status` labels (`hit`, `miss`, `error`). Latency and result counts use separate
histograms without per-user labels. If you upgrade from a release that labeled
memory metrics with `user_id`, follow the [Router Memory Prometheus label release
note](../../release-notes/router-memory-prometheus-labels).

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

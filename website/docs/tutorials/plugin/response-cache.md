# Response Cache

## Overview

`response_cache` is the route-local plugin for reusing exact or semantically
compatible prior responses.

## Key Advantages

- Reuses prior responses only on routes that benefit from cache hits.
- Keeps route-local thresholds separate from global store setup.
- Supports different cache policies for different routes.

## What Problem Does It Solve?

Some routes benefit strongly from reuse, while others need fresh generation
every time. `response_cache` keeps the reuse policy local to the route.

## When to Use

- one route should prefer cached responses when queries are very similar
- different routes need different similarity thresholds or TTLs
- the route should use a cache backend configured in `global.stores.response_cache`

## Configuration

Add the plugin under `recipes[].decisions[].plugins`:

```yaml
plugins:
  - type: response_cache
    configuration:
      enabled: true
      mode: exact
      scope: user
      ttl_seconds: 86400
      request_controls:
        enabled: true
        header: x-vsr-cache-control
        allowed: [no-cache, no-store, bypass, max-age, ttl]
        max_ttl_seconds: 86400
      personalized:
        mode: disabled
```

`mode` accepts:

- `semantic` (default): vector lookup only.
- `exact`: normalized exact request lookup only.
- `exact_then_semantic`: exact lookup first, then vector lookup on a miss.

The shipped `config/config.yaml`, the multi-objective example and the
`memory.yaml` fragment use `exact`: identical requests still hit, and no
answer is served to a different question only because its embedding is
similar. The `high-recall.yaml` fragment is the semantic-cache example.

:::caution
Use `semantic` (the default) or `exact_then_semantic` only after validating
the route's language and contradiction behavior. The built-in lexical guard
recognizes English negation cues only, so a German `nicht` or Chinese `不` can
otherwise receive a cached answer to the opposite question.
:::

Replay and freshness, in every mode:

- **Backends.** The exact tier works with the in-memory, Redis, Valkey,
  Milvus, Qdrant and hybrid cache backends.
- **Wire formats.** Anthropic client requests are replayed in the Anthropic
  response or SSE wire format. Streaming and non-streaming requests use
  separate cache identities, so a cached response is never translated across
  wire modes.
- **Streams.** Replay preserves content, reasoning, refusal, tool calls,
  terminal usage, finish reasons and choice indexes for complete single- or
  multi-choice streams. Incomplete streams are never cached.
- **Request controls.** When enabled, the configured header accepts the
  authorized directives: `max-age` bounds read freshness and `ttl` bounds
  write lifetime. Caller TTL values are clamped to `max_ttl_seconds`.
- **Promotion from L2 to L1** keeps an exact entry's original age and backend
  expiry. An entry whose age is unknown can serve an unrestricted lookup, but
  can't satisfy a `max-age` bound in either tier.

### Semantic reuse

The semantic tier finds the cached requests a request may reuse with the
[match block](./overview#match-block) in `semantic.match`. Without one it runs
the response cache's default, which recalls and confirms:

```yaml
semantic:
  match:
    recall: {method: embedding, top_k: 3, min_similarity: 0.8}
    judge:  {preset: same_request, min_score: 0.8}
```

- `recall` asks the cache store for the cached requests most similar to the
  request, at most `top_k`, each at least `min_similarity` similar (a cosine).
- `judge` asks the decision model the `same_request` question about the
  request and each recalled request, one batch entry each in one call, and
  serves the best one that scores at least `min_score`. It turns away a cached
  request that is close in embedding space but asks something else, such as
  the opposite question: a recalled request the judge rejects is a miss. When
  the judge fails, `on_error: skip` (the default) serves the recall result and
  `on_error: fail` reuses nothing.
- A block that writes `recall` alone serves the most similar cached request
  and asks no model.
- A request that matched no decision runs the block of the `response_cache`
  plugin in `global.plugins`.
- Semantic matching also needs a compatible fingerprint over the system
  prompt and history, tools, response format, generation parameters, client
  protocol and route policy. It never crosses the recipe, tenant,
  request-model or selected-model partitions.

### With a response guard

On a decision that also has a [`response_guard`](./response-guard.md), the
cache keeps only the answers the guard released. It never stores a blocked
answer, or one the guard flagged only in `x-vsr-warnings`, because a cache hit
carries no guard headers; an answer with the guard's note is stored with its
note. A hit is answered without calling the model, so the guard does not run on
it:

- An exact hit is the same request, so the guard's verdict on it still holds.
- A semantic hit serves the answer the guard released for a similar earlier
  request. The guard's conditions on the request itself, such as a `keyword`
  or `fact_check` condition, are not evaluated for the new request, so a rule
  they would have triggered does not apply to it: an answer released for a
  request that needed no fact check is served, unchecked, to a similar request
  that does. Where that matters, use `mode: exact` for that decision's cache.

- **Changing the decision's guard** starts its cache fresh, because the
  guard's configuration is part of the cache key.
- **Changing a signal the guard reads**, such as a jailbreak rule's
  `predicate`, doesn't change the key. A store that keeps its entries across a
  configuration reload, such as Redis, goes on serving the answers released
  under the old value until they expire.

After such a change, set a new `revision.policy_revision` in the
`response_cache` configuration of each decision whose guard reads that signal.
Every `revision` value is part of the cache key:

```yaml
plugins:
  - type: response_cache
    configuration:
      enabled: true
      mode: exact
      revision:
        policy_revision: guard-2
```

## Migration

The plugin type is `response_cache` and the store is
`global.stores.response_cache`. A configuration that still writes any of these
fails to load, with a line that names the field:

- the `semantic-cache`, `semantic_cache` or `response-cache` plugin types;
- `global.stores.semantic_cache`;
- the flat `similarity_threshold`, or `semantic.similarity_threshold`;
- `allow_request_controls` and `control_header`.

`vllm-sr config migrate` writes the threshold as
`semantic.match.recall.min_similarity` and the request-control fields under
`request_controls`.

For embeddings from the model runtime, including Vela Embedding, changing the
model, tokenizer, representation size or inference settings starts a separate
cache space:

- **Kept:** your tenant namespace and explicit cache revision. Earlier entries
  stay stored until their normal expiry or an explicit cleanup.
- **After an upgrade** the first requests are cache misses. Restarting with
  the same representation reuses its compatible cache.
- **No remote endpoint.** The Router rejects a
  [remote embedding endpoint](../../model-runtime/guides/embeddings.md#use-an-external-embedding-service)
  for the semantic cache, because the cache needs local tokenizer windows.

## Operations

The management API exposes redacted health, capabilities, statistics,
candidate configuration testing, scoped invalidation and epoch-based flush
under `/api/v1/storage/response-cache/*`:

- **Discovery.** The plugin descriptor at `/api/v1/plugins/response_cache`
  links to these operations.
- **Audit.** A hash-chained audit, shared across management operations, is at
  `/api/v1/observability/audit` (`audit.read`).
- **Invalidation** defaults to a dry run.
- **Flush** requires the explicit confirmation phrase `flush response cache`,
  and never calls a backend-wide `FLUSHALL`.

All six cache backends apply an always-on English lexical check before serving
a semantic hit:

- **Rejected:** near-identical questions with explicit negation or a known
  antonym swap, even when their vector similarity is high. Remote entries
  without their original question are misses too.
- **Bounded.** A rejected candidate doesn't stop a later eligible fetched
  candidate from being used; remote search stays bounded by its candidate
  limit.
- **Not covered:** word-order-only, cue-less and non-English changes. The
  check doesn't establish that such questions mean the same.

Every served semantic hit records whether that check could judge the pair, as
`cache.negation_guard` on the `response_cache` plugin span and as
`negation_guard` on the `cache_hit` log event:

| Value | Meaning |
| --- | --- |
| `checked` | Both questions use the same words, in any order, or differ only in negation cues such as `not` and `never`. |
| `not_applicable` | Some other word changed, which the cue list can't judge, so the hit relied on vector similarity alone. Reworded English questions and most non-English hits report it; the cue list doesn't recognize German `nicht` or Chinese `不`, for example. |

The NLI tier that earlier releases offered on the in-memory backend is retired;
`vllm-sr config migrate` keeps the lexical check. See
[Stores and Tools](../global/stores-and-tools.md#negation-guard).

:::caution
Cached responses can contain user or tenant data. Choose an appropriate scope,
TTL, backend authentication, encryption and invalidation process.
:::

- **Calibrate** semantic thresholds for the configured embedding model.
- **Long queries.** A query longer than the embedding deployment's input limit
  isn't cached, because a truncated embedding would match every query sharing
  that prefix.
- **Personalized routes.** Routes with personalized RAG or memory shouldn't
  reuse pre-enrichment responses without an explicit policy.

Complete examples:
[`high-recall.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/response-cache/high-recall.yaml)
and
[`memory.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/response-cache/memory.yaml).

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

The shipped `config/config.yaml`, multi-objective example, and `memory.yaml`
fragment use `exact`. It preserves reuse for identical requests without
returning an answer to a different question solely because its embedding is
similar. Choose `semantic` or `exact_then_semantic` explicitly only after
validating the route's language and contradiction behavior. The built-in
lexical guard recognizes English negation cues; a German `nicht` or Chinese
`不` can otherwise receive a cached answer to the opposite question. The
`high-recall.yaml` fragment remains an explicit semantic-cache example.

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
- A request that matched no decision recalls with the default `top_k` and the
  store's `similarity_threshold`.

The exact tier is available with the in-memory, Redis, Valkey, Milvus, Qdrant,
and hybrid cache backends. Anthropic client requests are replayed in the
Anthropic response or SSE wire format.

Streaming and non-streaming requests use separate cache identities so replay
never translates a cached response across wire modes. Semantic matching uses a
compatibility fingerprint over system/history, tools, response format,
generation parameters, client protocol, and route policy, plus hard recipe,
tenant, request-model, and selected-model partitioning.

Streaming replay preserves content, reasoning, refusal, tool calls, terminal
usage, finish reasons, and choice indexes for complete single- or multi-choice
streams. Incomplete streams are never cached.

When request controls are enabled, the configured header accepts the authorized
directives. `max-age` bounds read freshness and `ttl` bounds write lifetime;
caller TTL values are clamped to `max_ttl_seconds`.

Exact-cache promotion from L2 to L1 retains the original entry age and backend
expiry. An entry whose age is unknown can serve an unrestricted lookup, but it
cannot satisfy a `max-age` freshness bound in either tier.

## Migration

The plugin type is `response_cache` and the store is
`global.stores.response_cache`. A configuration that still writes the
`semantic-cache`, `semantic_cache` or `response-cache` plugin types,
`global.stores.semantic_cache`, the flat `similarity_threshold` or
`semantic.similarity_threshold`, or `allow_request_controls` and
`control_header` fails to load with a line that names the field;
`vllm-sr config migrate` writes the threshold as
`semantic.match.recall.min_similarity` and the request-control fields under
`request_controls`.

For embeddings from the model runtime, including Vela Embedding, changing the model,
tokenizer, representation size, or inference settings starts a separate cache
space. The router retains your tenant namespace and explicit cache revision;
historical entries remain stored until their normal expiry or explicit cleanup.
The first requests after a model upgrade are cache misses. Restarting with the
same representation reuses its compatible cache. The router rejects a
[remote embedding endpoint](../../model-runtime/guides/embeddings.md#use-an-external-embedding-service)
for the semantic cache, because the cache needs local tokenizer windows.

## Operations

The management API exposes redacted health, capabilities, statistics, candidate
configuration testing, scoped invalidation, and epoch-based flush under
`/api/v1/storage/response-cache/*`. The plugin descriptor at
`/api/v1/plugins/response_cache` links to these operations. Hash-chained audit
is shared across management operations at `/api/v1/observability/audit`
(`audit.read`). Invalidation defaults
to dry-run. Flush requires the explicit confirmation phrase
`flush response cache` and never calls backend-wide `FLUSHALL`.

All six cache backends apply an always-on English lexical check before serving
a semantic hit. Near-identical questions with explicit negation or a known
antonym swap are rejected even when their vector similarity is high. Remote
entries without their original question are also misses. A rejected candidate
does not prevent a later eligible fetched candidate from being used; remote
search remains bounded by its candidate limit. This check does not establish
semantic equivalence for word-order-only, cue-less, or non-English changes.

Every served semantic hit records whether that check could judge the pair, as
`cache.negation_guard` on the `response_cache` plugin span and as
`negation_guard` on the `cache_hit` log event. `checked` means both questions
use the same words, in any order, or differ only in negation cues such as `not`
and `never`. `not_applicable` means some other word changed, which the cue list
cannot judge, so the hit relied on vector similarity alone. Reworded English
questions and most non-English hits report `not_applicable`; for example, the
cue list does not recognize German `nicht` or Chinese `不`.

The NLI tier that earlier releases offered on the in-memory backend is retired;
`vllm-sr config migrate` keeps the lexical check. See
[Stores and Tools](../global/stores-and-tools.md#negation-guard).

Cached responses can contain user or tenant data. Choose an appropriate scope,
TTL, backend authentication, encryption, and invalidation process. Semantic
thresholds must be calibrated for the configured embedding model. A query longer
than the embedding deployment's input limit is not cached, because a truncated
embedding would match every query
sharing that prefix. Routes
with personalized RAG or memory should not reuse pre-enrichment responses
without an explicit policy. See complete examples:
[`high-recall.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/response-cache/high-recall.yaml)
and
[`memory.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/response-cache/memory.yaml).

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

Add the plugin under `routing.decisions[].plugins`:

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

## Migration

`semantic-cache`, `semantic_cache`, and `response-cache` are accepted as
deprecated aliases and normalize to `response_cache`. Likewise,
`global.stores.semantic_cache` is read as a deprecated alias for
`global.stores.response_cache`. Do not configure both spellings in the same
document. Export, Dashboard saves, and DSL decompilation always emit the
canonical names.

For local `mmbert` embeddings, including Vela Embedding, changing the model,
tokenizer, representation size, or inference settings starts a separate cache
space. The router retains your tenant namespace and explicit cache revision;
historical entries remain stored until their normal expiry or explicit cleanup.
The first requests after a model upgrade are cache misses. Restarting with the
same representation reuses its compatible cache. The router rejects a
[remote embedding endpoint](../../installation/runtime/embeddings.md#remote-embeddings)
for the semantic cache, because the cache needs local tokenizer windows.

Candle `bert` embeddings are keyed by an encoder version instead, which changes
whenever Candle BERT vectors change, as they did when padding tokens stopped
counting toward the average. Upgrading across such a change starts a new BERT
cache space. Entries written before the upgrade are not reused and remain until
they expire, and the cache fills again from new traffic. BERT served by another
runtime keeps its existing cache.

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

The in-memory backend additionally supports the optional NLI verifier
(`global.stores.response_cache.polarity_guard`; see
[Stores and Tools](../global/stores-and-tools.md#negation-guard)). With this
optional tier enabled, an NLI-rejected candidate is logged as
`cache_negation_reject` with `tier: nli`, is reported as a miss, and its
similarity still appears on `x-vsr-cache-similarity` so near-threshold
rejections stay diagnosable.

Cached responses can contain user or tenant data. Choose an appropriate scope,
TTL, backend authentication, encryption, and invalidation process. Semantic
thresholds must be calibrated for the configured embedding model. A query longer
than the embedding model's context window (512 tokens for the default `bert`
model) is not cached, because a truncated embedding would match every query
sharing that prefix. Routes
with personalized RAG or memory should not reuse pre-enrichment responses
without an explicit policy. See complete examples:
[`high-recall.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/response-cache/high-recall.yaml)
and
[`memory.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/response-cache/memory.yaml).

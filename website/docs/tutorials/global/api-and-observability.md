# API and Observability

## Overview

This page covers the shared runtime blocks that expose interfaces and telemetry.

These settings are router-wide and belong in `global:`, not in route-local plugin fragments.

## Key Advantages

- Keeps observability and interface controls consistent across routes.
- Avoids duplicating metrics or API settings inside route-local config.
- Makes replay and response APIs explicit shared services.
- Keeps operational controls in one router-wide layer.

## What Problem Does It Solve?

If API and telemetry behavior is configured per route, the operational surface becomes fragmented and hard to reason about.

This part of `global:` solves that by collecting shared interfaces and monitoring settings in one place.

## When to Use

Use these blocks when:

- the router should expose shared APIs
- the response API should be enabled for the whole router
- metrics and tracing should be configured once
- replay capture should be retained as a shared operational service

## Configuration

### Routing configuration

Without `global.stores.management`, the Router compiles one read-only manifest
before readiness. Validate that file offline with `vllm-sr validate`; the
listener has no config mutation, Recipe authoring, knowledge-base authoring,
backup, rollback, or runtime-sync API.

With a Management store, clients create Models, Recipes, and Entrypoints through
the Router's versioned `/management/v1` resources. The initial file seeds an
empty store atomically, then PostgreSQL remains the only mutable routing
authority.

### Billing currency

```yaml
global:
  billing:
    currency: USD
```

Every manifest requires this when a Provider Model defines
`providers.models[].pricing`; otherwise the block is optional. It is the one
ISO-4217 denomination used across Model fallback, multi-model execution, usage,
and cost quotas. A Management-store bootstrap seeds its Namespace with this
currency, and every immutable routing publication pins that value.

### API

```yaml
global:
  services:
    api:
      batch_classification:
        max_batch_size: 100
        concurrency_threshold: 5
        max_concurrency: 8
```

### Response API

```yaml
global:
  services:
    response_api:
      enabled: true
      store_backend: redis        # configure explicitly for durable, shared history
      redis:
        address: "redis:6379"
```

The `store_backend` field controls where response and conversation history is persisted. Available backends:

| Backend | Durability | Use case |
|---------|-----------|----------|
| `memory` | Lost on Router restart | Local, single-process use with no external dependency |
| `redis` | Survives Router restart, shared across replicas | Durable or multi-replica deployments |

### Observability

```yaml
global:
  services:
    observability:
      metrics:
        enabled: true
      tracing:
        enabled: true
        provider: opentelemetry
        exporter:
          type: otlp
          endpoint: jaeger:4317
          insecure: true
        sampling:
          type: probabilistic
          rate: 0.1
```

`probabilistic` is the recommended tracing sampling type. Existing
configurations that use `traceidratio` or `trace_id_ratio` continue to work as
compatibility aliases.

Common Prometheus metric families:

| Family | Example metrics |
|--------|-----------------|
| Requests | `llm_model_requests_total`, `llm_request_errors_total` |
| Errors | `llm_request_errors_total{reason="timeout"}` |
| Latency | `llm_model_completion_latency_seconds`, `llm_model_ttft_seconds`, `llm_model_tpot_seconds`, `llm_model_routing_latency_seconds` |
| Tokens and cost | `llm_model_tokens_total`, `llm_model_prompt_tokens_total`, `llm_model_completion_tokens_total`, `llm_model_cost_total` |
| Routing | `llm_model_routing_modifications_total`, `llm_routing_reason_codes_total` |
| Selection | `llm_model_selection_total`, `llm_model_selection_duration_seconds`, `llm_model_inflight_requests` |
| Cache | `llm_cache_plugin_hits_total`, `llm_cache_plugin_misses_total`, `llm_cache_warmth_estimate` |
| RAG | `rag_retrieval_attempts_total`, `rag_retrieval_latency_seconds`, `rag_cache_hits_total`, `rag_cache_misses_total` |
| Session | `llm_session_model_transitions_total`, `llm_session_turn_prompt_tokens`, `llm_session_turn_completion_tokens`, `llm_session_turn_cost` |
| Translation and request-parameter policy | `llm_translation_lossy_total`, `sr_request_params_blocked_total`, `sr_request_params_unknown_field_stripped_total` |

### Usage and request history

When the Management, runtime, and access services are configured, Router exposes
durable, tenant-scoped accounting from its Management API:

```text
GET /management/v1/usage
GET /management/v1/usage/series
GET /management/v1/usage/breakdowns
GET /management/v1/users/{userId}/usage
GET /management/v1/teams/{teamId}/usage
GET /management/v1/api-keys/{keyId}/usage
GET /management/v1/request-logs
GET /management/v1/namespaces/{namespaceId}/request-logs/{admissionId}
```

The three resource-detail usage routes return the same summary shape as the main
usage route, but Router fixes the subject filter to the resource in the URL. A denied
or unknown resource returns the same nondisclosing response. Clients may still choose
the bounded time range, time zone, minute/hour/day grain, and other dimensions that
their Management session is allowed to inspect.

Usage totals come from immutable request accounting and verified rollups. Live quota
remaining comes from the global counter engine instead; clients should not subtract
historical usage from a configured limit. Request history uses opaque cursor
pagination and an `admission_id` for exact detail, so large datasets do not require
offset scans. Internal Model, provider, and dispatch dimensions are omitted unless
the caller has their explicit read permission.

Every Usage summary and breakdown includes exact cost summaries when Model pricing is
configured. An API-key detail view should fetch both its `/usage` and `/quota`
resources: Usage reports actual historical spend, while a live `cost` meter reports
the enforced limit, remaining amount, currency, reset time, and completeness. A
sliding rule may use any supported bounded duration—for example, `window: PT8H`—and the
Router settles it from authoritative response tokens and the pinned Model price.

### Router Replay

```yaml
global:
  services:
    router_replay:
      enabled: true
      store_backend: postgres     # explicit durable, SQL-queryable audit storage
      async_writes: true
      postgres:
        host: postgres
        port: 5432
        database: vsr
        user: router
        password: ${ROUTER_REPLAY_POSTGRES_PASSWORD}
```

Router replay is disabled by default. Set `global.services.router_replay.enabled`
to enable it router-wide; when it is on, a decision captures replay unless that
decision adds a route-local `router_replay` plugin with `enabled: false`. A
decision may also opt in explicitly. If no durable backend is configured, the
default in-memory store is process-local and is lost on restart.

The `store_backend` field controls where routing-decision replay records are persisted. Available backends:

| Backend | Durability | Use case |
|---------|-----------|----------|
| `postgres` | Full SQL queryability, long-term audit retention | Production audit storage |
| `redis` | Survives router restart, shared across replicas | Lightweight deployments already running Redis |
| `milvus` | Vector-searchable replay records | Semantic replay search |
| `qdrant` | Vector-searchable replay records | Semantic replay search in a Qdrant deployment |
| `memory` | Lost on router restart | Local development only |

## Data and Security

- Response API and Router Replay may persist prompts, responses, routing
  outcomes, and tool traces. Set TTLs, capture limits, tenant/user scope, and
  read permissions before enabling them.
- Bind the management API to a private interface or enable its role-based token
  authentication before remote exposure.
- Traces and metric labels should carry bounded identifiers, not raw request
  content or secrets.
- See the complete service configuration in
  [`config/config.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml).

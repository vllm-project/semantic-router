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

### Router config validation

The management API validates and normalizes a candidate config without writing
it:

```http
POST /config/router/validate
Content-Type: application/json

{"yaml":"version: v0.3\n..."}
```

Successful responses include `valid: true` and the normalized canonical YAML.
Validation uses the same parser and semantic checks as `PATCH /config/router`
and `PUT /config/router`, but preserves `${ENV_VAR}` references verbatim rather
than reading process secrets. The endpoint requires `config.read`; plaintext
secret viewing is not implied.

### API

```yaml
global:
  services:
    api:
      batch_classification:
        max_batch_size: 100
```

### Response API

```yaml
global:
  services:
    response_api:
      enabled: true
      store_backend: redis        # default; use "memory" only for local development
      redis:
        address: "redis:6379"
```

The `store_backend` field controls where response and conversation history is persisted. Available backends:

| Backend | Durability | Use case |
|---------|-----------|----------|
| `redis` | Survives router restart, shared across replicas | Production (default) |
| `memory` | Lost on router restart | Local development only |

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

### Profiling

The Router can expose Go `pprof` endpoints on a dedicated listener for CPU,
heap, goroutine, and execution-trace investigations.

```yaml
global:
  services:
    observability:
      profiling:
        enabled: false        # default; opt in only while investigating
        port: 6060            # default
        bind: 127.0.0.1       # default; loopback only
```

Profiling is disabled by default. When enabled it binds `127.0.0.1:6060`, so
profiles stay reachable from the Router container or host and are never
published on a routable interface without an explicit `bind` change.

```bash
go tool pprof http://127.0.0.1:6060/debug/pprof/heap
```

Notes:

- `bind` must be an IP address or `localhost`. An empty or hostname value is
  rejected and the profiling listener is skipped.
- An explicit `port: 0` requests an ephemeral port; the effective address is
  reported in the `profiling_server_starting` startup log line.
- The port must not collide with the ExtProc, metrics, or management API port.
  A conflicting or unbindable listener is logged and skipped; it does not abort
  Router startup.
- This switch is read once at startup. Changing it requires a Router restart;
  config hot reload does not take over the profiling listener.

### Skip Processing Header

`global.router.skip_processing.enabled` is the deployment-level gate that
opts the router into honoring the `x-vsr-skip-processing` request header.
When the gate is on and an upstream filter sets that header to `true`, the
router becomes a no-op for that single request — every Envoy ext_proc
callback returns CONTINUE without classifying, routing, mutating, caching,
or inspecting the request or upstream response. When the gate is off (the
default) the header is ignored entirely.

```yaml
global:
  router:
    skip_processing:
      enabled: false        # default; flip to true to honor the header
```

The Helm chart exposes the same gate as a top-level value
(`router.skipProcessing.enabled`) so it can be enabled at install time
without editing the embedded canonical config:

```bash
helm install vsr ./deploy/helm/semantic-router \
  --set router.skipProcessing.enabled=true
```

Enable this gate only when an authenticated upstream filter (Envoy AI
Gateway, ext_authz, route-level filters, etc.) is responsible for setting
or stripping the header on trust grounds. Background on the AI Gateway
interop pattern that motivates this gate lives in
[issue #1808](https://github.com/vllm-project/semantic-router/issues/1808).

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
- `pprof` endpoints expose command-line arguments, goroutine stacks, and heap
  contents. Keep profiling disabled outside an investigation, and keep its
  `bind` on loopback unless a reachable listener is deliberately fronted by
  authenticated access controls.
- See the complete service configuration in
  [`config/config.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml).

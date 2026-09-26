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
POST /api/v1/config/validate
Content-Type: application/json

{"yaml":"version: v0.3\n...","compare_to_active":true}
```

Successful and invalid candidate evaluations both return HTTP 200 with
`contract_version: v1`. Existing fields `valid` and `normalized_yaml` keep their
meaning. The response also includes field-addressable `errors` and `warnings`
(`code`, `severity`, `resource`, `recipe`, `stage`, `field`, `message`). Set
`compare_to_active: true` to include a bounded, schema-aware, redacted
active-versus-candidate `diff`. The active side is the verified in-memory
runtime snapshot when one exists, otherwise the generated runtime document, not
desired source after an override or failed activation. Secrets are always
`[REDACTED]`; `${ENV_VAR}` references and `*_env` names stay visible.
Validation uses the same parser and semantic checks as `PATCH /api/v1/config`
and `PUT /api/v1/config`. The path does not write desired config, the active
snapshot, runtime state, persistence, or version history. The endpoint requires
`config.read`; plaintext secret viewing is not implied.

### API

```yaml
global:
  services:
    api:
      routing_preview:
        request_timeout_seconds: 120
        max_concurrency: 16
      batch_classification:
        max_batch_size: 100
```

`max_batch_size` bounds `texts` per `/api/v1/diagnostics/classify/batch` request. Larger
batches return `400 INVALID_INPUT`.

`routing_preview` applies to `POST /api/v1/routing/preview`. Its inference
deadline starts after the request body is decoded and defaults to 120 seconds.
Set `request_timeout_seconds` between 1 and 3600 using measured inference times
for the intended input lengths and deployment hardware. This setting can be
updated through config hot reload;
other HTTP routes keep their existing timeouts.

A deadline returns `504 REQUEST_TIMEOUT` and cancels queued or cancellable
inference. Native inference already running may finish later. Its model resources
and admission slot remain held until it finishes, including during shutdown.
`max_concurrency` is a positive worker limit, defaults to 16, and has no wait
queue: when all slots are occupied, new previews return `429 OVERLOADED`.
Changing this limit requires a deployment restart; hot reload rejects the change.

The response writer has five additional seconds to send the result or timeout
response. Dashboard Topology uses the configured Preview budget plus this
allowance and propagates client cancellation. Recipe probes retain their own
`evaluation.request_timeout_seconds` caller budget in `probes.yaml`; configure
it for the intended run, and allow at least five extra seconds in external HTTP
clients or proxies when they need to receive the Router's timeout response.

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
| Terminal requests | `llm_request_outcomes_total`, `llm_request_duration_seconds` (by bounded `traffic_kind` and `outcome`) |
| Backend dispatch and error events | `llm_model_requests_total`, `llm_request_errors_total` (not a completed-client-request denominator) |
| Latency | `llm_model_completion_latency_seconds`, `llm_model_first_response_observation_seconds`, `llm_model_response_duration_per_output_token_seconds`, `llm_model_routing_latency_seconds` |
| Tokens and cost | `llm_model_tokens_total`, `llm_model_prompt_tokens_total`, `llm_model_completion_tokens_total`, `llm_model_cost_total` |
| Routing | `llm_model_routing_modifications_total`, `llm_routing_reason_codes_total` |
| Selection | `llm_model_selection_total`, `llm_model_selection_duration_seconds`, `llm_model_inflight_requests` |
| Looper | `llm_looper_attempts_total`, `llm_looper_attempt_duration_seconds`, `llm_looper_attempt_first_byte_seconds`, `llm_looper_attempt_tokens_total`, `llm_looper_attempt_cost_total`, `llm_looper_execution_duration_seconds` |
| Cache | `llm_cache_plugin_hits_total`, `llm_cache_plugin_misses_total`, `llm_cache_warmth_estimate` |
| RAG | `rag_retrieval_attempts_total`, `rag_retrieval_latency_seconds`, `rag_cache_hits_total`, `rag_cache_misses_total` |
| Session | `llm_session_model_transitions_total`, `llm_session_turn_prompt_tokens`, `llm_session_turn_completion_tokens`, `llm_session_turn_cost` |
| Translation and request-parameter policy | `llm_translation_lossy_total`, `sr_request_params_blocked_total` |
| Signals | `llm_signal_extraction_total`, `llm_signal_match_total`, `llm_signal_extraction_latency_seconds` |
| Complexity verdicts | `llm_complexity_verdict_total` (by `rule`, `verdict`, `source`), `llm_complexity_evaluation_failures_total` |
| Remote classifier backends | `llm_remote_connector_requests_total` (by `operation`, `outcome`), `llm_remote_connector_request_duration_seconds`, `llm_remote_connector_retries_total` |
| Recipe routing | `llm_entrypoint_requests_total`, `llm_recipe_selections_total`, `llm_routing_stage_duration_seconds` |
| Projections | `llm_projection_score` (by configured recipe and projection name) |
| Trace export | `llm_trace_export_spans_total` (by exporter batch result) |

`llm_request_outcomes_total{traffic_kind="inference"}` counts public inference
requests once at their terminal boundary. Authenticated internal looper requests
use `inference_internal`; catalog, response-object and health operations have
separate traffic kinds. Outcomes distinguish success, client/server errors,
cancellation, timeout, incomplete responses and other failures. A backend
selection or an error event is not another completed public request.

The model latency names describe their actual observations:
`llm_model_first_response_observation_seconds` measures the first stream chunk
or non-streaming response headers, while
`llm_model_response_duration_per_output_token_seconds` divides complete response
time by reported output tokens. These are not first-token latency or decode-only
inter-token latency. The optional windowed metrics summarize at most 10,000
observed completed responses per model; they do not estimate utilization, queue
depth or provider error rates.

The `semantic_router.request` span covers the complete ExtProc request, including
streamed responses and cancellation. Its bounded `traffic.kind` and route
separate inference from catalog or health polling without storing URL queries or
resource IDs. Signal evaluation, decision evaluation, algorithm selection,
plugins, and the actual upstream request are child stages. Resolved
`routing.entrypoint`, `routing.recipe`, `decision.name`, and `routing.algorithm`
identify the routing boundary where available. A `routing.backend.resolved`
event records selection evidence; the upstream span measures provider duration.
If a local response guard blocks an upstream HTTP 200, the upstream span retains
200 while the root records the final client response status.

The upstream span also carries OpenTelemetry GenAI attributes, so GenAI-aware
trace backends can show each provider call: `gen_ai.operation.name` (`chat`),
`gen_ai.provider.name`, `gen_ai.request.model` (the provider model ID sent
upstream), and the provider-reported `gen_ai.usage.input_tokens` and
`gen_ai.usage.output_tokens`. Buffered responses also set
`gen_ai.response.model`; streamed responses do not yet. Usage the provider did
not report is left unset rather than estimated, and `model.name` keeps the
Router's logical model name. The GenAI conventions are still in Development
status upstream, so their attribute names can change.

Signal evidence events contain finite reported values and confidence separately,
including real zero values; missing evidence remains absent. Projection events
contain evaluated scores and configured names. Aggregate signal evaluation does
not invent a confidence score. Traces exclude raw prompts, signal errors and
retrieval exception text. Earlier traces cannot be enriched retroactively.

The local `vllm-sr serve` stack provisions the **vLLM Semantic Router**
in Grafana. Its main groups cover public inference outcomes, recipe workflow,
backend usage, plugins and response cache, and telemetry health. Additional
accounting and MoM panels distinguish reported model usage from supported looper
attempt evidence. Recipe stage duration is an observed mean; projection counts
show evaluations, with individual scores available in Insights. Model timing
panels use observed means rather than clipped long-call histogram quantiles.
Prometheus scrapes both the Router and Jaeger's internal admin metrics. An absent
series means no measurement, not zero traffic or a healthy collector. Export
success counts completed SDK exporter batches; it does not prove every request
was sampled or retained.

Local Jaeger uses the pinned all-in-one image with non-root Badger storage and a
stack-specific `<jaeger-container-name>-data` named volume mounted at `/tmp`.
Trace retention is seven days. Grafana keeps its database and preferences in
`<grafana-container-name>-data` at `/var/lib/grafana`, retaining the image's
non-root user. Prometheus retains its local TSDB for 15 days. Container replacement
preserves these stores; deleting a telemetry store starts a new history. Keep
telemetry reset operations separate from benchmark, authentication, learning and
configuration stores.

Changing an older in-memory Jaeger instance to Badger does not migrate its
memory. Sampling remains controlled by Router configuration, and a search limit
is not a count of all retained traces. For multi-node trace storage, configure an
external collector/storage deployment instead of sharing this single-node
Badger volume. See the [Jaeger 1.76 storage documentation](https://www.jaegertracing.io/docs/1.76/deployment/#badger---local-storage).

Looper metric labels are restricted to bounded algorithm, stage, status,
reason, token-type, and currency values. Request IDs, trace IDs, ordinals,
decision names, model names, scores, and thresholds are available through
traces or detailed Router Replay rather than Prometheus labels. Detailed
attempt metrics currently cover the Confidence algorithm; absent attempt
evidence must not be interpreted as zero calls or complete cost accounting.

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

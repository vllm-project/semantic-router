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

{"yaml":"version: v0.5\n..."}
```

Successful responses include `valid: true` and the normalized canonical YAML.
Validation uses the same parser and semantic checks as `PATCH /api/v1/config`
and `PUT /api/v1/config`, but preserves `${ENV_VAR}` references verbatim rather
than reading process secrets. The endpoint requires `config.read`; plaintext
secret viewing is not implied.

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

`routing_preview` applies to `POST /api/v1/routing/preview`:

| Setting | Meaning | Default | Change |
| --- | --- | --- | --- |
| `request_timeout_seconds` | The inference deadline, from when the request body is decoded; 1 to 3600. Set it from measured inference times for your input lengths and hardware. | 120 | Hot reload |
| `max_concurrency` | A positive worker limit with no wait queue: when every slot is busy, new previews return `429 OVERLOADED`. | 16 | Restart; hot reload rejects it |

A deadline returns `504 REQUEST_TIMEOUT` and cancels the preview's model calls:
the model runtime skips work it has not started and finishes a pass already
running. The admission slot frees once those calls return. Other HTTP routes
keep their own timeouts.

The response writer has five additional seconds to send the result or timeout
response. Dashboard Topology uses the configured Preview budget plus this
allowance and propagates client cancellation. Recipe probes retain their own
`evaluation.request_timeout_seconds` caller budget in `probes.yaml`; configure
it for the intended run, and allow at least five extra seconds in external HTTP
clients or proxies when they need to receive the Router's timeout response.

### Authz

```yaml
global:
  services:
    authz:
      fail_open: false
      identity:
        user_id_header: x-authz-user-id
        user_groups_header: x-authz-user-groups
      providers:
        - type: header-injection
        - type: static-config
      delegated_credential:
        enabled: false
        header: x-vsr-delegated-credential
```

`identity` names the headers an authenticating gateway sets for the user and
their groups. `providers` is the credential chain for upstream provider keys,
and the first match wins:

- `header-injection` reads the per-user keys an ext_authz service injected.
- `static-config` falls back to the `api_key` of the selected backend.

`fail_open: true` lets a request through with no key at all.

`delegated_credential` names the header in which a deployment authenticator
attaches the credential it issued the requesting user:

- **Only delegated upstreams receive it.** An upstream whose auth is
  `credential: delegated` (a `backend_refs` entry or a deployment) gets it
  under its own auth header, and only on a call made for that request.
- **A request without it** is answered `401` before the call.

See
[Security Hardening](../../installation/security-hardening#delegate-the-users-credential-to-an-upstream).

### Spend budgets

`budgets` caps what model calls may spend, set once for every route: per agent
run, and per user over a window. Without it nothing is capped.

```yaml
global:
  services:
    budgets:
      currency: USD
      unpriced: refuse
      output_reserve_tokens: 4096
      run:
        limit: 5.00
        idle_timeout_seconds: 86400
      users:
        - name: daily
          match:
            group: eng
          limit: 50.00
          window: day
```

- **A run** is one user's session: every request that carries the session id
  Router Replay records (`x-vsr-session-id`, a `session_id_headers` header,
  `learning.protection.identity.headers.session`, or the session a supported
  harness sends), so a harness's subagents share it. The Looper hops, fallback
  attempts and Router model calls, such as the decision model's, made for
  those requests count too. A request without a session is a run of its own.
  A run unused for `idle_timeout_seconds` (default a day) starts again from
  zero at its next request.
- **A user budget** applies to each user `match` selects, as a rate-limit rule
  does: an empty or `*` `user` or `group` matches everyone. The user is the
  `authz.identity.user_id_header` value, which a listener keeps only when it
  trusts identity headers. Requests without one share one budget per rule.
  `window` is `hour`, `day`, `week` (from Monday) or `month`, in UTC. Listener
  `api_keys` name no user, so a budget per key needs the authenticator in front
  to send the key's identity as the user.
- **Before a call is sent**, the Router reserves its estimated cost against
  every budget the request belongs to: the input it counts, at the model's
  highest input rate, and the output the request allows (`max_tokens`), else the
  model's `max_output_tokens`, else `output_reserve_tokens`, at its output rate.
  The call goes out only when that fits every budget. Calls in flight together
  each hold their reservation, so they cannot together spend past a budget.
- **After the reply**, its priced usage (`x-vsr-cost`) replaces the
  reservation. Each Looper call is charged its own usage at its model's price,
  as the request's record prices it, so a run spends what Insights and
  `vllm-sr usage` report for it. A reply without usage that can be priced is
  charged its estimate, and a call no provider answered is charged nothing. A
  call that costs more than its estimate can leave a budget below zero.
- **At the limit** the call is refused with `429` and the code
  `budget_exceeded`. A request whose budget is already spent is refused before
  its signals run. A fallback candidate that does not fit is skipped for the
  next one. A Looper run that fails because a call was refused answers with
  that call's refusal. In the run's attempt trace a refused call has the
  reason `budget_exceeded` or `budget_unpriced`, and it is no call in the
  record's cost.
- **The message** names the budget, its limit, and what it has left for a
  call now: the limit less what the budget spent and what calls in flight
  hold, the figure `x-vsr-budget-remaining` reports. It then gives the spend
  and the held amount on their own, so the limit less what is left is not
  mistaken for the spend, for example
  `out of budget: the run spend budget of 1.750000 has 1.734367 left (0.000600 spent, 0.015033 held by calls in flight), and this call is estimated at 1.920015`.
  With no call in flight it gives the spend alone.
- **Out of budget** means what the budget spent leaves no room for the call,
  so no retry before the window starts again can succeed. The message starts
  `out of budget`, the reply carries `x-should-retry: false`, and a user
  budget's refusal carries `Retry-After` until its window starts again and
  names that time in the message. On the Responses API the error's `type` is
  `usage_limit_reached`, with `resets_at`, the window's start in Unix
  seconds, and `x-codex-promo-message` repeats the message: Codex ends the
  turn on that error and shows the message. On the Messages API the reply
  carries `budget_exceeded` in
  `anthropic-ratelimit-unified-overage-disabled-reason`, as Anthropic marks a
  spend limit refusal: without it, Claude Code on a Claude login or a gateway
  login presents any 429 as
  `Server is temporarily limiting requests (not your usage limit)`.
- **Held by calls in flight** means the spend alone leaves room, but the
  reservations of other calls in flight do not. The message says so and asks
  for a retry once they finish, with `Retry-After: 5` and
  `x-should-retry: true`.
- **A model with no price** in `currency` is refused with `403`
  `budget_unpriced` under `unpriced: refuse`, the default. `unpriced: allow`
  sends the call, spends nothing for it, and counts it in Replay.

Once a request reaches the budgets, every reply to it carries
`x-vsr-budget-remaining` and `x-vsr-budget-scope` (`run` or `user/<name>`) for
the tightest budget of the request: a provider's reply, a refusal, and the
Router's own answers, such as a Looper answer, a fallback chain whose candidates
all failed, or a reply it could not read. A buffered reply reports the budget
after the call's cost, a failed call's reply after its reservation ends, and a
stream after its reservation. A Looper answer also carries `x-vsr-cost`, the
price of all its calls when every one has a price. The Replay record keeps the
same in `cost_accounting.budget`, where a Looper request's record counts every
call of its run and each hop's record its own. A refused request is a `failed`
record with `terminal_reason: budget_exceeded`, under the model the client
asked for. See [Router API errors](../../api/router.md#routing-errors).

What the harnesses do with a refusal, measured with Claude Code 2.1.325,
Codex 0.166.0 and Pi 1.5.2:

| Refusal | Claude Code | Codex | Pi |
| --- | --- | --- | --- |
| Out of budget, run or user window | Ends the turn at once. With an API key it shows `API Error: Request rejected (429) ·` and the message; with a Claude login or a gateway login, the message alone. | Ends the turn at once: `You’ve hit your usage limit.` with the message, then `or try again at` the window's start, or `or try again later.` for a run. | Ends the turn at once and prints the error with the message. |
| Held by calls in flight | Retries, at least 5 seconds apart. | Retries every 5 seconds, five times, then prints `exceeded retry limit, last status: 429 Too Many Requests`. | Retries after 2, 4 and 8 seconds, then prints the error with the message. |

The Router keeps the budgets in its own memory. A configuration reload keeps
what was spent; a restart forgets it. Each Router counts alone, so behind
several replicas a budget can be spent once per replica: send each user's and
session's requests to one Router.

### Response API

```yaml
global:
  services:
    response_api:
      enabled: true
      store_backend: redis
      redis:
        address: "redis:6379"
```

The `store_backend` field controls where response and conversation history is persisted. Available backends:

| Backend | Durability | Use case |
|---------|-----------|----------|
| `redis` | Survives router restart, shared across replicas | Production |
| `memory` | Lost on router restart | Local development (the default when no Redis server is named) |

A configuration that names no store gets `redis` when its `redis` block names a server (`address`, `cluster_addresses` or `config_path`), as configurations written when `redis` was the default do, and `memory` otherwise, so the Responses API works without another service. `vllm-sr serve` runs Redis for its stack and selects it when the configuration names no store.

The `redis` store needs `redis.address` (or `cluster_addresses` in cluster mode). When the store cannot start, the router still starts, logs `response_api_store_init_failed`, and answers every `/v1/responses` request with 503 and code `response_store_unavailable`. Set `enabled: false` to serve no Responses API at all, which answers 404.

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

An agent run is one trace. A request that carries `traceparent` stays in its
harness's trace; one that carries none and names a session is parented under a
`semantic_router.run` root and a `semantic_router.agent` span per subagent the
Router creates for that run, so all of the run's calls share a trace. See
[Agent runs](../plugin/router-replay#agent-runs).

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
| Request graphs | `llm_request_graph_node_duration_seconds` (by `node_type` and `template`) |
| Cache | `llm_cache_plugin_hits_total`, `llm_cache_plugin_misses_total`, `llm_cache_warmth_estimate` |
| RAG | `rag_retrieval_attempts_total`, `rag_retrieval_latency_seconds`, `rag_cache_hits_total`, `rag_cache_misses_total` |
| Router Memory | `llm_memory_retrieval_total` (`backend`, `status`), `llm_memory_retrieval_latency_seconds`, `llm_memory_retrieval_results`, `llm_memory_store_operations_total`, `llm_memory_cache_hits_total`, `llm_memory_cache_misses_total`, `llm_memory_extraction_total` |
| Session | `llm_session_model_transitions_total`, `llm_session_turn_prompt_tokens`, `llm_session_turn_completion_tokens`, `llm_session_turn_cost` |
| Translation and request-parameter policy | `llm_translation_lossy_total` (by `reason`, with any count in it written as `N`), `sr_request_params_blocked_total` |
| Signals | `llm_signal_extraction_total`, `llm_signal_match_total`, `llm_signal_extraction_latency_seconds` |
| Configuration lifecycle | `llm_config_updates_total` (`source`, `result`, `stage`), `llm_config_update_duration_seconds`, `llm_config_active_version`, `llm_config_active_info` (`version`, `hash`), `llm_config_last_rejection_timestamp_seconds` |
| Remote model calls | `llm_remote_connector_requests_total` (by `operation`, `outcome`), `llm_remote_connector_request_duration_seconds`, `llm_remote_connector_retries_total` |
| Recipe routing | `llm_entrypoint_requests_total`, `llm_recipe_selections_total`, `llm_routing_stage_duration_seconds` (stages `signals`, `decision`, `algorithm`, `prompt_compression`), `llm_prompt_compression_total` (by `recipe` and `outcome`: `compressed`, `skipped_disabled`, `skipped_min_length`, `skipped_max_tokens`) |
| Context window | `llm_context_window_checks_total` (by `count`: `exact` or `proxy`, and `verdict`: `fits`, `in_band` or `exceeds`), `llm_context_window_refusals_total` (by `count`), `llm_context_window_margin` (by `count`), `llm_context_window_wait_seconds`, `llm_context_window_tokenizer_loaded` (by `tokenizer`), from the [context window check](../../installation/model-configuration#check-prompts-against-the-context-window) |
| Routing input | `llm_routing_input_removed_total` (by `kind`: `skipped_message`, `stripped_block`), from [`global.router.routing_input`](../signal/overview#choose-the-user-text-signals-read) |
| Streamed request body | `llm_streamed_body_arrival_seconds`, `llm_streamed_body_bytes`, `llm_streamed_body_chunks` (by `recipe`; STREAMED and FULL_DUPLEX_STREAMED modes only) |
| Projections | `llm_projection_score` (by configured recipe and projection name) |
| Trace export | `llm_trace_export_spans_total` (by exporter batch result) |

Router Memory counters use bounded labels only. See the [Router Memory Prometheus label release note](../../release-notes/router-memory-prometheus-labels).

A request that names a model the configuration does not serve counts in
`llm_request_errors_total` as `model="unknown"`, `reason="model_not_found"`;
the `specified_model_not_found` log event keeps the name it sent.

`llm_request_graph_node_duration_seconds` times each step of a request graph,
such as a Looper template's algorithm and respond steps; a step that holds
others counts their time too. `template` names the graph (`looper.fusion`, for
example) and `node_type` the step's registered type.

`llm_request_outcomes_total{traffic_kind="inference"}` counts each public
inference request once, at its terminal boundary:

- **Traffic kinds.** Authenticated internal looper requests use
  `inference_internal`; catalog, response-object and health operations have
  their own kinds.
- **Outcomes** distinguish success, client and server errors, cancellation,
  timeout, incomplete responses and other failures.
- **One request, one count.** A backend selection or an error event is not
  another completed public request.

The model latency names describe what they observe:

| Metric | Measures |
| --- | --- |
| `llm_model_first_response_observation_seconds` | The first stream chunk, or a non-streaming response's headers, of successful responses only |
| `llm_model_response_duration_per_output_token_seconds` | Complete response time divided by reported output tokens |

Neither is first-token latency or decode-only inter-token latency. The
optional windowed metrics summarize at most 10,000 completed responses per
model; they don't estimate utilization, queue depth or provider error rates.

The `semantic_router.request` span covers the complete ExtProc request,
streamed responses and cancellation included:

- **Bounded labels.** Its `traffic.kind` and route separate inference from
  catalog or health polling, without storing URL queries or resource IDs.
- **Child stages:** signal evaluation, decision evaluation, algorithm
  selection, plugins, and the actual upstream request.
- **Routing attributes.** `routing.entrypoint`, `routing.recipe`,
  `decision.name` and `routing.algorithm` identify the routing boundary where
  available.
- **Selection and duration.** A `routing.backend.resolved` event records
  selection evidence; the upstream span measures provider duration.
- **Status.** If a local response guard blocks an upstream HTTP 200, the
  upstream span keeps 200 while the root records the final client status.

The upstream span also carries OpenTelemetry GenAI attributes, so GenAI-aware
trace backends can show each provider call:

| Attribute | Value |
| --- | --- |
| `gen_ai.operation.name` | `chat` |
| `gen_ai.provider.name` | The provider |
| `gen_ai.request.model` | The provider model ID sent upstream |
| `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens` | As the provider reports them; unset, never estimated, when it doesn't |
| `gen_ai.response.model` | Buffered responses only |

`model.name` keeps the Router's logical model name. The GenAI conventions are
still in Development status upstream, so their attribute names can change.

- **Signal evidence events** carry finite reported values and confidence
  separately, real zeros included; missing evidence stays absent, and aggregate
  signal evaluation never invents a confidence score.
- **Projection events** carry evaluated scores and configured names.
- **Never in traces:** raw prompts, signal errors and retrieval exception
  text. Earlier traces cannot be enriched after the fact.

The local `vllm-sr serve` stack provisions the **vLLM Semantic Router**
dashboard in Grafana:

- **Main groups:** public inference outcomes, recipe workflow, backend usage,
  plugins and response cache, and telemetry health.
- **Accounting and MoM panels** separate reported model usage from supported
  looper attempt evidence.
- **Means, not quantiles.** Recipe stage duration and the model timing panels
  are observed means; projection counts show evaluations, with individual
  scores in Insights.
- **Sources.** Prometheus scrapes both the Router and Jaeger's internal admin
  metrics.

:::note
An absent series means no measurement, not zero traffic or a healthy
collector. Export success counts completed SDK exporter batches; it does not
prove that every request was sampled or retained.
:::

Local telemetry storage:

| Store | Where | Retention |
| --- | --- | --- |
| Jaeger | The pinned all-in-one image, non-root Badger storage in the `<jaeger-container-name>-data` volume at `/tmp` | 7 days |
| Grafana | Database and preferences in `<grafana-container-name>-data` at `/var/lib/grafana`, as the image's non-root user | — |
| Prometheus | Its local TSDB | 15 days |

- Replacing a container keeps these stores; deleting one starts a new history.
  Keep telemetry resets separate from benchmark, authentication, learning and
  configuration stores.
- Moving an older in-memory Jaeger to Badger doesn't migrate its memory.
- Sampling stays controlled by the Router configuration, and a search limit is
  not a count of all retained traces.
- For multi-node trace storage, configure an external collector and storage
  instead of sharing this single-node Badger volume; see the
  [Jaeger 1.76 storage documentation](https://www.jaegertracing.io/docs/1.76/deployment/#badger---local-storage).

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
helm install vsr oci://ghcr.io/vllm-project/charts/semantic-router \
  --version 0.0.0-latest \
  --set router.skipProcessing.enabled=true
```

Enable this gate only when an authenticated upstream filter (Envoy AI
Gateway, ext_authz, route-level filters, etc.) is responsible for setting
or stripping the header on trust grounds. Background on the AI Gateway
interop pattern that motivates this gate lives in
[issue #1808](https://github.com/vllm-project/semantic-router/issues/1808).

### Debug Headers

`global.router.headers.debug` decides who may turn on the debug response
header, `x-vsr-trace`, with `x-vsr-debug: true`. The trace names the signal and
guard rules a request matched, so access is off by default:

| Value | Honors `x-vsr-debug` from |
| --- | --- |
| `none` (default) | No client |
| `any` | Every client; for development and tests |
| `authenticated` | Only the users and groups in `debug_principals`, as the identity headers your authenticator sets assert them |

```yaml
global:
  router:
    headers:
      debug: authenticated
      debug_principals:
        groups: [platform-sre]
```

See [VSR routing headers](../../api/headers#the-trace) for
what the trace carries.

### Router Replay

```yaml
global:
  services:
    router_replay:
      store_backend: postgres     # explicit durable, SQL-queryable audit storage
      async_writes: true
      postgres:
        host: postgres
        port: 5432
        database: vsr
        user: router
        password: ${ROUTER_REPLAY_POSTGRES_PASSWORD}
  plugins:
    - type: router_replay
      configuration:
        enabled: true
```

Router replay records routing data by default:

- **Router-wide.** Without a `router_replay` plugin anywhere, every request
  keeps a record of its routing data and none of its text. A `router_replay`
  plugin in `global.plugins` sets the capture defaults, and `enabled: false`
  there turns Replay off.
- **Per recipe or decision.** A recipe's or decision's `router_replay` plugin
  overrides it: `enabled: false` turns capture off there, and `enabled: true`
  opts in when the default is off.
- **Storage.** Without a configured backend, records go to the `file` store,
  which [keeps them across restarts](../plugin/router-replay#keep-records-across-restarts).

The `store_backend` field controls where routing-decision replay records are persisted. Available backends:

| Backend | Durability | Use case |
|---------|-----------|----------|
| `postgres` | Full SQL queryability, long-term audit retention | Production audit storage |
| `redis` | Survives router restart, shared across replicas | Lightweight deployments already running Redis |
| `milvus` | Vector-searchable replay records | Semantic replay search |
| `qdrant` | Vector-searchable replay records | Semantic replay search in a Qdrant deployment |
| `file` | Survives router restart on one Router; routing data only | The default for local and single-Router deployments |
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

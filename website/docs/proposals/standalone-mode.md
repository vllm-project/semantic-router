---
title: Standalone Mode
description: The Router serves the OpenAI-compatible API itself and runs the routing pipeline in process, so Envoy becomes optional. One transport-agnostic routing core with ext_proc and HTTP adapters, an Envoy-like upstream layer with timeout, retry and fallback, in-process request graphs, and a versioned configuration snapshot with hot reload and rollback.
created: 2026-10-06
status: Implemented
---

> **Status:** Implemented in [#4628](https://github.com/vllm-project/semantic-router/pull/4628) - **Created:** 2026-10-06 -
> **Tracking issue:** [#4623](https://github.com/vllm-project/semantic-router/issues/4623)
>
> **Lifecycle update:** Engine and Router modes share one persistent frontend
> and model pool. Router is the default even when a model is supplied. Use
> `vllm-sr serve ARTIFACT --engine` for Engine mode; see
> [the current Quickstart](../model-runtime/quickstart.md).

## Summary

Add a **standalone mode**: the Router itself is the OpenAI-compatible endpoint and reverse proxy, and the routing
pipeline runs in process, without Envoy or an ext_proc stream. `vllm-sr serve` uses it by default. Envoy stays
available as an explicit mode, `--gateway extproc`, where `vllm-sr` starts Envoy in front of the Router and the
Router serves ext_proc, as today. Kubernetes defaults to standalone mode too, and extproc stays the way to attach
an Envoy-based gateway you already run.

Both modes share **one routing core**. The core is transport-agnostic; ext_proc and HTTP are thin
adapters over it. Around the core standalone mode adds an **upstream layer** modeled on Envoy's clusters and
endpoints, first-class **timeout, retry and fallback**, a **request-graph executor** that runs Looper algorithms
inside the Router instead of looping back through Envoy, and a **configuration system** built on immutable,
versioned snapshots with warming, atomic activation, drain, history and rollback.

This proposal supersedes [Standalone HTTP Gateway](./standalone-http-gateway). That design kept the gateway as a
separate experimental binary; this one makes the Router binary serve both roles, keeps a single configuration
document, and makes standalone mode the default. Its analysis of the shared engine, the response pipeline
and the Looper executor carries over.

## Decisions

1. The Router gains a standalone mode: it accepts OpenAI-compatible traffic, runs the routing pipeline in
   process and proxies to model backends itself.
2. `--gateway standalone|extproc` selects the mode on every target, and `standalone` is the default. On docker,
   standalone runs only the Router container, with the model runtime inside, and the Router serves the
   listeners; `extproc` is today's stack, an Envoy container that `vllm-sr` starts in front of the Router, which
   serves ext_proc.
3. Kubernetes defaults to standalone as well. The Helm chart's Router pod serves the configuration's listeners,
   its Service exposes them, and readiness uses `/ready`. The operator defaults to standalone, and its
   `spec.gateway` integration selects extproc. `extproc` stays the opt-in for Envoy Gateway, AI Gateway, Istio
   and KServe integrations (the ext_proc Service on port 50051 and the five-header rule).
4. Everything runs in containers: `--target docker` (the default) or `--target kubernetes` (`k8s`, released in
   v0.4.0, stays one release as a hidden alias that warns). There is no bare-metal target. `--platform
   cpu|rocm|cuda` applies to both targets: on docker it selects the image and the GPU passthrough, and on
   kubernetes the same image plus the GPU resource request (`amd.com/gpu` or `nvidia.com/gpu`) in the generated
   Helm values. Engine mode (`vllm-sr serve MODEL --engine`) runs the model runtime in a container from the same image,
   on the docker target, using the release-pinned ROCm or CUDA dependencies. Numerical parity must still be validated for the selected hardware and profile.
5. `vllm-sr` is the only package on PyPI. The Router binary and the model runtime (`vllm-srun`) ship only inside
   one image family: `vllm-sr` (CPU, amd64 and arm64), `vllm-sr-rocm` and `vllm-sr-cuda` (amd64), which serve
   docker and kubernetes, both modes and engine mode. The former `extproc` and `extproc-rocm` images are alias
   tags of the same digests for one release. Upstream Envoy is used only for docker `extproc`.
6. The Router keeps talking to the model runtime over HTTP/JSON on a Unix domain socket. A binary fast path is
   considered only after the overhead is measured. Measured on CPU, the transport is about 2% of a runtime call,
   so there is none (see [Results](#results)).
7. Timeout, retry and fallback are part of the first release.
8. Looper request graphs run entirely inside the Router and never loop back through Envoy.
9. The configuration system is modular and versioned, supports hot reload and rollback, and borrows the core
   ideas of Envoy's configuration design.

## Background

Before standalone mode, a request traveled:

```text
client -> Envoy -> ext_proc (gRPC; bodies buffered whole) -> Router (Go) -> model runtime over a Unix socket
       <- header and body mutations <-
Envoy -> backend chosen by x-selected-model -> Envoy -> ext_proc (response phases) -> Router -> Envoy -> client
```

The default Envoy template processes request headers, the request body, response headers and the response body,
so every request costs four ext_proc messages and a full buffer of both bodies.

A Looper request graph (fusion, quorum, confidence escalation, ReMoM, ratings, grounding, workflows) sends every
intermediate call back to the local Envoy listener with the internal `x-vsr-looper-request`,
`x-vsr-looper-secret`, `x-vsr-looper-decision` and `x-vsr-looper-iteration` headers. The call re-enters ext_proc,
is recognized as internal, and only then reaches a backend: Router -> Envoy -> Router -> Envoy -> backend for
each hop.

This shape has costs:

- Deploying and debugging need two configurations and two mental models: Envoy's and the Router's.
- The Looper loopback exists only because ext_proc cannot issue an upstream call through Envoy's routing. It
  lengthens every hop, complicates the semantics, and needs a shared secret header to resist forgery.
- Every local stack runs an Envoy container, rendered from a second template, in front of the Router, even
  when one machine serves one user.

## Modes

Gateway transport (`--gateway`) and serving capability (`--engine`) are separate choices.
Router mode is the default; supplying `MODEL` changes its judgment model without
disabling routing. Engine mode disables recipe routing while retaining the frontend
and native model APIs.

| Command | What runs | Client traffic enters at | Use it for |
| --- | --- | --- | --- |
| `vllm-sr serve` (= `--gateway standalone`, default) | the Router container, with its managed runtime processes inside | the Router's OpenAI-compatible port | one machine, development, edge, most self-hosting |
| `vllm-sr serve --gateway extproc` | an Envoy container started by `vllm-sr`, in front of the Router container (ext_proc) | Envoy | Envoy features such as rate limiting, mTLS, advanced route matching |
| `vllm-sr serve MODEL --engine` | the persistent frontend and model workers (Engine mode) | the frontend’s System One API | calling a model from your own code |

Both gateway modes share one routing core: standalone mode uses the HTTP adapter, and extproc mode (called Envoy
mode below) uses the ext_proc adapter. On Kubernetes, `--target kubernetes` installs the Helm chart in either
mode, and an Envoy-based gateway you run (Envoy Gateway, AI Gateway, Istio, KServe) attaches in extproc mode.

Orthogonal parameters:

- `--target docker|kubernetes`: `docker` by default.
- `--platform cpu|rocm|cuda`: the image on both targets; GPU passthrough on docker, the GPU resource request on
  kubernetes.
- `--container-runtime docker|podman` (formerly `--runtime`, kept for one release as a hidden alias that warns).
- Every `serve` flag belongs to one group, and a group names where it applies: docker, kubernetes, engine mode
  or several of them. `--platform`, `--image` and `--log-level` apply to all three; `--minimal` and
  `--readonly` to both targets; `--image-pull-policy` and `--container-runtime` to docker and engine mode.
  `--help` prints the groups, and a flag used where its group does not apply is an error.
- `--minimal`: no dashboard and no observability stack.

## Architecture

### Ordinary requests

```text
Envoy mode (today)
client --> Envoy --gRPC ext_proc (buffered)--> Router --UDS--> runtime
                 <-- header and body mutations --
           Envoy --> backend LLM --> Envoy --ext_proc--> Router --> Envoy --> client

Standalone mode
client --> Router [listener -> ingress hygiene and auth -> routing core -> upstream layer] --> backend LLM
                                     |                                          |
                                     +--UDS--> runtime (signals, embeddings)    +-- stream --> response phases --> client
```

Standalone mode removes one proxy hop, at least four ext_proc messages per request, and the whole-body buffering and
re-serialization they require. Configuration shrinks from "Envoy config plus Router config" to one document.

### Request graphs (Looper)

```text
Today
client -> Envoy -> ext_proc (Router starts the Looper algorithm)
                    |- hop 1: Router -HTTP-> Envoy listener -> ext_proc (internal request) -> Envoy -> backend A
                    |- hop 2: Router -HTTP-> Envoy listener -> ext_proc -> Envoy -> backend B
                    `- aggregate -> immediate response -> Envoy -> client

Standalone mode (the graph runs inside the Router)
client -> Router: Plan -> graph executor
                    |- call A --upstream--> backend A   (per-hop plugins in process; timeout, retry, fallback)
                    |- call B --upstream--> backend B   (parallel fan-out)
                    |- aggregate / branch / loop
                    `- respond (the last hop can stream) --> client
```

The envoy and extproc modes use the same executor. Intermediate hops go straight to the upstream layer, so there
is no loopback. The final hop is either handed to Envoy as an ordinary routed request (keeping streaming) or
returned as an immediate response. The internal secret header retires with the loopback.

### Configuration management

```text
Today:  config.yaml --fsnotify--> Router rebuilds a generation (atomic swap)
        + the vllm-sr CLI renders an Envoy config (a second lifecycle)

Standalone: config source (file / HTTP API / CRD / a future control plane)
          --> compile   (canonical document -> typed resources)
          --> validate  (schema + cross-references + capability checks)
          --> warm      (upstream pools, runtime models ready, precomputation)
          --> activate  (atomic snapshot swap)
          --> drain     (requests on the old snapshot finish)
          --> history   (the last N versions, ready for rollback)
```

## Detailed design

### Routing core

The request and response handling that lives in `pkg/extproc` today becomes a transport-agnostic routing core
behind a small contract in a new package, `pkg/routing`. The ext_proc server and the standalone HTTP frontend are
adapters of that contract.

```go
// Processor opens one routing session per request; the routing core implements it.
type Processor interface {
    Open(ctx context.Context) (Session, error)
}

// Session processes one request's phases in order and answers each with an Effect:
// header and body mutations, a route-cache refresh, a streamed response body, or an
// immediate response.
type Session interface {
    RequestHeaders(header Header, endOfStream bool) (*Effect, error)
    RequestBody(body []byte, endOfStream bool) (*Effect, error)
    ResponseHeaders(header Header, endOfStream bool) (*Effect, error)
    ResponseBody(body []byte, endOfStream bool) (*Effect, error)
    Evidence() Evidence
    Close(err error)
}

// Engine is the transport-agnostic routing core shared by every gateway mode.
type Engine interface {
    // Plan resolves the entrypoint and recipe, evaluates signals, the decision and
    // request-side plugins, and returns what to do with the request.
    Plan(ctx context.Context, req *Request) (*Plan, error)
    // Respond runs response-side plugins on a buffered or streamed upstream response.
    Respond(ctx context.Context, plan *Plan, resp *UpstreamResponse) (*Response, error)
}

type Plan struct {
    Immediate *Response // block, cache hit, policy answer, completed Looper result
    Call      *Call     // one upstream call: the request as it leaves the Router, and its route
    Budget    Budget    // deadline, maximum hops, token or cost ceiling
    Evidence  Evidence  // signals, decision, selected candidates (for headers and traces)
}
```

- **One pipeline, one set of semantics.** Requests and responses move through the same phases in every mode:
  request headers, request body, response headers, response body. The ext_proc adapter encodes each effect as an
  ext_proc message and Envoy applies it. `routing.NewEngine` drives a session with the local Envoy template's
  processing mode (headers both ways, bodies buffered both ways unless the response is switched to streaming) and
  applies the effects with Envoy's own rules: removals before sets, routing headers (`host`, `:authority`,
  `:method`, `:scheme`) and `x-envoy-*` headers left alone, content lengths checked against mutated bodies, and
  immediate responses rendered as Envoy renders local replies. The upstream request and the client response are
  therefore identical in both modes.
- **The call and its route.** `Plan.Call` carries the upstream request after every Router mutation (method,
  path, headers, body) and its route key: the `x-selected-model` value Envoy's route table matches, taken from the
  mutated headers once an effect clears the route cache, as Envoy does. An empty route key means the default
  route. The request-graph executor adds `Plan.Program` for multi-step graphs.
- **One lifecycle per request.** A session pins one router generation (later, one configuration snapshot) for the
  whole request, exactly as an ext_proc stream does, and `Plan.Finish` ends it once, whatever the outcome: replay
  records, in-flight admission, session telemetry and trace spans close on the same path for both adapters.
- **Staged extraction.** The first step adds the contract, implements `routing.Processor` on the existing pipeline,
  and shares each phase's reply logic between the gRPC stream and the session. The pipeline's internals still
  build ext_proc messages as their internal representation, decoded at one strict boundary that fails closed on
  anything the contract cannot express. They move behind the boundary phase by phase, each step guarded by the
  parity recorder, so the ext_proc behavior stays byte for byte identical throughout.
- **Parity recorder.** `pkg/routing/parity` records, for a request set, each phase's input and effect, the
  upstream request after mutations, the route, the client response, and the routing evidence (decision, model,
  recipe, signals). The normalizer rewrites only volatile values (latencies, trace context, clock fields) and the
  order of header removals, which Envoy applies before any set. The repository's corpus runs three ways: through
  the ext_proc gRPC adapter with the exact messages pinned as goldens, through in-process sessions, and (end to end)
  through both gateway modes on the wire; all must agree.

### Standalone frontend

- **Listeners:** address, HTTP/1.1 and h2c, idle timeout (default as in the local Envoy template), request body
  limit, connection limit, optional downstream TLS.
- **Ingress hygiene, the edge's trust boundary:** client-supplied internal headers (`x-vsr-*` internal headers,
  `x-authz-user-*`) never reach routing decisions or backends. Neither do the proxy-control headers that only a
  trusted proxy may set, such as `x-envoy-max-retries`, `x-envoy-retry-on` and `x-envoy-upstream-rq-timeout-ms`.
  The Router forwards client headers upstream, and Envoy-based layers behind it (sidecars, AI gateways in front of
  model servers) obey those headers from a caller they trust, which the Router is. Stripping them keeps the
  Router's reliability policy the one retry and timeout authority. The list follows the headers Envoy strips
  from external requests and stays that narrow; it protects the edge and does not emulate Envoy.
- **API keys:** a Bearer token or `api-key`, checked and then removed so client credentials never reach a
  provider; failures return an OpenAI-style 401 JSON error.
- **Endpoints:** `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/models`, plus the other inference
  surfaces the Router serves today; any other path is forwarded to the default backend the way Envoy's default
  route does.
- **Operations:** `/health`, `/ready` (ready only when the routing core and the runtime are), `/metrics`, and a
  graceful drain on SIGTERM.

### Differences from Envoy mode

Standalone mode reproduces what Envoy and ext_proc do, and the parity suite holds it to that. These differences are
deliberate:

- **Probes.** The standalone listener answers `/health` and `/ready` itself; Prometheus metrics stay on the Router's
  metrics port.
- **Identity headers.** Client-sent `x-authz-*` identity headers are dropped before routing, because no
  authenticator stands in front of the standalone listener. Behind Envoy, ext_proc sees whatever the gateway in front
  of it forwards.
- **Proxy identity.** Standalone mode adds no `x-envoy-*` headers (such as `x-envoy-expected-rq-timeout-ms` and
  `x-envoy-original-host` upstream, or `x-envoy-upstream-service-time` downstream) and does not replace the
  backend's `server` header with `server: envoy`.
- **DNS endpoints.** A backend given by DNS name is one endpoint; Envoy's `STRICT_DNS` clusters make one host per
  resolved address.
- **Transport framing.** Content length and chunked framing follow Go's HTTP server, which may frame a buffered
  body with a length where Envoy would chunk it.
- **Cross-model fallback.** The upstream layer sends each candidate, so a candidate gets its provider model's
  retries, ejection and TLS. Envoy mode's response-phase fallback calls candidates over plain HTTP.
  - An unreachable candidate reaches the policy as the `503` or `504` local reply, not as a transport error. That
    changes the outcome only for a policy that leaves those statuses out.
  - Standalone mode also falls back on Envoy's local replies, such as a refused connection, a reset or a timeout. Behind
    Envoy, ext_proc ends processing on Envoy's own local replies, so the client gets the reply and no candidate is
    tried. In both modes a local reply that reaches the client skips the response phases.
  - In standalone mode a candidate's request is built like the primary's, from the client's request. Envoy
    mode's candidate gets only the provider's headers.
  - Standalone mode commits to a candidate when its successful headers arrive. Envoy mode commits once it has
    translated the buffered body.
  - `per_attempt_timeout` and the rest of `total_timeout` bound a standalone candidate until its response starts, so a
    stream is never cut short.

### Upstream layer

Modeled on Envoy's clusters and endpoints:

- A **cluster** is one provider model; its **endpoints** are that model's backends.
- **Connection pools:** HTTP/1.1 keep-alive and HTTP/2; upstream TLS with SNI and the system CA bundle.
- **Load balancing:** weighted round robin and least request (power of two choices).
- **Rewrites and injection:** host rewrite (automatic or a literal), path-prefix rewrite, headers a provider
  needs, and credentials injected from secret references.
- **Circuit breaking:** maximum concurrent requests and maximum pending requests.
- **Outlier ejection:** `consecutive_5xx`, `base_ejection_time` and `max_ejection_percent`, already in the
  canonical `providers.models[].reliability`.
- **Active health checks:** `health_check_path`, `health_check_interval` and `health_check_timeout`, also already in
  the canonical configuration.
- **Streaming:** SSE flushed chunk by chunk with backpressure; response plugins that need the whole body
  accumulate it on the side.

The defaults reproduce what the local Envoy template configures today, so moving between modes does not change
load balancing, ejection or health behavior.

### Timeout, retry and fallback

**Timeouts** have cluster defaults that an entrypoint, recipe, decision or graph node can override:

- `connect`, `first_byte` (time to the first token), `per_try`, `total` (the request deadline), and `idle` (stream
  idle time).
- The deadline propagates along the call chain: runtime signal calls and every graph node share the same budget.

**Retries:**

- Conditions (`retry_on`) keep Envoy's tokens: `connect-failure`, `refused-stream`, `reset`, `gateway-error` (502,
  503, 504), `5xx`, and `retriable-status-codes` with the listed codes (such as 429, honoring `Retry-After` up to a
  bound). As in Envoy, a per-try timeout is retried under `5xx`, `gateway-error` and `reset`.
- A retry count, exponential back-off with jitter, and a retry budget (a cap on the share of requests that are
  retrying at the same time).
- A retry prefers a different endpoint.
- **Safety boundary:** once any response byte has been written to the client, the Router neither retries nor
  falls back.

**Decision overrides** come first: `routing.decisions[].reliability` takes the provider block's timeout and retry
fields. Both modes merge an override the way Envoy merges its per-request headers. Timeouts and the retry count
replace the provider model's; retry conditions and retriable status codes add to them.

- In Envoy mode the Router's gRPC reply carries the override as `x-envoy-upstream-rq-timeout-ms`,
  `x-envoy-upstream-rq-per-try-timeout-ms`, `x-envoy-max-retries`, `x-envoy-retry-on` and
  `x-envoy-retriable-status-codes`. Requests in standalone mode never carry them.
- ext_proc ignores `x-envoy-*` mutations by default, so every ext_proc filter the repository ships allows exactly
  these five headers with `mutation_rules.allow_expression`. The rule is always there, because the Router reloads
  its configuration without Envoy. `allow_envoy: true` would let the Router change any `x-envoy-*` header, far more
  than these fields need.
- The reliability block is the only writer of these headers: config load drops them from a decision's
  `header_mutation` plugin, with a warning.
- Envoy has no per-request header for an idle or first-byte timeout, a back-off or a `Retry-After` bound. A
  decision that sets one is rejected outside standalone mode, at startup, on reload and when the CLI renders Envoy.

**Fallback** is an ordered chain of candidates:

1. other endpoints of the same provider model: the cluster's job, through retries that prefer an endpoint not yet
   tried, outlier ejection and health checks;
2. other providers of the same model;
3. other models: the decision's ranked candidates;
4. last resort: a cached or static response (optional).

- Cross-model fallback is configured on a graph node, a decision (`routing.decisions[].fallback`), a recipe
  (`routing.fallback`) or globally (`global.router.fallback`), and merged field by field in that order. A cluster
  has no cross-model fallback of its own.

- Triggers are configurable by error class: timeout, 5xx, 429, connection failure, upstream content refusal.
- A budget caps the number of fallback hops and shares the total deadline.
- Response headers name the model that actually served the request and the fallback path; metrics and trace
  spans record every attempt.
- There is one fallback authority per request. The existing cross-model policy (`global.router.fallback` and
  recipe `routing.fallback`) stays the policy; the upstream layer executes it in standalone mode.

### Request-graph executor (Looper v2)

The Looper stops being a special flow that loops back through Envoy and becomes a general **request graph**
executed inside the Router. The graph runs inside the engine's `Plan`, in the request-body phase, in both modes.
It ends in an immediate answer, or in an ordinary routed call that the frontend (or Envoy) sends, so neither
frontend changes. Every hop is a routing session on the request's pinned generation and snapshot, marked as a
hop in process instead of by a header.

- **Node types**, extensible through a registry (plugins can add node types):
  - `call`: call one model or a candidate set, with its own timeout, retry and fallback;
  - `parallel`: fan out with a concurrency cap and an optional "first k results" rule;
  - `aggregate`: arbiter, quorum, fusion, ratings and other aggregation strategies;
  - `branch`: choose a path with the decision rule engine's predicates, signals and confidence;
  - `loop`: repeat until a condition holds or a round limit is reached;
  - `transform`: context compression, prompt rewriting, tool-result injection;
  - `respond`: emit the result; the last hop can stream.
- **Execution semantics:**
  - one execution context per request with a single deadline, maximum hop count and token or cost ceiling;
  - cancellation propagates: when the client disconnects or the budget runs out, every in-flight upstream call is
    cancelled;
  - each hop runs that hop's plugin chain in process, without forging an internal request;
  - one trace span per node; attempt evidence keeps the bounded `looper.AttemptTrace` structure.
- **Authoring:** graphs are declared in YAML inside a recipe or algorithm, with reusable subgraphs. Every existing
  Looper algorithm ships as a built-in graph template, so existing configurations keep their behavior unchanged.
- **Equivalence:** on deterministic fixtures, every existing Looper algorithm produces the same upstream call
  sequence and the same final result as before.

### Configuration system

The design borrows five core ideas from Envoy's configuration model, not xDS itself:

1. **Typed resources referenced by name:** listeners, routes, clusters, endpoints and secrets are independent
   resources that refer to each other by name.
2. **Typed extensions:** filters and extensions are identified by `type`, and each registered extension
   validates its own configuration.
3. **Atomic snapshots with warming:** a new configuration warms up before it takes effect as a whole; on failure
   the old configuration keeps serving.
4. **Versions with ACK/NACK:** every update has a version; an invalid update is rejected explicitly.
5. **Drain:** in-flight requests on the old configuration finish naturally.

How it lands here:

- **The authoring format does not change:** the canonical document
  `version/listeners/providers/evaluation/routing/entrypoints/recipes/global`.
- **It compiles into an immutable snapshot** of typed resources:

  | Resource | From the canonical document | Envoy analogue |
  | --- | --- | --- |
  | Listener | `listeners` | Listener + HTTP connection manager |
  | Route | the matching part of `entrypoints` | RouteConfiguration |
  | Program | `recipes`, `routing`, `evaluation` | HTTP filter chain + extension config |
  | Cluster / Endpoint | `providers` models and backends | Cluster / ClusterLoadAssignment |
  | Secret | credential references (env, file, Kubernetes Secret) | Secret (SDS) |
  | RuntimeModel | runtime deployments | none (specific to the Router) |

- **Typed extension registries:** signals, algorithms, plugins, graph nodes and filters register by `type` with
  their Go implementation, schema, defaults and validator; adding an extension needs no change to core code.
- **Validation:** the JSON schema generated from the Go types (`router-config-v0.3.schema.json` today), plus
  cross-reference checks, plus capability checks. A field the active mode cannot honor fails at load and points
  to the mode that can (for example `--gateway extproc`).
- **Lifecycle:** compile, validate, warm, activate (an atomic swap, as the router generation swap does today),
  drain, history.
- **Incremental rebuilds:** only the resources that changed are rebuilt. Changing only endpoints does not rebuild
  signals or model bindings; changing only a recipe does not rebuild upstream connection pools.
- **Versions and rollback:** every snapshot has a monotonic version and a content hash; the last N are kept;
  `POST /api/v1/config/rollback` returns to a chosen version; the active version appears in `/api/v1/config`, in
  metrics and in a response header.
- **Pluggable sources:** file watching (today), the HTTP API (today's `/api/v1/config`, gaining ACK/NACK semantics
  and audit entries), Kubernetes CRDs through the operator, and later an xDS-like control plane that pushes
  resources.
- **Scaling out:** Router instances are stateless and read the same configuration source; runtimes can attach to
  a shared GPU runtime pool.
- **Upgrades and migration:** the `version` field governs the schema version; layout changes go through
  `vllm-sr config migrate` (the runtime parser accepts only the canonical layout). Dry-run validation
  (`/api/v1/config/recipes/validate`) and routing preview (`/api/v1/routing/preview`) already exist.

### Router and runtime

- Unchanged: the runtime is a child process managed by the Router, reached over HTTP/JSON on a Unix domain socket,
  with one `/v1/bundle` call per request phase per runtime process.
- This mirrors vLLM's own split between its frontend and EngineCore: vLLM's Rust frontend is a separate process
  that talks to EngineCore over ZMQ and MessagePack; only the offline `LLMEngine` runs in process.
- The runtime stays out of the Router process because embedding PyTorch in Go needs cgo (which the Router just
  shed), the Python GIL and the Go scheduler interfere, and a GPU fault would take the Router down with it.
- The Router-to-runtime overhead is measured first; MessagePack or shared memory comes only if the numbers ask
  for it.

### Distribution and CLI

- `vllm-sr` is the only PyPI package: `pip install vllm-sr` on a host with Docker or Podman is the whole install.
- The Router binary and the model runtime (`vllm-srun`) ship inside one image family (`vllm-sr`,
  `vllm-sr-rocm`, `vllm-sr-cuda`) with one entrypoint for the CLI stack, Helm and the operator; neither is
  published on its own. The Router binary's own default stays extproc, so a manifest that passes no `-gateway`
  flag behaves as before; every launcher passes the mode.
- On Kubernetes the Helm chart takes `gateway.mode: standalone|extproc` (standalone by default), and the CLI's
  kubernetes target writes it together with the image and, for GPU platforms, the GPU resource request.
- In standalone mode `vllm-sr serve` starts only the Router container, which publishes the listeners and manages the
  runtime processes inside it. The dashboard reaches the Router directly, and the Router's `/ready` answers the
  dashboard's probes.
- Engine mode, `vllm-sr serve MODEL`, starts `vllm-srun serve` in a container from the same image, with ports
  mapped and the model cache mounted. There is no host engine path, so on macOS engine mode runs on CPU only.
- The release notes state the new default; `--gateway extproc` (or `gateway.mode: extproc` in Helm) restores the
  previous behavior.

### Observability

- **Access log:** the fields of today's Envoy access log.
- **Metrics:** requests, latency percentiles, upstream errors, retries, fallbacks, outlier ejections and graph
  node durations, merged with the existing Router metrics.
- **Tracing:** OpenTelemetry, one root span per request with child spans for the routing core, each runtime call,
  each graph node and each upstream attempt; `traceparent` propagates to backends.

## Package layout and dependency rules

| Package | Owns |
| --- | --- |
| `pkg/routing` | the transport-agnostic contract: `Engine`, neutral requests and responses, phase effects and how they apply, `Plan`, `Budget`, `Evidence` |
| `pkg/routing/parity` | parity records, recorders, the normalizer, the differ, golden files |
| `pkg/routing/graph` | the request-graph executor, its node registry and the built-in Looper templates |
| `pkg/upstream` | clusters, endpoints, pools, load balancing, health, ejection, timeout, retry and fallback execution |
| `pkg/gateway` | the standalone HTTP frontend |
| `pkg/extproc` | the ext_proc adapter and, until the extraction completes, the pipeline that implements `routing.Engine` |

- `pkg/routing`, its subpackages, `pkg/upstream` and `pkg/gateway` import neither Envoy types nor `pkg/extproc`.
- `pkg/gateway` depends on `pkg/routing` and `pkg/upstream`; nothing depends on `pkg/gateway`.
- Only the composition root (the Router's `cmd` and server wiring) binds the ext_proc engine to the standalone frontend.
- A dependency test enforces these rules.

## First release and follow-ups

| Capability | First release | Note |
| --- | --- | --- |
| Listeners, HTTP/1.1, h2c, timeouts, body and connection limits | yes | matches the local Envoy template |
| Downstream TLS (one-way) | yes | Go standard library |
| Internal-header stripping, API keys | yes | matches the Envoy Lua filter and header rules |
| In-process routing pipeline, immediate responses, streamed request bodies | yes | the core value |
| Upstream: load balancing, pools, host and path rewrite, header and credential injection, TLS/SNI | yes | matches Envoy clusters |
| Circuit breaking, outlier ejection, active health checks | yes | the canonical configuration already has these fields |
| Timeout, retry, fallback | yes | |
| SSE streaming with backpressure | yes | |
| Request-graph executor (Looper v2), equivalent to the existing Looper algorithms | yes | |
| Configuration snapshots, warming, atomic activation, drain, versions, rollback, incremental rebuilds, ACK/NACK, audit | yes | |
| Access log, metrics, tracing, health and readiness, graceful shutdown | yes | |
| `--gateway standalone` and `--gateway extproc` on both targets, engine mode in a container, `serve` flag groups | yes | |
| Helm and the operator default to standalone; one image family for both targets | yes | |
| OpenClaw and the fleet simulator (`vllm-sr-sim`) removed from code, CI, images, Helm and current docs | yes | historical blog posts and versioned docs stay |
| Rate limiting (token buckets per key or route) | follow-up | use `--gateway extproc` meanwhile |
| mTLS, JWT/OIDC, external authorization | follow-up | same |
| Regex or multi-header route matching, weighted splits, mirroring, request hedging | follow-up | same |
| WebSocket (realtime), HTTP/3 | follow-up | |
| An xDS-like control plane | follow-up | the first release ships file, API and CRD sources |
| GPU acceleration on macOS through a host bridge (`--platform apple`) | follow-up ([#4636](https://github.com/vllm-project/semantic-router/issues/4636)) | on macOS the built-in models run on the CPU in the arm64 image |

A configuration that needs a follow-up capability fails at startup in standalone mode and points to
`--gateway extproc`.

## Testing and acceptance

- **Parity suite:** the same request set (covering every kind of signal, decision, plugin and Looper algorithm)
  runs in standalone and Envoy modes; routing decisions, upstream requests and response transformations must match.
- **End-to-end:** the CLI integration suite in standalone mode on docker and in engine mode; a Kind profile on
  the Helm default (standalone); the existing gateway-integration profiles pin extproc and keep their results.
- **Fault injection:** fake backends produce timeouts, 5xx, 429, connection resets and mid-stream drops; the tests
  check retry, fallback, circuit breaking and outlier ejection, and that nothing is retried after bytes reach the
  client.
- **Looper equivalence:** every existing Looper algorithm produces the same call sequence and result on
  deterministic fixtures.
- **Configuration:** hot reload, rejected updates (NACK, the old version keeps serving), rollback, and in-flight
  requests unaffected during drain.
- **Performance record:** end-to-end latency and throughput in both modes (at least five interleaved rounds with
  95% intervals), and the Router-to-runtime share of the time, which decides whether the runtime fast path is
  worth building.
- **Packaging:** on a clean host with only Docker, `pip install vllm-sr` followed by `vllm-sr serve` (standalone
  mode) and `vllm-sr serve MODEL` (engine mode) works on CPU.

### Results

The measurements below record the implementation and hardware used at the time.
They are not benchmarks of the current default model, replica pool or every
supported profile.

Recorded on the final tree before merge, on one node (AMD EPYC 9575F, Docker 29.8.1, Envoy 1.35.3), against fake
backends:

- **Parity:** the 18-case corpus gives the same status, body, route and Router headers through Envoy with ext_proc
  and in standalone mode. Only the differences listed above appear: `server`, the `x-envoy-*` headers and body
  framing.
- **Fault injection:** a 503, a 429, a reset or a per-try timeout once, a 400, a stream's 503, a mid-stream drop
  and a 503 on every attempt give the same backend calls and client results in both modes. A reset or a timeout
  on every attempt differs as described above: standalone mode falls back on the local reply.
- **Looper:** Confidence (plain, streamed, with a failing small model), Ratings, ReMoM and Fusion (plain,
  streamed) make the same model calls and return the same responses.
- **Configuration:** about 209,000 requests across seven activations, a rejected file and two rollbacks under
  load: none failed, no worker saw a version go back, each version served exactly one document, and 30 streams
  completed, 14 of them across a change. A rollback while the file holds a rejected document succeeds.
- **Packaging:** on a host with only Docker, the CLI's wheel in a clean venv runs `vllm-sr serve` in both gateway
  modes and `vllm-sr serve MODEL` on CPU. The ROCm image's Router, run as the chart runs it (uid 65532, read-only
  root, no capabilities), reaches the GPU through its render group only.
- **Performance:** 10 interleaved rounds per mode and concurrency, with a request that matches no decision;
  means over rounds, with 95% intervals. Envoy runs on 8 cores and the ext_proc Router on 16.
  - **On the merged tree** (2,000 requests a round), with the standalone Router on 16 cores for 1 and 8 clients
    and on 24 for 32 and 64:

    | Clients | p50, ext_proc → standalone | p99, ext_proc → standalone | Requests/s, ext_proc → standalone |
    | --- | --- | --- | --- |
    | 1 | 0.74 → 0.57 ms | 2.02 → 1.82 ms | 1,280 → 1,644 |
    | 8 | 0.87 → 0.64 ms | 1.77 → 1.56 ms | 7,771 → 9,737 |
    | 32 | 2.17 → 2.20 ms | 4.55 → 6.28 ms | 14,067 → 13,435 |
    | 64 | 4.20 → 5.06 ms | 9.36 → 10.92 ms | 14,561 → 13,080 |

    From 32 clients standalone mode saturated 5–10% below Envoy mode, and more cores barely moved its ceiling.
  - **After [#4666](https://github.com/vllm-project/semantic-router/issues/4666)**, on the same node (2,000
    requests a round), with the standalone Router on 24 cores at every level, the same total as Envoy mode:

    | Clients | p50, ext_proc → standalone | p99, ext_proc → standalone | Requests/s, ext_proc → standalone |
    | --- | --- | --- | --- |
    | 1 | 0.57 → 0.41 ms | 1.12 → 0.95 ms | 1,692 → 2,318 |
    | 8 | 0.66 → 0.43 ms | 1.40 → 1.34 ms | 10,831 → 14,523 |
    | 32 | 1.03 → 0.93 ms | 2.54 → 3.46 ms | 26,744 → 27,935 |
    | 64 | 1.97 → 1.70 ms | 4.52 → 8.43 ms | 30,152 → 29,473 |

    The ceiling came from the routing core, which both modes share. Every request updated the TTFT history under
    a global lock and then copied the history out under it, so a copy stalled by garbage collection held up every
    request queued behind it. Requests also allocated about 300 KB each, mostly defensive copies of the provider
    catalog and log fields that the sampler then dropped. Without these, both modes about double. Standalone mode
    answers faster at every load and serves more requests per second up to 32 clients. At 64 clients the two are
    level in rounds of 2,000 requests, about 70 ms at these rates (−680 ± 721 requests/s); in rounds of 10,000,
    standalone mode serves 30,319 against 28,837 requests/s at 64 clients and 28,830 against 26,370 at 32. Its p99
    from 32 clients is higher: both Routers are CPU-bound there, and garbage collection still takes a third or more
    of their CPU. `BenchmarkNativeGatewayClients` in `internal/gatewayparity` sends the same request through the
    standalone path from 1, 8, 32 and 64 clients in process, so a request-path regression shows up without Envoy
    or Docker.
- **Router-to-runtime share:** with one CPU jailbreak signal (the 307M Vela Guard), a standalone request takes
  13.9 ms, 12.4 ms (89%) of it in the runtime call. The runtime now reports its own time for every request, and
  the Router records each call's transport ([#4667](https://github.com/vllm-project/semantic-router/issues/4667)).
  On CPU, inference is 95–99% of a call. The transport is 0.6% for the Vela 2.0 0.3B defaults, 1.9–2.2% for the
  Vela 1.0 signals and 2.1% for the Guard alone, and the runtime's own HTTP and JSON handling takes 0.4–1.6%.
  A fast path could save a request at most about 0.8 ms, so the Router keeps HTTP/JSON
  ([record](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/runtime-transport-cpu.md)).

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| The default mode changes under existing users, including Helm and operator installs | release notes, a startup log line, `--gateway extproc` or `gateway.mode: extproc` restores the old behavior; unsupported configurations fail fast |
| A home-grown proxy is less proven than Envoy | built from mature Go standard-library parts; fault-injection tests; Envoy mode stays the recommendation for production gateways |
| Rewriting the Looper changes behavior | equivalence tests before the switch; the old path stays in Envoy mode until equivalence is proven |
| The configuration schema changes | `vllm-sr config migrate` handles migrations; the runtime parser accepts only the canonical layout |

## Implementation phases

1. **P0, design:** the tracking issue and this document.
2. **P1, routing core:** `pkg/routing` and the ext_proc adapter as a pure refactor; every existing end-to-end
   profile unchanged; the parity recorder.
3. **P2, upstream layer:** clusters and endpoints, pools, load balancing, host and path rewrite, header and
   credential injection, TLS/SNI, circuit breaking, outlier ejection, active health checks, streaming proxy.
4. **P3, timeout, retry and fallback:** semantics, budgets, the safety boundary, observability, fault-injection
   tests.
5. **P4, standalone frontend:** listeners, TLS, API keys, header stripping, OpenAI endpoints, health, readiness and
   metrics, access log, tracing, graceful shutdown.
6. **P5, request-graph executor:** node registry, executor, budgets and cancellation, per-hop plugins, traces;
   the Looper algorithms become built-in templates and pass equivalence tests; the ext_proc modes use the
   executor, and the loopback and the internal secret header go away.
7. **P6, configuration system:** typed resource snapshots, extension registries, validation, warming, atomic
   activation, drain, version history and rollback, incremental rebuilds, ACK/NACK and audit on the API.
8. **P7, CLI and deployment:** `--gateway standalone|extproc` on both targets (standalone by default),
   `--target kubernetes`, engine mode in a container, `serve` flag groups and `--container-runtime`; the Helm and
   operator defaults; one image family; the CLI suite on a Docker-only host and a standalone Kind profile.
9. **P8, verification and documentation:** the parity suite, end-to-end profiles, the performance record, user
   documentation (choosing a mode, reliability, request graphs, configuration management), release notes.

## Relationship to other proposals

- [Standalone HTTP Gateway](./standalone-http-gateway): superseded by this proposal. Its shared-engine analysis,
  the single neutral response pipeline and the in-process Looper executor carry over; the separate binary, the
  separate bootstrap configuration and the experimental-only scope do not.
- [Model Execution Fallback](./model-execution-fallback): the fallback chain here is that boundary made concrete,
  with one fallback authority per request.
- [Multi-Protocol Adapter Architecture](./multi-protocol-adaptor): both adapters consume the same neutral codecs.
- [Router Flow Workflows](./router-flow-workflows): workflows become request graphs with the same behavior.
- [Unified Config Contract v0.3](./unified-config-contract-v0-3): the canonical document stays the only authoring
  format; snapshots are compiled from it.
- [vLLM Production Stack Integration](./production-stack-integration): the layering stands; standalone mode
  owns transport and backend selection for a logical model, while serving platforms keep replica lifecycle.

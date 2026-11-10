# VSR routing headers

The router uses these request and response headers for session continuity,
routing observability, replay correlation, and opt-in debugging. The
[header reference](#header-reference) at the end of this page lists every
header the Router reads or writes.

## What appears by default

Every response of the routing pipeline carries the public set: answers from
the upstream model, response-cache hits, and the responses the Router produces
itself, such as a block, a fast response or a routing error.

| Header | Meaning |
| ------ | ------- |
| `x-vsr-schema-version` | The revision of this header contract, `3`. |
| `x-vsr-config-version` | The version of the configuration that served the request. |
| `x-vsr-response-path` | What answered: `upstream`, `cache`, `fast_response`, `looper`, `image_generation`, `blocked`, `rate_limited`, `error` or `fallback`. |
| `x-vsr-selected-recipe`, `x-vsr-selected-decision`, `x-vsr-selected-model` | The final routing facts, each once routing reached it. |
| `x-vsr-replay-id` | The Router Replay record, when replay is enabled and the record was persisted. |
| `x-vsr-warnings` | The [response warnings](#response-warnings), omitted when none applies. |
| `x-vsr-cost`, `x-vsr-cost-currency` | The [configured price](#cost-headers) of a buffered response. |

- `x-vsr-selected-decision` is omitted when no decision matched and the request
  went to the default model; `x-vsr-selected-model` then names the default
  model.
- A request keeps the configuration version it started on, even when a hot
  reload activates another one while it runs. Responses answered by the
  Router's own handlers, such as `GET /v1/models`, carry
  `x-vsr-config-version` only.
- The one management endpoint that sets an `x-vsr-*` header is
  `POST /api/v1/routing/preview`, which returns `x-vsr-config-hash` on a
  successful result.

Everything else the Router knows about a request, such as the matched signals,
the confidence and the algorithm, rides in one debug header,
[`x-vsr-trace`](#the-trace). When replay is enabled, the same context remains
available through `x-vsr-replay-id`.

Clients steer a request with the `control` headers in the reference, such as
`x-vsr-session-id`, `x-vsr-memory` or `x-vsr-debug`. The Router drops its other
`x-vsr-*` headers when a client sends them, and every `x-vsr-*` header a
provider returns.

A header reaches a provider only when its direction in the reference lists
`provider request`: the Router removes its own control headers, its internal
headers, identity and secrets from every provider-bound request.

## Who may turn on the trace

The trace names the signal and guard rules a request matched, the decision
ranking and the Router Learning reasons, so a client can't turn it on by
itself. A client asks for it with `x-vsr-debug: true`, and
`global.router.headers.debug` decides who may; the Router ignores
`x-vsr-debug` from everyone else.

| `debug` | Who gets the trace |
| ------- | ------------------ |
| `none` (default) | Nobody. |
| `any` | Every client that sends `x-vsr-debug: true`. Use it for development and tests, or behind a gateway that removes the header from untrusted clients. |
| `authenticated` | Only the users and groups in `debug_principals`, as the identity headers your authenticator sets assert them (`global.services.authz.identity`). |

```yaml
global:
  router:
    headers:
      debug: authenticated
      debug_principals:
        users: [oncall-bot]
        groups: [platform-sre]
```

`authenticated` trusts the identity headers. A standalone Router keeps them only
on a listener with `identity.trust_headers`, and refuses `authenticated` when no
listener trusts them. Behind an Envoy-based gateway, the gateway must remove the
identity headers that clients send. The Dashboard Playground,
`vllm-sr route --debug` and sr-bench ask for the trace; when the caller isn't
allowed, they show the public set only.

## The trace

`x-vsr-trace` is an [RFC 8941](https://www.rfc-editor.org/rfc/rfc8941)
dictionary of at most 4 KB. The table after the header reference lists its
members, highest priority first; each is omitted when it doesn't apply. A
string percent-encodes `%` and every byte outside printable ASCII, so decode it
after unquoting. When the encoded trace would exceed 4 KB, the Router leaves out
members from the lowest priority up and ends the dictionary with `truncated`.

```http
x-vsr-trace: confidence=0.91, algorithm=static, routing_latency_ms=3.2, signals=("context:long-context" "projection:use-reasoning-model"), session_phase=tool_loop
```

- `confidence` is absent when the decision rests on policy leaves, aggregates
  several evidence leaves, or resolves through an error policy.
- `fallback_attempts` counts the same way in both gateway modes and is omitted
  when the primary answered.
- `ranking` is omitted when only one decision matched; parse
  `decision_ranking` in Router Replay for the full ranking.
- `learning` lists the Router Learning methods with their `action`. Show the
  actions as user-facing phrases such as `tool/protocol pinned`,
  `model switched`, or `learning bypassed`; fresh conversation or session-start
  diagnostics are usually useful only in debug views, as neutral status text.
- `effective_input_tokens` and `effective_max_output_tokens` appear together on
  a successful upstream response when automatic output was resolved, in the
  initial headers of streaming and buffered responses alike. Explicit output
  limits, cache hits, skipped processing, and Looper responses omit them. Their
  values come from the final dispatch after request changes and backend
  rendering; they are not estimates or the model's configured maximum context
  length.
- `config_hash`, `inference_calls` and `model_usage` are the receipts of a
  config-bound evaluation request, one sent with
  `x-vsr-bench-expected-config-hash`.
- `retention` shows the fields a matched decision's retention directive set, so
  operators can verify policy wiring; clients should not use them as commands.
- `prompt_cache` is content-free: it reports counts and outcomes, never the
  cached text itself, and it never claims a cache hit or provider-side savings.

Projection scores and full projection traces are stored in router replay records rather than expanded into the trace. Use `x-vsr-replay-id` to inspect those details in the Dashboard or through the authenticated Router management API; public inference listeners do not serve replay records.

## Response warnings

`x-vsr-warnings` is an RFC 8941 list of warning codes, omitted when none
applies. The codes of the `response_guard` rules that annotated the answer come
first, in rule order, then the protocol translation losses:

| Code | Meaning |
| ---- | ------- |
| `hallucination` | A rule over a `hallucination` signal flagged the answer. |
| `unverified_factual` | The answer was factual, but there was no context to check it against. |
| `response_jailbreak` | A rule over a response-stage `jailbreak` signal flagged the answer. |
| `response_guard` | A rule over any other signal flagged the answer. |
| `protocol` | Protocol translation changed or dropped part of the request, with its `action`, `reason` and `field`. |
| `truncated` | Warnings left out to keep the header within 4 KB, with their `count`. |

Per-warning detail, such as hallucination spans or jailbreak confidence, is
kept in the replay record instead of being expanded into response headers.

## Session headers

`x-vsr-session-id` names the session a request belongs to: sessions pin to it,
and Router Learning protects it unless `identity.headers.session` names another
header. A client that already sends a session id under its own name keeps it:
`global.router.headers.session_id_headers` lists the names the Router also
accepts, `x-session-id` and `x-claude-code-session-id` by default.
`x-vsr-session-id` wins when a request carries several.

```yaml
global:
  router:
    headers:
      session_id_headers: [x-session-id, x-claude-code-session-id]
```

`x-conversation-id` names one conversation for Router Learning's
`scope: conversation`. `x-vsr-memory: off` keeps Router-managed memory out of a
request.

## Cross-model KV transfer headers (issue #2976)

When the router attempts cross-model KV reuse on a model switch, it injects
request-side hints on the upstream call: `x-vsr-kv-source-pod`,
`x-vsr-kv-cache-id` and `x-vsr-kv-mapper-id`. These headers are internal to the
router and inference pool; clients should not depend on them.

## Cost headers

On a buffered (non-streaming) response, the router prices the usage the model
reported with the served model's `pricing` configuration. This is a
configured-price figure, not a provider bill. Streamed responses and models
without `pricing` omit both headers.

## Example response

The public set:

```http
HTTP/1.1 200 OK
Content-Type: application/json
x-vsr-schema-version: 3
x-vsr-config-version: 7
x-vsr-response-path: upstream
x-vsr-selected-recipe: default
x-vsr-selected-decision: complex-request
x-vsr-selected-model: reasoning-model
x-vsr-replay-id: replay_01J...
```

`x-vsr-replay-id` appears here only because Router Replay was enabled and the
record was persisted. With replay disabled, or when persistence fails, the
response carries the other headers and omits this one.

With `x-vsr-debug: true` on the request from a caller that
`global.router.headers.debug` allows, the response carries the trace as well:

```http
x-vsr-trace: algorithm=static, routing_latency_ms=1.8, signals=("context:long-context" "projection:use-reasoning-model"), session_phase=tool_loop
```

## Identity headers

An authenticator in front of the Router sets these, and the Router reads them
for role bindings, rate limits, memory, and the user, team and tenant
response-cache scopes. A listener drops the ones a client sends unless it
trusts identity headers (see
[Gateway modes](../installation/gateway-modes#identity-headers)), and no
identity header reaches a model provider.

| Header | Description |
| ------ | ----------- |
| `x-authz-user-id` | The authenticated user. `global.services.authz.identity.user_id_header` names another header. |
| `x-authz-user-groups` | The user's groups, comma-separated. `global.services.authz.identity.user_groups_header` names another header. |
| `x-authz-team-id` | The team, for the `team` response-cache scope. |
| `x-authz-tenant-id` | The tenant, for the `tenant` response-cache scope. |

## Compatibility and interpretation

- Use `x-vsr-schema-version` before parsing the other headers; the current
  value is `3`. Responses answered by the Router's own handlers, such as
  `GET /v1/models`, never carry it, so its absence there is expected rather
  than a contract violation.
- `x-vsr-config-version` counts the activations of one Router, which keeps its
  own configuration history. To check that replicas serve the same
  configuration, compare the document hash that
  `GET /api/v1/config/hash` reports as `active_runtime_hash`.
- Recipe names scope local signal, projection, decision, cache, replay, metric, and learning/session identities. Use `x-vsr-selected-recipe` together with the local decision/signal names when correlating a response with Insights or metrics.
- `event` is the public signal type used by decisions and DSL. Canonical YAML stores event rules under `recipes[].signals.events`, matching other plural signal containers.
- Router Learning uses router-owned online state internally. Users enable online model-choice learning through `global.router.learning.adaptation`, enable stability protection through `global.router.learning.protection`, pass stable identity headers, and optionally set `recipes[].decisions[].adaptations.mode`, component modes, or `adaptations.adaptation.candidate_set`. `scope: conversation` protects one `x-conversation-id`; `scope: session` protects the broader session of `x-vsr-session-id`. The old `recipes[].decisions[].algorithm.session_aware` shape is not part of the public contract.

## Header reference

The Router's header registry, `src/semantic-router/pkg/headers/registry.go`,
generates this table; run `make headers-generate` after changing it.

- **Direction** — `request` (client to Router), `response` (Router to client),
  `provider request` (Router to provider) and `provider response` (provider to
  Router).
- **Surface** — `public` rides on the default response, `debug` only for a
  caller allowed the trace, and `control` headers are sent by clients to steer
  a request. `internal` headers are not part of the client contract, and no
  response the Router writes carries them.
- **Sensitivity** — `identity` values name an authenticated caller and `secret`
  values are credentials. Neither reaches a client; a per-user key reaches a
  provider only as its credential.
- **Since** — the release the header first shipped in.

<!-- BEGIN-GENERATED-HEADER-REFERENCE -->

| Header | Direction | Surface | Sensitivity | Owner | Since | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `x-content-type-options` | response | public | none | router_replay | v0.4 | `nosniff` on Router Replay management responses. |
| `x-envoy-overloaded` | response | public | none | upstream | v0.5 | `true` on the 503 a full circuit breaker returns, as Envoy sets it. |
| `x-envoy-upstream-service-time` | response | public | none | gateway_parity | v0.5 | Envoy's upstream time on responses in Envoy mode; gateway-parity checks ignore its value. |
| `x-ratelimit-limit` | response | public | none | rate_limit | v0.2 | Limit of the rate-limit window that rejected the request. |
| `x-ratelimit-remaining` | response | public | none | rate_limit | v0.2 | Requests left in that window. |
| `x-ratelimit-reset` | response | public | none | rate_limit | v0.2 | Unix time the window resets at. |
| `x-request-id` | request, response, provider request | public | none | observability | v0.1 | Request correlation id, forwarded to providers; the native gateway adds one when absent, and the management API echoes it. |
| `x-vsr-config-hash` | response | public | none | config | v0.4 | SHA-256 of the active runtime document, on routing previews and configuration activations. |
| `x-vsr-config-version` | response | public | none | config | v0.5 | Version of the configuration snapshot that served the request. |
| `x-vsr-cost` | response | public | none | cost | v0.4 | Usage priced with the served model's configured pricing; buffered responses only. |
| `x-vsr-cost-currency` | response | public | none | cost | v0.4 | Currency of `x-vsr-cost`, from `pricing.currency`. |
| `x-vsr-replay-id` | response | public | none | router_replay | v0.2 | Router Replay record id, when the record was persisted. |
| `x-vsr-response-path` | response | public | none | response_contract | v0.4 | How the response was produced: `upstream`, `cache`, `fast_response`, `looper`, `image_generation`, `blocked`, `rate_limited`, `error` or `fallback`. |
| `x-vsr-schema-version` | response | public | none | response_contract | v0.4 | Response-header contract revision; currently `3`. |
| `x-vsr-selected-decision` | response | public | none | routing | v0.1 | Decision the decision engine selected; omitted when no decision matched. |
| `x-vsr-selected-model` | response | public | none | routing | v0.1 | Model alias the Router selected. |
| `x-vsr-selected-recipe` | response | public | none | routing | v0.4 | Recipe the effective entrypoint selected; omitted for concrete backend models. |
| `x-vsr-warnings` | response | public | none | response_contract | v0.5 | Warnings about the response, as an RFC 8941 list: the codes of the `response_guard` rules that annotated the answer, then each `protocol` translation loss with its `action`, `reason` and `field`. |
| `x-vsr-trace` | response | debug | none | debug_headers | v0.5 | Routing details as an RFC 8941 dictionary of at most 4 KB, for a caller `global.router.headers.debug` allows; the members follow this table. |
| `x-claude-code-session-id` | request, provider request | control | none | session | v0.3 | Conversation id Claude Code sends, which `session_id_headers` accepts by default after `x-session-id`; providers receive it. |
| `x-conversation-id` | request, provider request | control | none | router_learning | v0.4 | Client conversation id; Router Learning `scope: conversation` protects it, and providers receive it. |
| `x-session-id` | request, provider request | control | none | session | v0.3 | Client session id `session_id_headers` accepts by default; providers receive it. |
| `x-vsr-bench-expected-config-hash` | request | control | none | benchmark | v0.5 | Configuration hash an evaluation request requires; another active configuration fails it with 412. |
| `x-vsr-bench-max-inference-calls` | request | control | none | benchmark | v0.5 | Provider-call limit of an evaluation request; only `1` is supported. |
| `x-vsr-cache-control` | request | control | none | response_cache | v0.4 | Response-cache directives for the request, when the plugin's `request_controls` allow them. |
| `x-vsr-compression-control` | request | control | none | context | v0.4 | Context-compression directives for the request, when the plugin's `request_controls` allow them. |
| `x-vsr-debug` | request | control | none | debug_headers | v0.4 | `true` asks for `x-vsr-trace` on the response; honored only for a caller `global.router.headers.debug` allows. |
| `x-vsr-memory` | request | control | none | memory | v0.5 | `off` keeps Router-managed memory out of the request. |
| `x-vsr-protection-release` | request | control | none | router_learning | v0.5 | Any non-empty value asks router-learning protection for a fresh decision on the request's protection key; `global.router.learning.protection.release.header` can rename it. |
| `x-vsr-session-id` | request | control | none | session | v0.5 | Session id the client pins; without it the Router reads the headers `global.router.headers.session_id_headers` names, in order. |
| `x-vsr-skip-processing` | request | control | none | skip_processing | v0.3 | `true` opts the request out of Router processing when `global.router.skip_processing.enabled` is set. |
| `api-key` | request, provider request | internal | secret | credentials | v0.2 | Azure OpenAI key: a client's authenticates to the Router and is removed, and the Router sets the provider's. |
| `api-subscription-key` | provider request | internal | secret | credentials | v0.4 | Provider credential a catalog provider names. |
| `authorization` | request, provider request | internal | secret | credentials | v0.1 | Bearer credential: a client's key for the Router, and the provider credential the Router sets. |
| `ocp-apim-subscription-key` | provider request | internal | secret | credentials | v0.5 | Azure API Management credential, which a model connection sets through `auth`. |
| `x-api-key` | provider request | internal | secret | credentials | v0.2 | Provider credential of Anthropic-style APIs. |
| `x-authz-team-id` | request | internal | identity | authz | v0.4 | Authenticated team id; partitions the response cache. |
| `x-authz-tenant-id` | request | internal | identity | authz | v0.4 | Authenticated tenant id; partitions the response cache. |
| `x-authz-user-groups` | request | internal | identity | authz | v0.2 | Authenticated user's groups, comma-separated; `identity.user_groups_header` names another header. |
| `x-authz-user-id` | request | internal | identity | authz | v0.2 | Authenticated user id; `identity.user_id_header` names another header. |
| `x-envoy-expected-rq-timeout-ms` | request | internal | none | ingress | v0.5 | Envoy timeout control; the edge drops it from client requests. |
| `x-envoy-force-trace` | request | internal | none | ingress | v0.5 | Envoy tracing control; the edge drops it from client requests. |
| `x-envoy-hedge-on-per-try-timeout` | request, provider request | internal | none | benchmark | v0.4 | `false` on evaluation requests, so a per-try timeout never hedges. |
| `x-envoy-internal` | request | internal | none | ingress | v0.5 | Envoy's internal-request marker; the edge drops it from client requests. |
| `x-envoy-ip-tags` | request | internal | none | ingress | v0.5 | Envoy IP tagging control; the edge drops it from client requests. |
| `x-envoy-max-retries` | request, provider request | internal | none | reliability | v0.4 | Retry count the Router sets from the decision's `reliability` block, and `0` on evaluation requests. |
| `x-envoy-original-url` | request | internal | none | ingress | v0.5 | Envoy redirect control; the edge drops it from client requests. |
| `x-envoy-ratelimited` | provider response | internal | none | upstream | v0.5 | Marks a provider response an Envoy rate limit rejected; the native upstream can retry it. |
| `x-envoy-retriable-header-names` | request | internal | none | ingress | v0.5 | Envoy retry control; the edge drops it from client requests. |
| `x-envoy-retriable-status-codes` | request, provider request | internal | none | reliability | v0.5 | Retriable status codes the Router sets from the decision's `reliability` block. |
| `x-envoy-retry-grpc-on` | request | internal | none | ingress | v0.5 | Envoy retry control; the edge drops it from client requests. |
| `x-envoy-retry-on` | request, provider request | internal | none | reliability | v0.5 | Retry conditions the Router sets from the decision's `reliability` block. |
| `x-envoy-upstream-alt-stat-name` | request | internal | none | ingress | v0.5 | Envoy stats control; the edge drops it from client requests. |
| `x-envoy-upstream-rq-per-try-timeout-ms` | request, provider request | internal | none | reliability | v0.5 | Per-try timeout the Router sets from the decision's `reliability` block. |
| `x-envoy-upstream-rq-timeout-alt-response` | request | internal | none | ingress | v0.5 | Envoy timeout control; the edge drops it from client requests. |
| `x-envoy-upstream-rq-timeout-ms` | request, provider request | internal | none | reliability | v0.5 | Route timeout the Router sets from the decision's `reliability` block. |
| `x-forwarded-for` | request, provider request | internal | none | native_gateway | v0.5 | Client address chain; the native gateway's access log records it, and providers receive it. |
| `x-forwarded-proto` | request, provider request | internal | none | native_gateway | v0.5 | Scheme of the client connection; the native gateway sets it when absent, as Envoy does. |
| `x-goog-api-key` | provider request | internal | secret | credentials | v0.4 | Provider credential of Google APIs. |
| `x-user-anthropic-key` | request | internal | secret | authz | v0.2 | Per-user Anthropic key the authenticator injects. |
| `x-user-azure-openai-key` | request | internal | secret | authz | v0.2 | Per-user Azure OpenAI key the authenticator injects. |
| `x-user-bedrock-key` | request | internal | secret | authz | v0.2 | Per-user AWS Bedrock token the authenticator injects. |
| `x-user-cloudflare-workers-ai-key` | request | internal | secret | authz | v0.4 | Per-user Cloudflare Workers AI token the authenticator injects. |
| `x-user-gemini-key` | request | internal | secret | authz | v0.2 | Per-user Google Gemini key the authenticator injects. |
| `x-user-minimax-key` | request | internal | secret | authz | v0.3 | Per-user MiniMax key the authenticator injects. |
| `x-user-openai-key` | request | internal | secret | authz | v0.2 | Per-user OpenAI key the authenticator injects. |
| `x-user-vertex-ai-key` | request | internal | secret | authz | v0.2 | Per-user Vertex AI token the authenticator injects. |
| `x-vsr-delegated-credential` | request | internal | secret | authz | v0.5 | The requesting user's credential the authenticator attaches; only an upstream with `auth.credential: delegated` receives its value, under its own auth header, and `delegated_credential.header` names another header. |
| `x-vsr-internal-auth` | request | internal | secret | looper | v0.4 | In-process request authentication; the Router drops it from every client request. |
| `x-vsr-kv-cache-id` | provider request | internal | none | kv_transfer | v0.4 | Opaque id of the source KV block. |
| `x-vsr-kv-mapper-id` | provider request | internal | none | kv_transfer | v0.4 | Mapper artifact for the source and target model pair. |
| `x-vsr-kv-source-pod` | provider request | internal | none | kv_transfer | v0.4 | Address of the pod that holds the source model's KV cache. |
| `x-vsr-outcome-principal` | request | internal | identity | router_learning | v0.4 | Opaque attested identity that isolates outcome-ingest rate limits; never persisted. |
| `x-vsr-outcome-source` | request | internal | none | router_learning | v0.4 | Outcome provenance a trusted control plane attests to the management API. |
| `x-vsr-route-model` | request, provider request | internal | none | routing | v0.5 | Route key: the model whose route the gateway sends the provider request on; the Router drops a client's value and sets its own. |
| `x-vsr-systemone-backend` | request, provider request | internal | none | systemone | v0.5 | Marks a System One request one Router forwards to another, which refuses to forward it again. |
| `x-vsr-upstream-host` | provider response | internal | none | kv_transfer | v0.5 | Endpoint Envoy reports on the provider response for KV registry writes; removed before the client. |

The members of `x-vsr-trace`, highest priority first:

| Member | Owner | Description |
| --- | --- | --- |
| `config_hash` | benchmark | SHA-256 of the runtime document that served a config-bound evaluation request. |
| `inference_calls` | benchmark | Provider calls a config-bound evaluation request made. |
| `model_usage` | benchmark | Per-model usage receipt of a config-bound evaluation request, as JSON. |
| `confidence` | routing | Model-derived score of the selected decision, such as `0.91`. |
| `algorithm` | routing | Selector or executor that chose the model after the decision matched, such as `static`. |
| `routing_latency_ms` | routing | Time spent choosing the model, in milliseconds. |
| `fallback_attempts` | fallback | Attempts cross-model fallback made, the primary included, when a fallback candidate served the response. |
| `on_unknown` | routing | Decisions whose unknown result `rules.on_unknown` resolved, each with its `policy`. |
| `client_protocol` | protocol | Client wire protocol, such as `anthropic`. |
| `upstream_protocol` | protocol | Provider wire protocol, such as `openai`. |
| `effective_input_tokens` | automatic_output | Rendered input tokens of the final automatic-output dispatch, chat template included. |
| `effective_max_output_tokens` | automatic_output | Output token limit of the final automatic-output dispatch, reasoning included. |
| `signals` | signals | Matched signal rules as `type:name`, and `type:rule:option` for a choice rule without labels; projections are `projection:name`. |
| `context_tokens` | signals | Token count the `context` signal measured. |
| `category` | classification | Domain category the classifier chose, such as `math`. |
| `reasoning` | classification | Reasoning mode chosen for the request: `on` or `off`. |
| `modality` | routing | Output kinds of the selected decision's output contract, such as `(image)` or `(text image)`; omitted for a decision that declares none. |
| `system_prompt` | system_prompt | Present when a `system_prompt` plugin injected text into the request. |
| `cache_similarity` | response_cache | Similarity score of the response-cache lookup. |
| `retention` | retention | The matched decision's retention directive: the `drop`, `ttl_turns`, `keep_current_model` and `prefer_prefix` it sets. |
| `prompt_cache` | prompt_cache | Outcome of the `prompt_cache` plugin, with its `reason` and the `inserted` and `preserved` marker counts. |
| `tools` | tools | Outcome of the `tools` plugin's selection, with its `candidates`, `kept` and `latency_ms`. |
| `response_guard` | response_guard | Outcome of the selected decision's `response_guard`, such as `blocked`, with `buffered` when the Router held a streamed answer until the guard read it. |
| `session_phase` | router_learning | Protection phase: `user_turn`, `tool_loop` or `provider_state`. |
| `learning` | router_learning | Router Learning methods, each with its `action`, `scope` and `reason`. |
| `looper` | looper | Models the Looper called, with its `algorithm`, final `model`, `iterations`, `latency_ms` and token counts. |
| `ranking` | routing | The matched decision the selected one beat, with the `reason` that settled it. |
| `truncated` | response_contract | Present when members were left out to keep the trace within 4 KB. |

<!-- END-GENERATED-HEADER-REFERENCE -->

# VSR routing headers

The router uses these request and response headers for session continuity,
routing observability, replay correlation, and opt-in debugging. The
[header reference](#header-reference) at the end of this page lists every
header the Router reads or writes.

## What appears by default

The router splits response headers across two surfaces:

- **Public surface** — every routed inference response that is not served from
  the response cache includes `x-vsr-schema-version` and `x-vsr-response-path`.
  These routing keystone headers are produced by the routing pipeline, so
  responses answered by the Router's own handlers instead, such as
  `GET /v1/models`, omit them. The one management endpoint that does set an
  `x-vsr-*` header is `POST /api/v1/routing/preview`, which returns
  `x-vsr-config-hash` on a successful result. Successful routed responses can
  also include the final recipe, decision, confidence, algorithm, model, routing
  latency, cost, and replay id — the last only when Router Replay is enabled and
  the record was persisted. Protocol markers appear when translation occurs;
  protocol warnings appear only when there are warnings.
- **Configuration version** — `x-vsr-config-version` names the version of the
  configuration that served the request. It rides on every routed response
  that is not served from the response cache, success or error, and on every
  response the Router answers itself, such as a block, a validation error, a
  cache hit or `GET /v1/models`. A request keeps the version it started on,
  even when a hot reload activates another one while it runs.
- **Debug surface** — intermediate classification details, matched signals,
  tool-selection metrics, and `x-vsr-retention-*` directives appear inline
  only when the request sets `x-vsr-debug: true` and
  [`global.router.headers.debug`](#who-may-turn-on-the-debug-headers) allows
  the caller. It allows nobody by default. When replay is enabled, the same
  diagnostic context remains available through `x-vsr-replay-id`.

Decision and matched-signal headers additionally require all of the following:

1. The upstream response is successful (`2xx`).
2. The response was not served from the response cache.
3. The router evaluated a routing decision or signal for the request.

Cache-hit responses can emit cache headers, but they do not re-run routing and therefore do not attach fresh matched-signal headers.

Clients steer a request with the `control` headers in the reference, such as
`x-session-id`, `x-disable-router-memory` or `x-vsr-debug`.

A header reaches a provider only when its direction in the reference lists
`provider request`: the Router removes its own control headers, its internal
headers, identity and secrets from every provider-bound request.

## Who may turn on the debug headers

The debug headers name the signal and guard rules a request matched, the
decision ranking and the Router Learning reasons, so a client can't turn them
on by itself. `global.router.headers.debug` decides who may; the Router ignores
`x-vsr-debug` from everyone else.

| `debug` | Who gets the debug headers |
| ------- | -------------------------- |
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
`vllm-sr route --debug` and sr-bench ask for the debug headers; when the caller
isn't allowed, they show the default surface only.

## Response warnings

`x-vsr-response-warnings` lists the response-quality warning codes of the
completion in a fixed order and is omitted when none applies. Per-warning
detail, such as hallucination spans or jailbreak confidence, is kept in the
replay record instead of being expanded into response headers.

## Decision headers

Final routing facts use the public surface. Intermediate details, including
Router Learning observability, require `x-vsr-debug`.

- `x-vsr-selected-decision` is omitted when no decision matched and the request
  went to the default model; `x-vsr-response-path` is still `upstream`, and
  `x-vsr-selected-model` names the default model.
- `x-vsr-selected-confidence` is absent when the decision rests on policy
  leaves, aggregates several evidence leaves, or resolves through an error
  policy.
- `x-vsr-fallback-attempts` counts the same way in both gateway modes and is
  omitted when the primary answered.
- `x-vsr-decision-ranking` is prose; parse `decision_ranking` in Router Replay
  instead. It is omitted when only one decision matched.

For UI display guidance, translate `x-vsr-learning-actions` into user-facing
phrases such as `tool/protocol pinned`, `model switched`, or `learning bypassed`.
Fresh conversation or session-start diagnostics are usually useful only in debug
views, where they should be shown as neutral status text rather than a primary
route state.

The two effective-token headers are emitted together on successful upstream
responses only when automatic output was resolved. They are available in the
initial headers for both streaming and buffered responses. Explicit output
limits, cache hits, skipped processing, and Looper responses omit them. Their
values come from the final dispatch after request changes and backend rendering;
they are not estimates or the model's configured maximum context length.

## Matched signal headers

Matched signal headers contain comma-separated rule names. They require
`x-vsr-debug` and are omitted when that signal family did not match. A rule
asked of a decision model (an application signal or a `systemone` rule) is
listed by name when it matched. A choice rule without labels, which conditions
name by label, is listed as `<rule>:<option>` with the option its answer chose,
such as `task_difficulty:hard`. An option that did not make its rule match is
never listed.
`x-vsr-matched-projections` lists the projection mapping outputs that matched
the request.

Projection scores and full projection traces are stored in router replay records rather than expanded into response headers. Use `x-vsr-replay-id` to inspect those details in the Dashboard or through the authenticated Router management API; public inference listeners do not serve replay records.

## Retention headers

When a matched decision emits a retention directive, debug responses expose
the fields that were set. These headers help operators verify policy wiring;
clients should not use them as commands.

Unset fields are omitted. Cache hits do not emit these headers because no
decision was evaluated for that response.

## Cross-model KV transfer headers (issue #2976)

When the router attempts cross-model KV reuse on a model switch, it injects
request-side hints on the upstream call: `x-vsr-kv-source-pod`,
`x-vsr-kv-cache-id` and `x-vsr-kv-mapper-id`. The target vLLM KVConnector
plugin reports the outcome on the response in `x-vsr-kv-transfer-status`; an
absent status means `unsupported`. These headers are internal to the router
and inference pool; clients should not depend on them.

## Cost headers

On a buffered (non-streaming) response, the router prices the usage the model
reported with the served model's `pricing` configuration. This is a
configured-price figure, not a provider bill. Streamed responses and models
without `pricing` omit both headers.

## Cache and plugin headers

`x-vsr-cache-hit` and `x-vsr-fast-response` identify an immediate response on
the public surface. Cache-similarity, tool-selection metrics, and prompt-cache
receipts require `x-vsr-debug`.

The prompt-cache headers are content-free: they report counts and outcomes,
never the cached text itself, and they never claim a cache hit or
provider-side savings.

## Example response

Public surface — keystone headers, final routing facts and the replay-id entry point:

```http
HTTP/1.1 200 OK
Content-Type: application/json
x-vsr-schema-version: 2
x-vsr-response-path: upstream
x-vsr-config-version: 7
x-vsr-selected-recipe: default
x-vsr-selected-decision: complex-request
x-vsr-selected-algorithm: static
x-vsr-selected-model: reasoning-model
x-vsr-replay-id: replay_01J...
```

`x-vsr-replay-id` appears here only because Router Replay was enabled and the
record was persisted. With replay disabled, or when persistence fails, the
response carries the other headers and omits this one.

With `x-vsr-debug: true` on the request from a caller that `global.router.headers.debug` allows, the demoted intermediate details and matched signals are emitted inline as well:

```http
HTTP/1.1 200 OK
Content-Type: application/json
x-vsr-schema-version: 2
x-vsr-response-path: upstream
x-vsr-selected-recipe: default
x-vsr-selected-decision: complex-request
x-vsr-selected-algorithm: static
x-vsr-selected-model: reasoning-model
x-vsr-session-phase: tool_loop
x-vsr-matched-context: long-context
x-vsr-matched-projections: use-reasoning-model
x-vsr-replay-id: replay_01J...
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

- Use `x-vsr-schema-version` before parsing optional headers; the current value
  is `2`. Routed inference responses not served from the response cache carry it. Responses answered by the
  Router's own handlers, such as `GET /v1/models`, never carry this header, so
  its absence there is expected rather than a contract violation.
- `x-vsr-matched-projections` is the projection header. The singular form is
  not part of the public contract.
- `x-vsr-config-version` counts the activations of one Router, which keeps its
  own configuration history. To check that replicas serve the same
  configuration, compare the document hash that
  `GET /api/v1/config/hash` reports as `active_runtime_hash`.
- Recipe names scope local signal, projection, decision, cache, replay, metric, and learning/session identities. Use `x-vsr-selected-recipe` together with the local decision/signal names when correlating a response with Insights or metrics.
- `event` is the public signal type used by decisions and DSL. Canonical YAML stores event rules under `routing.signals.events`, matching other plural signal containers.
- Router Learning uses router-owned online state internally. Users enable online model-choice learning through `global.router.learning.adaptation`, enable stability protection through `global.router.learning.protection`, pass stable identity headers, and optionally set `routing.decisions[].adaptations.mode`, component modes, or `adaptations.adaptation.candidate_set`. `scope: conversation` protects one `x-conversation-id`; `scope: session` protects the broader `x-session-id`. The old `routing.decisions[].algorithm.session_aware` shape is not part of the public contract.

## Header reference

The Router's header registry, `src/semantic-router/pkg/headers/registry.go`,
generates this table; run `make headers-generate` after changing it.

- **Direction** — `request` (client to Router), `response` (Router to client),
  `provider request` (Router to provider) and `provider response` (provider to
  Router).
- **Surface** — `public` rides on the default response, `debug` only for a
  caller allowed the debug headers, and `control` headers are sent by clients
  to steer a request. `internal` headers are not part of the client contract,
  and no response the Router writes carries them.
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
| `x-vsr-applied-unknown-policy` | response | public | none | routing | v0.4 | Decisions whose unknown result `rules.on_unknown` resolved, as `decision=policy` pairs; also on the `fail_request` 503. |
| `x-vsr-cache-hit` | response | public | none | response_cache | v0.1 | `true` when the response cache served the response. |
| `x-vsr-client-protocol` | response | public | none | protocol | v0.4 | Client wire protocol, such as `anthropic`; on cross-protocol responses and with the debug headers. |
| `x-vsr-config-hash` | response | public | none | config | v0.4 | SHA-256 of the active runtime document, on routing previews, configuration activations and config-bound evaluation requests. |
| `x-vsr-config-version` | response | public | none | config | v0.5 | Version of the configuration snapshot that served the request. |
| `x-vsr-cost` | response | public | none | cost | v0.4 | Usage priced with the served model's configured pricing; buffered responses only. |
| `x-vsr-cost-currency` | response | public | none | cost | v0.4 | Currency of `x-vsr-cost`, from `pricing.currency`. |
| `x-vsr-effective-input-tokens` | response | public | none | automatic_output | v0.4 | Rendered input tokens of the final automatic-output dispatch, chat template included. |
| `x-vsr-effective-max-output-tokens` | response | public | none | automatic_output | v0.4 | Output token limit of the final automatic-output dispatch, reasoning included. |
| `x-vsr-fallback-attempts` | response | public | none | fallback | v0.4 | Attempts cross-model fallback made, the primary included, when a fallback candidate served the response. |
| `x-vsr-fast-response` | response | public | none | fast_response | v0.2 | `true` when the `fast_response` plugin answered without a provider call. |
| `x-vsr-inference-call-count` | response | public | none | benchmark | v0.4 | Provider calls a config-bound evaluation request made. |
| `x-vsr-model-usage` | response | public | none | benchmark | v0.4 | Per-model usage receipt of a config-bound evaluation request. |
| `x-vsr-protocol-warnings` | response | public | none | protocol | v0.4 | Protocol translation warnings, as comma-separated `severity;reason;field` entries. |
| `x-vsr-replay-id` | response | public | none | router_replay | v0.2 | Router Replay record id, when the record was persisted. |
| `x-vsr-response-path` | response | public | none | response_contract | v0.4 | How the response was produced: `upstream`, `cache`, `fast_response`, `looper`, `image_generation`, `blocked`, `rate_limited`, `error` or `fallback`. |
| `x-vsr-response-warnings` | response | public | none | response_guard | v0.4 | Codes of the `response_guard` rules that annotated the answer, in rule order: `hallucination`, `unverified_factual`, `response_jailbreak` or `response_guard`. |
| `x-vsr-routing-latency-ms` | response | public | none | routing | v0.4 | Time spent choosing the model, in milliseconds, such as `0.412`. |
| `x-vsr-schema-version` | response | public | none | response_contract | v0.4 | Response-header contract revision; currently `2`. |
| `x-vsr-selected-algorithm` | response | public | none | routing | v0.4 | Model-selection algorithm used after the decision matched, such as `static`. |
| `x-vsr-selected-confidence` | response | public | none | routing | v0.2 | Model-derived score of the selected decision, such as `0.9100`. |
| `x-vsr-selected-decision` | response | public | none | routing | v0.1 | Decision the decision engine selected; omitted when no decision matched. |
| `x-vsr-selected-model` | response | public | none | routing | v0.1 | Model alias the Router selected. |
| `x-vsr-selected-recipe` | response | public | none | routing | v0.4 | Recipe the effective entrypoint selected; omitted for concrete backend models. |
| `x-vsr-upstream-protocol` | response | public | none | protocol | v0.4 | Provider wire protocol; on cross-protocol responses and with the debug headers. |
| `x-vsr-cache-similarity` | response | debug | none | response_cache | v0.3 | Similarity score of the response-cache lookup. |
| `x-vsr-context-token-count` | response | debug | none | signals | v0.2 | Token count the `context` signal measured. |
| `x-vsr-decision-ranking` | response | debug | none | routing | v0.5 | The matched decision the selected one beat, and the comparison that settled it. |
| `x-vsr-injected-system-prompt` | response | debug | none | system_prompt | v0.1 | Whether a `system_prompt` plugin injected text into the request. |
| `x-vsr-learning-actions` | response | debug | none | router_learning | v0.4 | Learning action per method, such as `adaptation=propose_switch`. |
| `x-vsr-learning-methods` | response | debug | none | router_learning | v0.4 | Router Learning methods the learning headers summarize, such as `adaptation,protection`. |
| `x-vsr-learning-reasons` | response | debug | none | router_learning | v0.4 | Reason per method, such as `adaptation=sampled_win`. |
| `x-vsr-learning-scopes` | response | debug | none | router_learning | v0.4 | Identity scope per method, such as `protection=conversation`. |
| `x-vsr-looper-algorithm` | response | debug | none | looper | v0.2 | Looper algorithm, such as `confidence`. |
| `x-vsr-looper-completion-tokens` | response | debug | none | looper | v0.4 | Completion tokens across the Looper's calls. |
| `x-vsr-looper-iterations` | response | debug | none | looper | v0.2 | Model calls the Looper made. |
| `x-vsr-looper-latency-ms` | response | debug | none | looper | v0.4 | Looper wall-clock latency, in milliseconds. |
| `x-vsr-looper-model` | response | debug | none | looper | v0.2 | Final model of the Looper. |
| `x-vsr-looper-models-used` | response | debug | none | looper | v0.2 | Models the Looper called, comma-separated. |
| `x-vsr-looper-prompt-tokens` | response | debug | none | looper | v0.4 | Prompt tokens across the Looper's calls. |
| `x-vsr-looper-total-tokens` | response | debug | none | looper | v0.4 | Total tokens across the Looper's calls. |
| `x-vsr-matched-action` | response | debug | none | signals | v0.5 | Matched `action` rule. |
| `x-vsr-matched-authz` | response | debug | none | signals | v0.2 | Matched `authz` rules. |
| `x-vsr-matched-complexity` | response | debug | none | signals | v0.2 | Matched `complexity` rules, and `rule:option` for a rule without labels, such as `task_difficulty:hard`. |
| `x-vsr-matched-context` | response | debug | none | signals | v0.2 | Matched `context` rules. |
| `x-vsr-matched-conversation` | response | debug | none | signals | v0.3 | Matched `conversation` rules. |
| `x-vsr-matched-decision-model` | response | debug | none | signals | v0.5 | Matched `systemone` rules, and `rule:option` for a choice rule without labels. |
| `x-vsr-matched-domains` | response | debug | none | signals | v0.1 | Matched `domain` rules. |
| `x-vsr-matched-event` | response | debug | none | signals | v0.3 | Matched `event` rules. |
| `x-vsr-matched-fact-check` | response | debug | none | signals | v0.1 | Matched `fact_check` rules. |
| `x-vsr-matched-hallucination` | response | debug | none | signals | v0.4 | Matched `hallucination` rules, once the answer is checked. |
| `x-vsr-matched-input-modality` | response | debug | none | signals | v0.4 | Matched `input_modality` rules. |
| `x-vsr-matched-jailbreak` | response | debug | none | signals | v0.2 | Matched `jailbreak` rules, of the request and of the response. |
| `x-vsr-matched-keywords` | response | debug | none | signals | v0.1 | Matched `keyword` rules, comma-separated. |
| `x-vsr-matched-language` | response | debug | none | signals | v0.2 | Matched `language` rules. |
| `x-vsr-matched-modality` | response | debug | none | signals | v0.3 | Matched `modality` rules, and `rule:option` for a rule without labels. |
| `x-vsr-matched-pii` | response | debug | none | signals | v0.2 | Matched `pii` rules. |
| `x-vsr-matched-projections` | response | debug | none | signals | v0.3 | Matched projection outputs. |
| `x-vsr-matched-safety` | response | debug | none | signals | v0.4 | Matched content `safety` rules. |
| `x-vsr-matched-structure` | response | debug | none | signals | v0.3 | Matched `structure` rules. |
| `x-vsr-matched-turn-relation` | response | debug | none | signals | v0.5 | Matched `turn_relation` rules, and `rule:option` for a rule without labels. |
| `x-vsr-matched-user-feedback` | response | debug | none | signals | v0.1 | Matched `user_feedback` rules, and `rule:option` for a rule without labels. |
| `x-vsr-prompt-cache-action` | response | debug | none | prompt_cache | v0.4 | Outcome of the `prompt_cache` plugin: `inserted`, `preserved`, `skipped` or `rejected`. |
| `x-vsr-prompt-cache-inserted` | response | debug | none | prompt_cache | v0.4 | Cache markers the Router inserted, at most `2`. |
| `x-vsr-prompt-cache-preserved` | response | debug | none | prompt_cache | v0.4 | Cache markers the caller supplied. |
| `x-vsr-prompt-cache-reason` | response | debug | none | prompt_cache | v0.4 | Reason for an outcome other than `inserted`: `caller_markers`, `no_eligible_target` or `unsupported_target`. |
| `x-vsr-retention-drop` | response | debug | none | retention | v0.4 | Whether the matched decision's retention directive keeps the response out of the response cache. |
| `x-vsr-retention-keep-current-model` | response | debug | none | retention | v0.4 | Whether the directive asks later routing to keep the current model. |
| `x-vsr-retention-prefer-prefix` | response | debug | none | retention | v0.4 | Whether the directive prefers prefix retention. |
| `x-vsr-retention-ttl-turns` | response | debug | none | retention | v0.4 | Retention lifetime the directive sets, in conversation turns. |
| `x-vsr-selected-category` | response | debug | none | classification | v0.1 | Domain category the classifier chose, such as `math`. |
| `x-vsr-selected-modality` | response | debug | none | classification | v0.2 | Modality result and optional method, such as `AR;classifier`. |
| `x-vsr-selected-reasoning` | response | debug | none | classification | v0.1 | Reasoning mode chosen for the request: `on` or `off`. |
| `x-vsr-session-phase` | response | debug | none | router_learning | v0.3 | Protection phase: `user_turn`, `tool_loop` or `provider_state`. |
| `x-vsr-tools-confidence` | response | debug | none | tools | v0.3 | Highest similarity score among the selected tools. |
| `x-vsr-tools-latency-ms` | response | debug | none | tools | v0.3 | Tool-selection retriever latency, in milliseconds. |
| `x-vsr-tools-strategy` | response | debug | none | tools | v0.3 | Retriever strategy of semantic tool selection. |
| `x-claude-code-session-id` | request, provider request | control | none | session | v0.3 | Conversation id Claude Code sends; `x-session-id` takes precedence, and providers receive it. |
| `x-conversation-id` | request, provider request | control | none | router_learning | v0.4 | Client conversation id; Router Learning `scope: conversation` protects it, and providers receive it. |
| `x-disable-router-memory` | request | control | none | memory | v0.3 | `true` keeps Router-managed memory out of the request. |
| `x-session-id` | request, provider request | control | none | session | v0.3 | Client session id; Router Learning `scope: session` protects it, and providers receive it. |
| `x-sr-bench-expected-config-hash` | request | control | none | benchmark | v0.4 | Configuration hash an evaluation request requires; another active configuration fails it with 412. |
| `x-sr-bench-max-inference-calls` | request | control | none | benchmark | v0.4 | Provider-call limit of an evaluation request; only `1` is supported. |
| `x-vsr-cache-control` | request | control | none | response_cache | v0.4 | Response-cache directives for the request, when the plugin's `request_controls` allow them. |
| `x-vsr-compression-control` | request | control | none | context_compression | v0.4 | Context-compression directives for the request, when the plugin's `request_controls` allow them. |
| `x-vsr-debug` | request | control | none | debug_headers | v0.4 | `true` asks for the debug headers on the response; honored only for a caller `global.router.headers.debug` allows. |
| `x-vsr-protection-release` | request | control | none | router_learning | v0.5 | Any non-empty value asks router-learning protection for a fresh decision on the request's protection key; `global.router.learning.protection.release.header` can rename it. |
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
| `x-selected-model` | request, provider request | internal | none | routing | v0.1 | Route key: the model whose route the gateway sends the provider request on; the Router replaces or drops a client's value. |
| `x-user-anthropic-key` | request | internal | secret | authz | v0.2 | Per-user Anthropic key the authenticator injects. |
| `x-user-azure-openai-key` | request | internal | secret | authz | v0.2 | Per-user Azure OpenAI key the authenticator injects. |
| `x-user-bedrock-key` | request | internal | secret | authz | v0.2 | Per-user AWS Bedrock token the authenticator injects. |
| `x-user-cloudflare-workers-ai-key` | request | internal | secret | authz | v0.4 | Per-user Cloudflare Workers AI token the authenticator injects. |
| `x-user-gemini-key` | request | internal | secret | authz | v0.2 | Per-user Google Gemini key the authenticator injects. |
| `x-user-minimax-key` | request | internal | secret | authz | v0.3 | Per-user MiniMax key the authenticator injects. |
| `x-user-openai-key` | request | internal | secret | authz | v0.2 | Per-user OpenAI key the authenticator injects. |
| `x-user-vertex-ai-key` | request | internal | secret | authz | v0.2 | Per-user Vertex AI token the authenticator injects. |
| `x-vsr-fusion-depth` | request | internal | none | looper | v0.4 | Retired Looper hop header; the Router drops it from every request. |
| `x-vsr-gateway-credential` | request | internal | secret | authz | v0.5 | Dedicated credential the authenticator attaches for a trusted upstream gateway; only a backend marked `trusted_gateway` receives its value, under its own auth header, and `gateway_credential.header` names another header. |
| `x-vsr-internal-auth` | request | internal | secret | looper | v0.4 | In-process request authentication; the Router drops it from every request. |
| `x-vsr-kv-cache-id` | provider request | internal | none | kv_transfer | v0.4 | Opaque id of the source KV block. |
| `x-vsr-kv-mapper-id` | provider request | internal | none | kv_transfer | v0.4 | Mapper artifact for the source and target model pair. |
| `x-vsr-kv-source-pod` | provider request | internal | none | kv_transfer | v0.4 | Address of the pod that holds the source model's KV cache. |
| `x-vsr-kv-transfer-status` | provider response | internal | none | kv_transfer | v0.4 | KV reuse outcome the target pod reports: `applied`, `fallback_reprefill` or `unsupported`. |
| `x-vsr-looper-decision` | request | internal | none | looper | v0.2 | Retired Looper hop header; the Router drops it from every request. |
| `x-vsr-looper-iteration` | request | internal | none | looper | v0.2 | Retired Looper hop header; the Router drops it from every request. |
| `x-vsr-looper-request` | request | internal | none | looper | v0.2 | Retired Looper hop header; the Router drops it from every request. |
| `x-vsr-looper-secret` | request | internal | secret | looper | v0.3 | Retired Looper hop secret; the Router drops it from every request. |
| `x-vsr-outcome-principal` | request | internal | identity | router_learning | v0.4 | Opaque attested identity that isolates outcome-ingest rate limits; never persisted. |
| `x-vsr-outcome-source` | request | internal | none | router_learning | v0.4 | Outcome provenance a trusted control plane attests to the management API. |
| `x-vsr-systemone-backend` | request, provider request | internal | none | systemone | v0.5 | Marks a System One request one Router forwards to another, which refuses to forward it again. |
| `x-vsr-upstream-host` | provider response | internal | none | kv_transfer | v0.5 | Endpoint Envoy reports on the provider response for KV registry writes; removed before the client. |

<!-- END-GENERATED-HEADER-REFERENCE -->

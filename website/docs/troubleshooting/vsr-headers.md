# VSR Routing Headers

This page documents the public `x-vsr-*` headers emitted by vLLM Semantic Router for routing observability, replay correlation, and debugging.

## Emission Rules

The router emits the keystone headers (`x-vsr-schema-version`, `x-vsr-response-path`) on every non-cache-hit response. The client/upstream protocol markers ride only on cross-protocol handling or when `x-vsr-debug` is set, and `x-vsr-protocol-warnings` only when warnings exist. It emits routing decision and matched-signal headers only when all of the following are true:

1. The upstream response is successful (`2xx`).
2. The response was not served from semantic cache.
3. The router evaluated a routing decision or signal for the request.

Cache-hit responses can emit cache headers, but they do not re-run routing and therefore do not attach fresh matched-signal headers.

## Request Headers

| Header | Direction | Description |
| ------ | --------- | ----------- |
| `x-session-id` | request | Stable client-provided session identifier for Chat Completions. `session_aware` uses this to reason about stay-vs-switch decisions across turns. |
| `x-vsr-skip-processing` | request | Opts a request out of router processing when `global.router.skip_processing.enabled` is enabled. Use value `true`. |
| `x-vsr-debug` | request | Opts the request into verbose/debug response headers — headers the contract otherwise omits or demotes to replay are emitted inline for that request. Use value `true`. |

## Protocol And Replay Headers

| Header | Description |
| ------ | ----------- |
| `x-vsr-client-protocol` | Inbound protocol shape seen by the router, for example `openai` or `anthropic`. Emitted only on cross-protocol handling (client protocol differs from upstream), or when `x-vsr-debug` is set. |
| `x-vsr-upstream-protocol` | Protocol shape sent to the selected upstream backend. Emitted only on cross-protocol handling, or when `x-vsr-debug` is set. |
| `x-vsr-protocol-warnings` | Comma-separated protocol translation warnings encoded as `severity;reason;field`. Emitted only when warnings exist. |
| `x-vsr-replay-id` | Opaque router replay record identifier for correlating a response with replay/Insights data. |

## Response Warnings

| Header | Description |
| ------ | ----------- |
| `x-vsr-response-warnings` | Comma-separated response-quality warning codes for the completion, in fixed order: `hallucination`, `unverified_factual`, `response_jailbreak`. Emitted only when at least one applies. |

This header consolidates the v0.3 `x-vsr-hallucination-detected`, `x-vsr-unverified-factual-response`, and `x-vsr-response-jailbreak-detected` headers. The per-warning detail (hallucination spans, jailbreak type/confidence, fact-check verification context) is kept in the replay record rather than the response headers, recoverable via `x-vsr-replay-id`; the individual warning and detail headers are removed from the v0.4 contract.

## Decision Headers

| Header | Description | Example |
| ------ | ----------- | ------- |
| `x-vsr-selected-category` | Domain/category classifier result when domain routing runs. | `math` |
| `x-vsr-selected-decision` | Final decision selected by the decision engine. | `formal_math_proof` |
| `x-vsr-selected-confidence` | Confidence score for the selected decision. | `0.9100` |
| `x-vsr-selected-reasoning` | Reasoning mode selected for the request. | `on` |
| `x-vsr-selected-modality` | Modality result and optional method. | `AR;classifier` |
| `x-vsr-selected-model` | Logical model alias selected by the router. | `qwen/qwen3.5-rocm` |
| `x-vsr-session-phase` | Session-aware phase from the selected policy trace. | `user_turn`, `tool_loop`, `provider_state` |
| `x-vsr-injected-system-prompt` | Whether a system-prompt plugin injected text into the request. | `true` |

## Matched Signal Headers

Matched signal headers contain comma-separated rule names. They are omitted when the corresponding signal family did not match.

| Header | Signal family |
| ------ | ------------- |
| `x-vsr-matched-keywords` | `keyword` |
| `x-vsr-matched-embeddings` | `embedding` |
| `x-vsr-matched-domains` | `domain` |
| `x-vsr-matched-fact-check` | `fact_check` |
| `x-vsr-matched-user-feedback` | `user_feedback` |
| `x-vsr-matched-reask` | `reask` |
| `x-vsr-matched-preference` | `preference` |
| `x-vsr-matched-language` | `language` |
| `x-vsr-matched-context` | `context` |
| `x-vsr-context-token-count` | Context token estimate used by `context` |
| `x-vsr-matched-structure` | `structure` |
| `x-vsr-matched-complexity` | `complexity` |
| `x-vsr-matched-modality` | `modality` |
| `x-vsr-matched-authz` | `authz` |
| `x-vsr-matched-jailbreak` | `jailbreak` |
| `x-vsr-matched-pii` | `pii` |
| `x-vsr-matched-kb` | `kb` |
| `x-vsr-matched-conversation` | `conversation` |
| `x-vsr-matched-event` | `event` |

## Projection Headers

| Header | Description |
| ------ | ----------- |
| `x-vsr-matched-projections` | Comma-separated projection mapping outputs that matched the request. |

Projection scores and full projection traces are stored in router replay records rather than expanded into response headers. Use `x-vsr-replay-id` to inspect those details in the dashboard or replay APIs.

## Cache And Plugin Headers

| Header | Description |
| ------ | ----------- |
| `x-vsr-cache-hit` | Response came from semantic cache. |
| `x-vsr-cache-similarity` | Similarity score from the semantic-cache lookup. |
| `x-vsr-fast-response` | Response was generated by the `fast_response` plugin without an upstream model call. |
| `x-vsr-tools-strategy` | Semantic tool-selection retriever strategy used for the request. |
| `x-vsr-tools-confidence` | Highest tool-selection retriever similarity score. |
| `x-vsr-tools-latency-ms` | Tool-selection retriever latency in milliseconds. |

## Example Response

```http
HTTP/1.1 200 OK
Content-Type: application/json
x-vsr-client-protocol: anthropic
x-vsr-upstream-protocol: openai
x-vsr-selected-decision: critical_event_tool_session_route
x-vsr-selected-confidence: 1.0000
x-vsr-selected-model: anthropic/claude-opus-4.6
x-vsr-session-phase: tool_loop
x-vsr-matched-conversation: active_tool_use
x-vsr-matched-event: critical_payment_event
x-vsr-matched-projections: verification_required
x-vsr-replay-id: replay_01J...
```

## Notes

- `x-vsr-matched-projections` is the v0.3 projection header. The old singular form is not part of the public v0.3 contract.
- `event` is the public signal type used by decisions and DSL. Canonical YAML stores event rules under `routing.signals.events`, matching other plural signal containers.
- `session_aware` uses router-owned session state internally. Users configure it through `routing.decisions[].algorithm.session_aware` and pass stable session identity with `x-session-id`; there is no separate user-managed session-state block in the v0.3 public contract.

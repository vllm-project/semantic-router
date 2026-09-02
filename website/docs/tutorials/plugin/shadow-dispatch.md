# Shadow Dispatch

## Overview

`shadow_dispatch` is a route-local plugin that sends a bounded, sampled copy of the approved request to a secondary model and records the outcome without changing or delaying the primary response.

## Key Advantages

- Observes a candidate model on real traffic before it serves any user.
- Keeps the live response independent from shadow latency, failure, or output.
- Bounds sampling, concurrency, queue depth, timeout, response size, and retries explicitly.
- Records primary and shadow identity, timing, outcome, and a content hash on the replay record for audit and later comparison.

## What Problem Does It Solve?

Promoting a new model into a routing recipe needs evidence from production-shaped requests. Offline evaluation misses real prompt distributions, and gray release exposes users to an unproven model. Shadow dispatch fills the step in between: the primary model still answers every request, while a copy of the same finalized request is sent to the candidate in the background. The shadow result is stored as an immutable outcome on the request's replay record, so operators can correlate primary and shadow observations later without exposing protected content.

The shadow copy starts from the same approved neutral request the primary dispatch was built from, after every request plugin has run. It is then rendered for the shadow model through the same encode and provider-adaptation steps a primary dispatch uses, so reasoning controls follow the shadow model's family, not the primary's. It is always sent non-streaming. Decision header mutations and trace headers are applied; client headers are not. Only the router's own static credentials for the shadow model's backend are used, so a shadow can never leave the router's trust boundary.

## When to Use

- a candidate model should be evaluated on live traffic before gray release
- the primary response must stay identical whether the shadow succeeds, fails, or times out
- resource use for the observation must be explicit and bounded per route
- replay or audit tooling needs to join primary and shadow observations by request identity

Do not use it to influence the live request, to grade output quality, or to build a training set. The plugin records outcomes only.

## Configuration

Add the plugin to a decision alongside `router_replay` so outcomes have a record to attach to:

```yaml
plugins:
  - type: router_replay
    configuration:
      enabled: true
  - type: shadow_dispatch
    configuration:
      enabled: true
      model: candidate-model
      sample_rate: 0.05
      max_concurrency: 2
      max_queue_depth: 8
      timeout_seconds: 30
      max_response_bytes: 1048576
      max_retries: 0
      capture_response_body: false
      max_capture_bytes: 4096
      tls_skip_verify: false
```

| Field | Default | Meaning |
| --- | --- | --- |
| `enabled` | required | Turns the shadow on for this decision. |
| `model` | required when enabled | Configured logical model that receives the shadow copy. Must have a backend in `providers.models`. |
| `sample_rate` | `1.0` | Fraction of eligible requests to shadow, in `[0, 1]`. `0` keeps the plugin declared but never dispatches. |
| `max_concurrency` | `2` | In-flight shadow calls for this decision. |
| `max_queue_depth` | `8` | Calls waiting for a slot. Anything beyond is dropped with reason `queue_full`. |
| `timeout_seconds` | `30` | Deadline for queue wait plus execution, shared by all retries. |
| `max_response_bytes` | `1048576` | Largest shadow response body that is read. Larger bodies fail with `response_too_large`. |
| `max_retries` | `0` | Extra attempts on transport errors or retryable statuses. Capped at `3`. |
| `capture_response_body` | `false` | Store a bounded excerpt of the shadow text in the outcome. Off by default; only sizes, tokens, and a SHA-256 are kept. |
| `max_capture_bytes` | `4096` | Excerpt bound when capture is on. |
| `tls_skip_verify` | `false` | Skip certificate verification for an https shadow backend signed by an internal CA. The primary path reaches backends through Envoy, which does not verify upstream certificates. |

A shadow is skipped, with a metric but no outcome, when the request is sampled out or when the primary dispatch already selected the shadow model. Decisions that execute through the looper (ratings, confidence, fusion, ReMoM, workflows) reject the plugin at config load, because the shadow hook runs only on single-model provider dispatch.

### Fail-open behavior

The request path does one non-blocking slot check and returns. Everything else runs in a bounded worker after the primary dispatch response has been built. A slow, unavailable, malformed, or overloaded shadow endpoint cannot change the primary response or its latency. Every shadow ends in exactly one result:

| Result | Reasons |
| --- | --- |
| `completed` | `completed` |
| `failed` | `backend_unresolved`, `credential_unresolved`, `encode_failed`, `timeout`, `transport_error`, `upstream_status`, `response_too_large`, `malformed_response` |
| `dropped` | `queue_full`, `queue_timeout`, `router_closing`, `same_as_primary`, `internal_request`, `request_unavailable` |
| `sampled_out` | `sampled_out` |

Results and reasons are exported as `sr_shadow_dispatch_total{decision,result,reason}`, with `sr_shadow_dispatch_latency_seconds`, `sr_shadow_dispatch_inflight`, and `sr_shadow_dispatch_queued`. Drops caused by resource bounds are reported through metrics and a structured `shadow_dispatch_dropped` event rather than a replay write, so an overloaded shadow lane cannot amplify load on the replay store.

### Interpreting failures

A `failed` outcome does not always say something about the candidate model. Read the reason together with `status_code`, `attempts`, and the truncated `error` in the outcome metadata:

| Meaning | Reasons | How to read it |
| --- | --- | --- |
| Candidate rejected the request | `upstream_status` with a 4xx `status_code` | The shadow model could not accept the approved request, for example an unsupported parameter or a context window that is too small. Count it against the candidate. |
| Candidate health or capacity | `upstream_status` with a 5xx `status_code`, `timeout`, `transport_error` | The backend was unreachable, overloaded, or too slow within `timeout_seconds` after `max_retries`. This measures the deployment, not answer quality. |
| Candidate output problem | `malformed_response`, `response_too_large` | The backend answered but the body was not a valid response for its wire format or exceeded `max_response_bytes`. |
| Router-side, not about the candidate | `backend_unresolved`, `credential_unresolved`, `encode_failed` | The router could not build or address the shadow call. Fix the configuration and exclude these from any candidate comparison. |

`dropped` and `sampled_out` never reach the replay record. They mean the router chose not to send the shadow, so they carry no signal about the candidate model and are visible only in metrics and logs.

### Replay and audit capture

Completed and failed shadows append one outcome to the primary request's replay record with `source: shadow_dispatch`, `target: model`, `target_ref: <shadow model>`, the verdict, and the reason. The outcome metadata carries primary and shadow request identity, primary and shadow model and backend, decision and recipe, sample rate, enqueue, start, and finish timestamps, queue wait and latency, attempts, status code, response size, stop reason, token counts, and a SHA-256 of the shadow text. Outcomes are append-only.

Replay redaction applies to shadow outcomes the same way it applies to the rest of the record: viewers without content rights see the routing and timing fields but not `target_ref`, `reason`, or `metadata`. Keep `capture_response_body` off unless the replay store and its readers are cleared for prompt-level content. See the fragment:
[`config/fragments/plugin/shadow-dispatch/sampled.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/shadow-dispatch/sampled.yaml).

# Router Replay

## Overview

`router_replay` is a route-local plugin for overriding replay/debug capture on one route.

## Key Advantages

- Lets one route override the router-wide replay default.
- Supports request and response body controls.
- Makes storage limits explicit instead of hidden.

## What Problem Does It Solve?

Replay capture is useful, but some routes need different capture policy than the router-wide default. `router_replay` lets one route opt out or override request/response body capture limits without changing global replay storage settings.

## When to Use

- one route should override the router-wide replay policy
- capture limits should be explicit per route
- replay should be disabled for a specific route while staying on elsewhere

## Configuration

To disable replay for a route, add:

```yaml
plugins:
  - type: router_replay
    configuration:
      enabled: false
```

To customize capture for a route, add:

```yaml
plugins:
  - type: router_replay
    configuration:
      enabled: true
      max_records: 10000
      capture_request_body: true
      capture_response_body: true
      max_body_bytes: 4096
      max_tool_trace_steps: 100
```

## Looper diagnostics

Confidence Looper records include a versioned `route_diagnostics.looper`
object. It contains bounded attempt metadata, token and cost accounting,
latencies, disposition reason codes, the OpenTelemetry trace ID when tracing is
active, and `final_attempt_ordinal`. Ordinary records also include
`effective_max_output_tokens`, `effective_max_output_tokens_source`, and
`effective_max_output_tokens_fallback` on `route_diagnostics`. Looper attempts
repeat those three fields so Confidence escalation can show each model's
composed ceiling. Attempt details are omitted from viewer-redacted responses
and remain available to principals with replay-detail permission.

Looper diagnostics never contain prompts, responses, hidden reasoning, tool
arguments, endpoint URLs, credentials, or raw errors. Attempt count and encoded
size are capped; truncation is explicit and dropped token usage remains
accounted for.

Request bodies, response bodies, and tool traces can contain secrets or personal
data. Capture the minimum needed, set retention in the shared replay service,
and restrict replay read permissions. See a complete example:
[`config/fragments/plugin/router-replay/debug.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/router-replay/debug.yaml).

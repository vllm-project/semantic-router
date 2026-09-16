# Router Replay

## Overview

`router_replay` is a route-local plugin for overriding replay/debug capture on one route.

A recipe's `routing.data_policy.replay: false` takes precedence over global and
route-local replay settings. It prevents capture even for rejected requests;
`router_replay.enabled: true` cannot override it. All five built-in MoM recipes,
including Vault, enable PostgreSQL replay by default; operators can use this
policy to forbid capture for any recipe. See the
[Replay API and privacy controls](../../api/router#router-replay).

The default `memory` store loses records when configuration is reloaded or the
router restarts. To keep session history available while changing recipes,
configure a durable store such as Postgres or Redis in the
[shared replay service](../learning/memory-and-replay#configuration).

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

Captured body and structured-text limits count UTF-8 bytes. Truncation keeps
complete characters, so a captured excerpt can be slightly shorter than its
byte limit. Malformed UTF-8 sequences are replaced with `U+FFFD` before applying
the limit. Truncation flags indicate when the byte limit removed text; existing
flags remain set. Original raw tool arguments and outputs remain unchanged.

## Looper diagnostics

Confidence Looper records include a versioned `route_diagnostics.looper`
object. It contains bounded attempt metadata, token and cost accounting,
latencies, disposition reason codes, the OpenTelemetry trace ID when tracing is
active, and `final_attempt_ordinal`. Attempt details are omitted from
viewer-redacted responses and remain available to principals with replay-detail
permission.

Looper diagnostics never contain prompts, responses, hidden reasoning, tool
arguments, endpoint URLs, credentials, or raw errors. Attempt count and encoded
size are capped; truncation is explicit and dropped token usage remains
accounted for.

Request bodies, response bodies, and tool traces can contain secrets or personal
data. Capture the minimum needed, set retention in the shared replay service,
and restrict replay read permissions. See a complete example:
[`config/fragments/plugin/router-replay/debug.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/router-replay/debug.yaml).

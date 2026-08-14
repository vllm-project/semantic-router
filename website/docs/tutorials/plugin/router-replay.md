# Router Replay

## Overview

`router_replay` is a route-local plugin for overriding replay/debug capture on one route.

It aligns to `config/fragments/plugin/router-replay/debug.yaml`.

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

Use this fragment under `routing.decisions[].plugins`:

```yaml
plugins:
  - type: router_replay
    configuration:
      enabled: false
```

Use this fragment when one route needs custom capture settings:

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

Request bodies, response bodies, and tool traces can contain secrets or personal
data. Capture the minimum needed, set retention in the shared replay service,
and restrict replay read permissions. Maintained example:
[`config/fragments/plugin/router-replay/debug.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/router-replay/debug.yaml).

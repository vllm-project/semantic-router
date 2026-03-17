# Router Replay

## Overview

`router_replay` is a route-local plugin for capturing replay/debug artifacts.

It aligns to `config/plugin/router-replay/debug.yaml`.

## Key Advantages

- Keeps replay capture local to routes under active debugging or audit.
- Supports request and response body controls.
- Makes storage limits explicit instead of hidden.

## What Problem Does It Solve?

Replay capture is useful, but not every route should record the same volume of data. `router_replay` lets one route opt into replay behavior without globalizing the storage cost.

## When to Use

- one route is under debugging, audit, or controlled replay analysis
- capture limits should be explicit per route
- replay should be enabled for selected traffic only

## Configuration

Use this fragment under `routing.decisions[].plugins`:

```yaml
plugin:
  type: router_replay
  configuration:
    enabled: true
    max_records: 5000
    capture_request_body: true
    capture_response_body: false
    max_body_bytes: 65536
```

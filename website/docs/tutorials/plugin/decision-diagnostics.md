# Decision Diagnostics

## Overview

`decision_diagnostics` is an opt-in route-local plugin that emits a bounded,
content-free summary of the selected routing decision as ExtProc dynamic
metadata. Following Envoy filters can consume the summary during the same
request without calling a replay API or exposing diagnostics as public headers.

It aligns to `config/plugin/decision-diagnostics/bounded.yaml`.

## Key Advantages

- Keeps routing diagnostics request-local instead of requiring a replay lookup.
- Uses ExtProc dynamic metadata rather than public response headers.
- Enforces startup-validated cardinality, text, and payload-size bounds.

## What Problem Does It Solve?

Following filters sometimes need the facts behind the selected route while the
request is still in flight. Persisting full replay records or copying details
into headers adds storage, privacy, and lifecycle concerns. This plugin exposes
only a bounded summary to filters later in the same Envoy chain.

## When to Use

- a following Envoy filter needs selected-decision facts during the request
- access logging needs bounded signal and projection summaries
- Router Replay is disabled or intentionally independent from data-plane logs

## Safety and Bounds

The output contains decision, category, selected model, executed signal facts,
and projection scores. It never contains request or response bodies, full
messages, tool arguments, authorization headers, or credentials. The configured
limits are validated at startup and the runtime marks truncated payloads.

## Configuration

Use this fragment under `routing.decisions[].plugins`:

```yaml
plugin:
  type: decision_diagnostics
  configuration:
    enabled: true
    max_signals: 32
    max_projections: 16
    max_text_runes: 128
    max_payload_bytes: 16384
```

The defaults are the values above. Maximum accepted values are 128 signals, 64
projection scores, 512 runes per text value, and 65536 serialized bytes.

The plugin is independent of `router_replay`: enabling diagnostics does not
create a replay record or require a replay store.

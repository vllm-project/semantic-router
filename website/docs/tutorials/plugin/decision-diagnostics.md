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

## Dynamic Metadata Contract

The schema version is `1`. The ExtProc response writes untyped Envoy dynamic
metadata using native `google.protobuf.Struct` values at this exact location:

```text
ProcessingResponse.dynamic_metadata
  ["vllm.semantic_router"]                 Struct
    ["decision_diagnostics"]               Struct
```

`decision_diagnostics` is a structured value, not a JSON-encoded string.

Envoy accepts ExtProc response metadata only for explicitly configured
receiving namespaces. Add the namespace to the ext_proc filter before adding a
following filter:

```yaml
metadata_options:
  receiving_namespaces:
    untyped:
      - vllm.semantic_router
```

Without this allowlist Envoy discards the returned metadata. With it enabled, a
Lua HTTP filter placed after ext_proc can traverse the native structure
directly:

```lua
local namespace = handle:streamInfo():dynamicMetadata():get("vllm.semantic_router")
local diagnostics = namespace["decision_diagnostics"]
local decision = diagnostics["decision"]
local first_signal_name = diagnostics["signals"][1]["name"]
```

The schema uses these field names and logical types. Protobuf `Struct` carries
all logical `number` values as protobuf `double` values; `schemaVersion` is an
integer-valued number.

| Field | Type | Presence |
| --- | --- | --- |
| `schemaVersion` | number (`1`) | always |
| `decision` | string | always |
| `category` | string | omitted when empty |
| `selectedModel` | string | omitted when empty |
| `selectionAlgorithm` | string | omitted when empty |
| `selectionMethod` | string | omitted when empty |
| `decisionConfidence` | number | always |
| `matchedRules` | list of strings | always, possibly empty |
| `signals` | list of signal objects | always, possibly empty |
| `projections` | list of projection objects | always, possibly empty |
| `truncated` | boolean | always |

Each signal object contains `key`, `type`, and `name` strings plus the boolean
fields `executed` and `matched`. `value` and `confidence` are optional numbers
and are omitted when the evaluator did not produce them. `executed` is true
only when that signal type's evaluator actually started for this request.

Each projection object contains a `name` string and `matched` boolean. Its
numeric `score` is omitted when no projection score was produced.

The metadata is best-effort. It is absent when the plugin is disabled, no
decision was selected, serialization fails, or the bounded payload cannot fit.
For a selected decision it is attached to normal routing responses and
decision-owned immediate responses such as fast response, cache hit, rate
limit, and modality routing paths. It never changes the selected route or the
public response-header contract.

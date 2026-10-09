# Router Replay

## Overview

Use Router Replay to inspect requests in Dashboard Insights: the selected route,
model, token usage, response, and tool trajectory. Configure storage and capture
defaults in `global.services.router_replay`; use a decision's `router_replay`
plugin to override only the capture fields that differ.

## What Problem Does It Solve?

Replay keeps request-level evidence together so you can understand a routing
choice and follow a conversation across requests.

## When to Use

Enable Replay when you need request history for inspection or troubleshooting.
Choose retention and capture limits that match the data you are allowed to keep.

## Configuration

### Keep records across restarts

All five built-in MoM recipes, including Vault, enable PostgreSQL Replay by
default. Local `vllm-sr serve` manages PostgreSQL and its persistent volume.
For another deployment method, configure the PostgreSQL connection in the
[shared Replay service](../learning/memory-and-replay#configuration).

The generic `memory` store is useful for temporary inspection but loses records
when the Router restarts or reloads configuration. Use PostgreSQL or Redis when
you need durable history. Set retention for the data you intend to keep.

### Set shared capture defaults

```yaml
global:
  services:
    router_replay:
      enabled: true
      store_backend: postgres
      capture_request_body: true
      capture_response_body: true
      capture_personal_data: false
      max_records: 10000
      max_body_bytes: 4096
      max_tool_trace_bytes: 0
      max_tool_trace_steps: 100
```

Absent service configuration leaves Replay disabled. Capture defaults are
request and response bodies enabled, personal data enabled, 10,000 records,
4,096 bytes per body, no per-field tool-trace byte limit, and 100 tool steps.
Storage, retention and asynchronous writes are global service settings.

### Limit capture on a decision

Add this fragment to the decision's `plugins` to record bounded excerpts:

```yaml
plugins:
  - type: router_replay
    configuration:
      enabled: true
      capture_request_body: true
      capture_response_body: false
      capture_personal_data: false
      max_body_bytes: 4096
      max_tool_trace_steps: 100
```

Body limits and structured-text limits count UTF-8 bytes. Excerpts end on
complete characters, so they can be slightly shorter than the limit. A truncated
raw body may no longer be complete JSON; check its truncation flag. Tool traces
have separate limits from raw request and response bodies.

To opt one decision out of capture, use `configuration: {enabled: false}`.
To change the deployment-wide default, set
`global.services.router_replay.enabled: false`; a decision can still opt in.

Every omitted plugin field inherits its global capture default. Explicit
`false` remains an override, and `max_tool_trace_bytes: 0` or
`max_tool_trace_steps: 0` removes that limit. There is no recipe-level Replay
policy. Requests rejected before a decision is selected use the global defaults;
a decision override applies after that decision is selected. These settings do
not change provider-side retention or delete existing records.

The existing recorder fallbacks still apply to explicit `max_body_bytes: 0`
(4,096 bytes) and `max_records: 0` (200 records for the memory store). These zeros
override an inherited value; omit a field when you want to inherit it.

### Keep personal data out of Replay

Set `capture_personal_data: false` globally or on a decision's `router_replay`
plugin. The record keeps routing evidence, model, signals and detected PII types,
while personal-data requests omit request and response bodies, prompts, tool
definitions and tool traces. The recipe's configured PII signals are evaluated
when needed for this capture setting, even if no decision references them.

When PII evidence is unavailable, classification fails, or the recipe has no
PII detector configured, capture omits content conservatively. A PII signal is
optional; configuring one allows content capture for requests it identifies as
free of personal data. For example:

```yaml
routing:
  signals:
    pii:
      - name: personal_data
        pii_types_allowed: []
```

Request bodies, responses, and tool traces can contain secrets or personal data.
Capture only what you need and restrict access with `replay.read` and
`replay.detail`. See the [Replay API and privacy controls](../../api/router#router-replay).

## Read the results

Open **Insights** in the Dashboard, or use the
[Replay API](../../api/router#router-replay) to filter records and inspect a session.
Costs are [estimates from recorded tokens and configured rates](../../api/router#configured-rate-cost-estimates),
not invoices. Missing prices and usage remain unknown.

Every record carries `route_diagnostics.decision_ranking`: the strategy
selection ran under, the tier the winning decision came from, whether that pool
ranked by confidence and which decision stopped it, and the key that separated
the winner from the decision behind it, which is recorded as `runner_up`, with
a `reason` stating the values compared. A replayed request therefore explains
its route the same way the eval API does.

Detail responses can also include `route_diagnostics.prepared_dispatch`, a
versioned receipt for the exact primary provider-bound body returned to Envoy
after protocol encoding and provider adaptation. It records the wire format,
SHA-256 digest, and encoded byte length without retaining another copy of the
body. Internal Looper calls and response-time fallback attempts are not part of
this single-dispatch receipt. Because the digest is content-derived, responses
without `replay.detail` omit the field.

Confidence Looper requests also include `route_diagnostics.looper`: bounded
attempts, token and cost accounting, timings, disposition reasons, and the final
attempt. This diagnostic object excludes prompts, tool arguments, credentials,
and raw errors. Attempt details require replay-detail permission, and any
truncation is marked.

For a complete decision fragment, see
[`config/fragments/plugin/router-replay/debug.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/router-replay/debug.yaml).

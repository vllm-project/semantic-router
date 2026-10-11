# Tools

## Overview

`tools` is a route-local plugin for tool filtering and semantic tool selection.

## Key Advantages

- Keeps tool policy attached to the matched route.
- Lets one route disable tools while another route filters or semantically selects them.
- Composes with the global tools database instead of overloading `routing.decisions[]`.

## What Problem Does It Solve?

Tool behavior is part of route policy. Some routes should strip tools entirely, some should pass tools through unchanged, and some should constrain the semantic tool candidate pool. The `tools` plugin makes that route-local contract explicit.

## When to Use

- a route should disable all tools
- a route should semantically select tools from the global tools database
- a route should restrict tool access with explicit allow/block lists
- a privacy route should keep tool history available for routing but omit prior tool/function calls and results from the selected model request

## Configuration

Add the plugin under `routing.decisions[].plugins`:

```yaml
plugins:
  - type: tools
    configuration:
      enabled: true
      mode: filtered
      semantic_selection: true
      allow_tools:
        - docs.search
        - tickets.lookup
      block_tools:
        - admin.delete
```

In `filtered` mode, `allow_tools` and `block_tools` also bound the tools that
semantic selection or the `tool_selection` plugin adds from the tools
database, whether or not `advanced_filtering` is enabled.

Set `mode: none` with `strip_tool_history: true` when the selected backend must not receive prior
assistant tool/function calls or tool/function result messages. The router
applies this policy after signal and decision evaluation, so it does not change
which route matched. It only changes the provider-bound request body. Validation
rejects `strip_tool_history: true` with any other tool mode.

### Trusted tool facts

`trusted_facts` declares which facts may authorize tools for the decision
before relevance ranking or Looper execution. It is disabled unless
`enabled: true`, and requires the parent `tools` plugin to be enabled.

```yaml
plugins:
  - type: tools
    configuration:
      enabled: true
      mode: passthrough
      trusted_facts:
        enabled: true
        enforcement: authoritative
        trust_sources: [operator-policy, runtime-fresh]
        freshness_seconds: 300
        stage_roles: [candidate, final]
```

- `enforcement`: `advisory` (the default when enabled) only records the
  outcome; `authoritative` enforces it; `disabled` skips the gate.
- `trust_sources`: `operator-policy` authorizes tools by declaration.
  `gateway-attested` does not authorize on its own yet, because the router has
  no per-request attestation signal. `runtime-fresh` can only narrow: the
  tools database must have loaded successfully within `freshness_seconds`,
  which is required with this source.
- `stage_roles`: the roles that may receive tools. Request-path selection is
  the `candidate` stage. Looper checks each model call at its own stage:
  Confidence attempts, Ratings calls, Fusion panel members, and Workflows
  steps are `candidate`; Confidence self-verification is `verifier`; Fusion's
  separate analysis and the Workflows planner are `advisor`; Fusion synthesis
  and its quorum fallback, Workflows final synthesis, the ReMoM final round,
  and the Base Looper's answer are `final`.

Under `authoritative`, a missing authorization or excluded stage removes every
tool, including the Responses hosted `image_generation` tool. Missing or stale
availability keeps only the tools the plugin's mode and allow/block lists
already permit, with no retrieval expansion; `mode: none` still removes every
tool. Each outcome is recorded on the Router Replay record of the request,
or of the Looper call it gated.

Tool selection controls what reaches the model; it does not authorize tool
execution. Enforce permissions at the tool service and treat tool schemas and
results as provider-bound content. See a complete example:
[`config/fragments/plugin/tools/semantic-select.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/tools/semantic-select.yaml).

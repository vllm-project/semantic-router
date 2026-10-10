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

Set `mode: none` with `strip_tool_history: true` when the selected backend must not receive prior
assistant tool/function calls or tool/function result messages. The router
applies this policy after signal and decision evaluation, so it does not change
which route matched. It only changes the provider-bound request body. Validation
rejects `strip_tool_history: true` with any other tool mode.

Tool selection controls what reaches the model; it does not authorize tool
execution. Enforce permissions at the tool service and treat tool schemas and
results as provider-bound content. See a complete example:
[`config/fragments/plugin/tools/semantic-select.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/tools/semantic-select.yaml).

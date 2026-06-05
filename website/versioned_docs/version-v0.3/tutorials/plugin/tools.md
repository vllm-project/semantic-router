# Tools

## Overview

`tools` is a route-local plugin for tool filtering and semantic tool selection.

It aligns to `config/plugin/tools/semantic-select.yaml`.

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

## Configuration

Use this fragment under `routing.decisions[].plugins`:

```yaml
plugin:
  type: tools
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

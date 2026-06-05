# Tool Selection

## Overview

`tool_selection` is a decision plugin that controls how tools are chosen for a matched route.  
It supports two modes:

- `add`: retrieve tools from a tools database
- `filter`: filter tools that are already present in the incoming request

It aligns to fragments under `config/plugin/tool-selection/`.

## Key Advantages

- Separates route decision logic from tool retrieval/filter behavior.
- Supports both database-driven tool addition and request-tool semantic filtering.
- Keeps compatibility with route-local tool policies while making selection behavior explicit.

## What Problem Does It Solve?

Different routes need different tool-selection behavior. Some routes should add tools from a curated database, while others should keep only the most relevant tools from the caller-provided set. `tool_selection` provides one plugin contract for both cases, with per-route controls such as threshold, `top_k`, and preserve behavior.

## When to Use

- when a decision should add the most relevant tools from `tools_db`
- when a decision should semantically filter caller-provided `tools`
- when per-route tool selection mode must be explicit and configurable

## Configuration

Use this fragment under `routing.decisions[].plugins`:

```yaml
plugin:
  type: tool_selection
  configuration:
    enabled: true
    mode: filter
    relevance_threshold: 0.55
    preserve_count: 2
```

For add mode:

```yaml
plugin:
  type: tool_selection
  configuration:
    enabled: true
    mode: add
    top_k: 5
```

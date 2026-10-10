# Tool Selection

## Overview

`tool_selection` is a decision plugin that controls how tools are chosen for a matched route.  
It supports two modes:

- `add`: retrieve tools from a tools database
- `filter`: filter tools that are already present in the incoming request

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

Add the plugin under `routing.decisions[].plugins`:

```yaml
plugins:
  - type: tool_selection
    configuration:
      enabled: true
      mode: filter
      relevance_threshold: 0.25
      preserve_count: 2
```

For add mode:

```yaml
plugins:
  - type: tool_selection
    configuration:
      enabled: true
      mode: add
      tools_db_path: config/tools_db.json
      top_k: 5
      similarity_threshold: 0.35
```

`add` mode requires a populated tool database; `filter` mode only considers
tools already supplied by the caller. Semantic relevance is not authorization,
so enforce tool permissions separately. See complete examples:
[`add-from-database.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/tool-selection/add-from-database.yaml)
and
[`filter-request-tools.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/tool-selection/filter-request-tools.yaml).

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

### Session-scoped sticky selection

`sticky` is an opt-in, bounded policy layered on top of either mode (issue
[#3347](https://github.com/vllm-project/semantic-router/issues/3347)). A
trusted session retains the exact order of previously selected tools, pins
tools observed in assistant tool calls, and appends a bounded number of newly
relevant tools per turn — reducing prompt churn across a multi-turn tool-use
session without ever skipping current-turn authorization or availability
checks.

```yaml
plugins:
  - type: tool_selection
    configuration:
      enabled: true
      mode: add
      top_k: 3
      sticky:
        enabled: false
        max_tools: 16
        max_new_tools_per_turn: 2
        pin_called_tools: true
```

- `max_tools` (`1`-`128`, default `16`): hard bound on retained tools; once
  full, only called or definitionally-changed tools are re-evaluated.
- `max_new_tools_per_turn` (`0`-`max_tools`, default `2`): how many newly
  relevant tools may be appended in one turn. `0` disables relevance-driven
  growth entirely (reuse and call-pinning only) — this is a valid explicit
  setting, distinct from omitting the field.
- `pin_called_tools` (default `true`): tools observed in an assistant tool
  call are pinned and are not evicted by ordinary bounded growth.

Sticky state is scoped to a trusted, authenticated session — it is never
active for an anonymous or derived session identity — and every stored
identity is re-authorized and re-validated against the current request's
catalog, policy, and model/wire capabilities before use; a stored identity
is never trusted blindly. Full runtime behavior (this configuration
contract is Phase 1 of 4; the plugin does not yet read or write session
state) is tracked in
[PL-0042](https://github.com/vllm-project/semantic-router/blob/main/tools/agent/docs/plans/pl-0042-sticky-tool-selection.md).
See the complete disabled example:
[`sticky-add-from-database.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/tool-selection/sticky-add-from-database.yaml).

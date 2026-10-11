# Tool Selection

## Overview

`tool_selection` is a decision plugin that controls how tools are chosen for a matched route.  
It supports two modes:

- `add`: retrieve tools from a tools database
- `filter`: filter tools that are already present in the incoming request

Database tool names are trimmed and must be unique. Catalog enumeration uses
name order; equal similarity scores use name order before `top_k` truncation.
File loads and incremental additions reject duplicate names or non-finite
embedding values. A failed file batch leaves the existing catalog unchanged.

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
trusted session keeps the order of the tools it was given, pins tools its
assistant called, and adds a bounded number of newly relevant tools per turn.
Stable tool lists reduce prompt churn across a multi-turn tool-use session.
Every request still rechecks current authorization, availability, and
capability before a stored tool is emitted.

```yaml
plugins:
  - type: tool_selection
    configuration:
      enabled: true
      mode: add
      top_k: 3
      sticky:
        enabled: true
        max_tools: 16
        max_new_tools_per_turn: 2
        pin_called_tools: true
  - type: tools
    configuration:
      enabled: true
      mode: passthrough
      trusted_facts:
        enabled: true
        enforcement: authoritative
        trust_sources: [operator-policy]
        stage_roles: [candidate, final]
```

- `max_tools` (`1`-`128`, default `16`): hard bound on the tools a session
  keeps. Relevance never replaces a kept tool; a called tool can replace the
  newest unpinned one.
- `max_new_tools_per_turn` (`0`-`max_tools`, default `2`): how many newly
  relevant tools one turn may add. `0` keeps reuse and pinning only, which is
  different from omitting the field.
- `pin_called_tools` (default `true`): tools the assistant called in the
  conversation history are pinned. More pins than `max_tools` restart the
  session's selection.

Sticky selection is accepted only where the Router can recheck every reused
tool, at both config admission and router construction:

- the same decision enables the [`tools`](tools.md#trusted-tool-facts) plugin
  with `trusted_facts.enforcement: authoritative`, an authorizing trust source
  such as `operator-policy`, and `stage_roles` that include `candidate` and
  `final`, and its mode is not `none`;
- the decision dispatches to one model: Looper algorithms are rejected;
- `global.stores.tool_sessions.backend` is `local`, the in-process store owned
  by each router generation. A reload starts empty state and a restart loses
  it. The Redis backend is not supported yet;
- `USER_SCOPE_NAMESPACE_SECRET` is set.

**Which requests use session state.** State is read and written only for a
request with an authenticated principal and a session from the `x-session-id`
header or a Responses API conversation, under an entrypoint recipe, when its
trusted facts allow tool use for both stages and the target protocol can carry
tools. State is partitioned by recipe, principal, and session; a session value
from another principal never shares state. Everything else, and a request
whose `tool_choice` is not `auto` or that has no text to rank against, uses
ordinary selection without reading or writing state.

**What a session can reuse.** In `add` mode, tools come from the current tools
database; in `filter` mode, from the tools the current request offers. Only
tools the decision's allow and block lists permit are eligible. A stored tool
is emitted only with its current definition, so a tool that is removed,
blocked, or changed disappears on the next request, even when pinned. Any
change to the effective policy, the eligible catalog (including schema
numbers), or the target's capabilities restarts the session's selection.

**Failures.** If the store is unavailable, closed, or slow, the request uses
the current ordinary selection. If tool retrieval fails and `fallback_to_empty`
is not `true`, the request fails with
[`tool_selection_unavailable`](../../api/router.md) rather than forwarding the
tools it arrived with. Router response caching is skipped for sticky decisions;
provider prompt caching is unaffected. Each request records a bounded receipt
on its Router Replay record: an outcome, a reason, and counts, never tool
names, prompts, or identities.

See the complete example:
[`sticky-add-from-database.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/tool-selection/sticky-add-from-database.yaml).

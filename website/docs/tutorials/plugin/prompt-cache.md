# Prompt Cache

## Overview

`prompt_cache` is a route-local plugin that inserts Anthropic prompt-cache
breakpoints on a matched route. It never claims a cache hit or reports
savings; it only asks the provider to consider caching the marked content.

## Key Advantages

- Keeps cache-breakpoint policy attached to the route that benefits from it,
  instead of requiring every caller to manage markers itself.
- Never overrides a caller that already manages its own markers.
- Disabled by default, so enabling it on one route cannot change behavior on
  any other route.

## What Problem Does It Solve?

Anthropic supports explicit cache breakpoints on reusable prompt blocks.
Clients using a neutral request format cannot place those Anthropic-specific
markers themselves. `prompt_cache` lets a route mark repeated instructions and
tool definitions without requiring each client to understand the Anthropic
wire format. It does not enable Anthropic's separate top-level automatic
caching mode.

## When to Use

- an Anthropic-backed route sends the same system instructions or tool
  definitions on most turns
- callers on the route do not already send their own `cache_control` markers
- the route can tolerate a fixed 5-minute or 1-hour cache lifetime

Skip it when the route's traffic already sets its own markers, or when the
selected model does not speak the Anthropic Messages wire format.

## Configuration

Add the plugin under `routing.decisions[].plugins`:

```yaml
plugins:
  - type: prompt_cache
    configuration:
      enabled: true
      ttl: 1h
      targets: [instructions, tools]
      on_unsupported: skip
```

`ttl` accepts `5m` (default) or `1h`, matching the two lifetimes Anthropic
supports. `targets` accepts `instructions`, `tools`, or both (default both).
`on_unsupported` controls what happens when the selected backend does not
speak the Anthropic Messages wire format: `skip` (default) leaves the request
unchanged, `reject` returns the typed error code
`prompt_cache_target_unsupported`.

## Marker placement

When enabled and the neutral request carries no caller-supplied cache marker
anywhere, the router deterministically adds at most one marker to the last
eligible text instruction block and at most one marker to the last eligible
tool, so a route can add at most two router-inserted markers per request. If
any caller-supplied marker already exists anywhere in the request, caller
intent wins: every existing marker is preserved and the router inserts none.
User messages, assistant messages, tool calls, and tool results are not
marked by this plugin.

Anthropic supports up to four explicit cache breakpoints per request. This
plugin deliberately stays well under that ceiling: it only targets the two
blocks most likely to be stable across turns, and it never adds a marker once
the caller has taken ownership of cache placement. See Anthropic's
[prompt caching documentation](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
for the full provider-side contract.

## Runtime order

Marker insertion runs after routing and after request, context, and tool
transformations, immediately before the request is encoded onto the Anthropic
wire format. It only ever touches the outgoing Anthropic-shaped request; it
never changes the neutral request that routing and other plugins evaluate.

## Observability

Each evaluation records `llm_plugin_execution_total` with
`plugin_type=prompt_cache` and a `status` of `inserted`, `preserved`,
`skipped`, or `rejected`. With `x-vsr-debug: true` on the request, the same
outcome appears inline as `x-vsr-prompt-cache-action`,
`x-vsr-prompt-cache-reason`, `x-vsr-prompt-cache-inserted`, and
`x-vsr-prompt-cache-preserved`; see
[VSR routing headers](../../troubleshooting/vsr-headers.md#cache-and-plugin-headers).
No Router Replay persistence is part of this plugin.

See a complete example:
[`config/fragments/plugin/prompt-cache/anthropic-agent.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/prompt-cache/anthropic-agent.yaml).

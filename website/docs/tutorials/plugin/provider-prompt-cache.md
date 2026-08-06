# Provider Prompt Cache

## Overview

`provider_prompt_cache` injects provider-native prompt cache markers after a
decision selects a model. The initial implementation targets Anthropic request
translation and can mark the final system block, final tool definition, and
latest user block.

Explicit client-supplied cache markers take precedence over injected markers.

## Key Advantages

- Keeps stable system and tool prefixes cacheable across turns.
- Applies after model selection, so provider-specific translation owns markers.
- Preserves explicit client cache markers.
- Feeds provider cache usage into existing cost and replay telemetry.

## What Problem Does It Solve?

Stable agent instructions and tool definitions are repeatedly billed and
processed when clients do not add provider cache markers themselves.

## When to Use

Use it for stable multi-turn agent routes, ideally together with conversation
protection. Avoid marking rapidly changing user content unless the provider
cache behavior is intentional.

## Configuration

```yaml
routing:
  decisions:
    - name: stable-agent-route
      plugins:
        - type: provider_prompt_cache
          configuration:
            enabled: true
            system: true
            tools: true
            last_user: false
            ttl: 5m
            allow_request_controls: true
            control_header: x-vsr-provider-cache-control
```

Supported TTL values are `5m` and `1h`. Provider cache usage is read from the
normal response usage fields and flows into cost accounting, session telemetry,
and Router Replay.

When explicitly enabled, the control header accepts `bypass` or `no-cache` to
disable automatic marker injection for that request. Callers cannot use the
header to enable markers that the route did not authorize.

Use conversation protection when the workload benefits from retaining the same
provider model across turns.

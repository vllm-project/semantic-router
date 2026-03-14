# Response and Mutation Plugins

## Overview

This page covers the route-local plugins that either change the request or response shape, or short-circuit the route entirely.

It aligns to:

- `config/plugin/fast-response/`
- `config/plugin/header-mutation/`
- `config/plugin/system-prompt/`
- `config/plugin/image-gen/`

## Key Advantages

- Keeps response shaping local to the route that needs it.
- Supports cheap fallbacks without changing global defaults.
- Makes route-specific downstream headers and prompts explicit.
- Lets multimodal or image-generation routes opt into different behavior cleanly.

## What Problem Does It Solve?

Some routes need custom response behavior, but not every route should inherit it. If that logic is global, unrelated routes become harder to reason about.

These plugins solve that by attaching mutation or short-circuit behavior to one matched decision.

## When to Use

Use this family when:

- one route should return a deterministic fallback
- one route needs downstream headers
- one route needs a route-specific instruction layer
- one route hands off to image generation instead of standard chat flow

## Configuration

Each plugin lives inside `routing.decisions[].plugins`.

### Fast Response

```yaml
routing:
  decisions:
    - name: overload_fallback
      plugins:
        - type: fast_response
          configuration:
            message: The primary model is saturated, so a lightweight response was returned immediately.
```

### Header Mutation

```yaml
routing:
  decisions:
    - name: premium_route
      plugins:
        - type: header_mutation
          configuration:
            add:
              - name: X-Tenant-Tier
                value: premium
```

### System Prompt

```yaml
routing:
  decisions:
    - name: expert_route
      plugins:
        - type: system_prompt
          configuration:
            enabled: true
            mode: insert
            system_prompt: You are a domain expert. Answer precisely and keep the response actionable.
```

### Image Generation

```yaml
routing:
  decisions:
    - name: image_route
      plugins:
        - type: image_gen
          configuration:
            enabled: true
            backend: image-primary
```

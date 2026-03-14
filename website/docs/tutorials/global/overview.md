# Global

## Overview

`global:` is the router-wide override layer.

Unlike `signal/`, `decision/`, `algorithm/`, and `plugin/`, this section is not route-local. It defines shared runtime behavior, shared backing services, and built-in model bindings.

## Key Advantages

- Gives the router one shared place for runtime overrides.
- Avoids duplicating shared backing-service settings across routes.
- Keeps route-local matching in `routing:` and runtime-wide behavior in `global:`.
- Works with router-owned defaults, so users only override what they need.

## What Problem Does It Solve?

Some configuration belongs to the whole router, not to any one route. If that state leaks into route-local config, it becomes harder to reuse routes and harder to understand what is shared versus local.

`global:` solves that by holding sparse, router-wide overrides on top of built-in defaults.

## When to Use

Use `global:` when:

- a setting should apply across multiple routes
- a shared backing store or runtime service must be configured once
- built-in system models or runtime policy need an override
- the behavior is not specific to a single matched decision

## Configuration

Canonical placement:

```yaml
global:
  observability:
    metrics:
      enabled: true
```

The latest global docs mirror the main runtime groupings:

| Global area | Examples | Doc |
|-------------|----------|-----|
| interfaces and observability | `api`, `response_api`, `observability`, `router_replay` | [API and Observability](./api-and-observability) |
| stores and tools | `semantic_cache`, `memory`, `vector_store`, `tools` | [Stores and Tools](./stores-and-tools) |
| safety, models, and policy | `prompt_guard`, `classifier`, `embedding_models`, `external_models`, `authz`, `ratelimit`, `model_selection`, `system_models`, `looper` | [Safety, Models, and Policy](./safety-models-and-policy) |

Keep these rules in mind:

- keep `global:` sparse; rely on router defaults when possible
- put shared backing services in `global:`
- keep route-local matching in `routing:`

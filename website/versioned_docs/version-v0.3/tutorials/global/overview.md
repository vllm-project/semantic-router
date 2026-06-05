# Global

## Overview

`global:` is the router-wide override layer.

Unlike `signal/`, `decision/`, `algorithm/`, and `plugin/`, this section is not route-local. It defines shared runtime behavior, shared backing services, built-in model assets, and shared capability modules.

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
  router:
    config_source: file
  services:
    observability:
      metrics:
        enabled: true
```

The latest global docs mirror the main runtime groupings:

| Global area | Examples | Doc |
|-------------|----------|-----|
| router and services | `router.config_source`, `router.model_selection`, `services.api`, `services.response_api`, `services.observability`, `services.router_replay` | [API and Observability](./api-and-observability) |
| stores and integrations | `stores.semantic_cache`, `stores.memory`, `stores.vector_store`, `integrations.tools`, `integrations.looper` | [Stores and Tools](./stores-and-tools) |
| model catalog and modules | `model_catalog.embeddings`, `model_catalog.external`, `model_catalog.system`, `model_catalog.modules.prompt_guard`, `model_catalog.modules.classifier`, `model_catalog.modules.hallucination_mitigation` | [Safety, Models, and Policy](./safety-models-and-policy) |

Keep these rules in mind:

- keep `global:` sparse; rely on router defaults when possible
- keep `global.router.config_source` at `file` unless Kubernetes CRD reconciliation is intentionally driving runtime config
- put shared backing services in `global:`
- keep route-local matching in `routing:`

# Global Configuration

## Overview

`global:` contains Router-wide settings and shared infrastructure. It is the
counterpart to each route-local `recipes[].document`: define a service or store
once under `global:`, then opt individual routes into it with signals,
algorithms, or plugins.

## Key Advantages

- Defines shared infrastructure once for all recipes and routes.
- Keeps route-local policy separate from platform-wide services.
- Makes model, storage, and external-service dependencies explicit.

## What Problem Does It Solve?

Embedding runtimes, APIs, storage backends, observability, and helper services
are shared resources. Keeping them outside decisions avoids duplicated
connections and makes the data and trust boundaries visible.

## When to Use

Use `global:` for behavior or infrastructure shared by several Recipes. Keep
route matching, algorithms, and plugins in each Recipe document; keep candidate
Models in Entrypoint assignments. Global settings are shared across Recipes,
while signals, projections, decisions, algorithms, and plugins remain
Recipe-scoped.

## Configuration

```yaml
global:
  services:
    observability:
      metrics:
        enabled: true
  stores:
    response_cache:
      enabled: true
      backend_type: memory
```

Global configuration has five groups:

| Group | Owns | Guide |
|---|---|---|
| `global.router` | Router engine controls, selection defaults, streamed-body policy, learning | [Algorithms](../algorithm/overview), [Router Learning](../learning/overview) |
| `global.services` | API, Response API, observability, Router-native access, backend dispatch, management API, startup status, replay | [API and Observability](./api-and-observability) |
| `global.stores` | response cache, memory, vector store | [Stores and Tools](./stores-and-tools) |
| `global.integrations` | tool catalog and Looper endpoint/state | [Stores and Tools](./stores-and-tools) |
| `global.model_catalog` | embeddings, system models, external helpers, knowledge bases, capability modules | [Safety, Models, and Policy](./safety-models-and-policy) |

Entrypoints and named recipes are top-level objects rather than global
settings; see [Virtual Models](./entrypoints-and-recipes).
Remote text embeddings are covered in
[Remote Embedding Providers](./remote-embeddings).

## Operational Boundaries

- Keep overrides sparse; omitted fields inherit Router defaults.
- Put credentials in environment variables or Kubernetes Secrets, not literal
  YAML values.
- Persistent stores may contain prompts, responses, embeddings, memories, or
  replay records. Set backend authentication, transport security, retention,
  and tenant/user scope deliberately.
- `models[].reasoning` belongs to the immutable Model value, not `global:`.
- See the complete configuration reference in
  [`config/config.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml).

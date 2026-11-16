---
title: Global settings
sidebar_label: Global settings
description: The global block holds Router-wide settings and shared infrastructure that every recipe can use.
---

# Global settings

`global:` holds Router-wide settings and shared infrastructure: the Router
engine, its services, stores, integrations, model catalog and default plugins.
Define a service or store once under `global:`, then opt individual routes into
it with signals, algorithms or plugins.

```yaml title="config.yaml"
global:
  services:
    observability:
      metrics:
        enabled: true
  stores:
    response_cache:
      backend_type: memory
  plugins:
    - type: response_cache
      configuration:
        enabled: true
```

## Why a separate block

- **Define shared infrastructure once** for all recipes and routes, instead of
  repeating connections in every decision.
- **Keep route-local policy apart** from platform-wide services: route matching,
  candidate models, algorithms and plugins stay in `routing:`.
- **Make dependencies explicit.** Models, storage and external services appear
  in one place, so their data and trust boundaries are visible.

:::note
Global settings are shared across recipes. Signals, projections and decisions
are recipe-scoped, while algorithms and plugins belong to individual
decisions. Entrypoints and named recipes are top-level objects rather than
global settings; see [Virtual Models](../tutorials/global/entrypoints-and-recipes.md).
:::

## The six groups

| Group | Owns | Guide |
| --- | --- | --- |
| `global.router` | Router engine controls, selection defaults, streamed-body policy, routing input, learning | [Algorithms](../tutorials/algorithm/overview.md), [Router Learning](../tutorials/learning/overview.md), [Routing input](../tutorials/signal/overview.md#choose-the-user-text-signals-read) |
| `global.services` | API, Response API, observability, authz, rate limits, management API, startup status, replay | [API and Observability](../tutorials/global/api-and-observability.md) |
| `global.stores` | response cache, memory, vector store | [Stores and Tools](../tutorials/global/stores-and-tools.md) |
| `global.integrations` | tool catalog and Looper runtime/state | [Stores and Tools](../tutorials/global/stores-and-tools.md) |
| `global.model_catalog` | deployments, bindings, admission, the embedding and decision models | [Connect the Router to the model runtime](../model-runtime/deploy/connect-the-router.md) |
| `global.plugins` | the default plugins every request runs, such as `context` (memory, `compress.signals`) and `router_replay` | [Plugins](../tutorials/plugin/overview.md) |

Remote text embeddings are covered in
[Embeddings](../model-runtime/guides/embeddings.md#use-an-external-embedding-service).

## Operational boundaries

- **Keep overrides sparse.** Omitted fields inherit Router defaults.
- **Keep secrets out of YAML.** Put credentials in environment variables or
  Kubernetes Secrets, not literal YAML values.
- **Model reasoning is per model.** Built-in reasoning behavior comes from
  `providers.models[].catalog`. Self-hosted models can select a built-in family
  or define inline behavior in `providers.models[].reasoning`; neither belongs
  under `global:`.

:::warning
Persistent stores may contain prompts, responses, embeddings, memories or
replay records. Set backend authentication, transport security, retention and
tenant or user scope deliberately.
:::

The complete configuration reference is
[`config/config.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml).

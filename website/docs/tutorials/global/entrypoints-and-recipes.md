---
title: Virtual Models
description: Give clients stable virtual model names backed by isolated routing policies in one Semantic Router deployment.
---

# Virtual Models

## Overview

Entrypoints and recipes turn one Semantic Router deployment into a set of
purpose-built virtual models:

- an **Entrypoint** is the callable model name and its complete assignments;
- a **Recipe** is reusable routing policy; and
- Models and shared services remain available to every
  recipe.

## What Problem Does It Solve?

This separation lets an application choose an objective such as low latency,
high quality, or a balanced trade-off without knowing which backend model will
serve the request.

In canonical YAML, `entrypoints` holds the public-name mappings and `recipes`
holds the named routing policies.

## How the pieces fit

```text
request model name -> Entrypoint -> Recipe -> decision assignment -> backend
```

When the request `model` matches one of an Entrypoint's `model_names`, the Router
selects its Recipe and assigns Models to every Decision name. The virtual Model
name is then replaced by the
selected backend Model.

There is no implicit default Recipe or automatic alias for a named Recipe.
Every virtual model authored in
`entrypoints` is explicit, and every Entrypoint assigns all of its Recipe's
decisions. The preserved top-level `routing` shorthand is the only exception:
when it contains a complete profile, it retains the established automatic names
`vllm-sr/auto`, `auto`, and `MoM` (or the configured primary name) unless an
explicit Entrypoint claims them. Set `global.router.auto_model_names: []` to
disable those shorthand names. Generated snapshot identities are not part of
human authoring. `model_names` lists every callable alias for an explicit
Entrypoint; those aliases all resolve to the same Entrypoint.

Concrete backend model names are different: they select that model directly
and bypass recipe routing. Use a virtual entrypoint when clients should ask for
an objective, and a concrete model name only when they intentionally need that
exact backend.

## Configuration

Models are shared resources. Each named Recipe owns its signals, projections,
decisions, strategy, algorithms, and route-local plugins.

```yaml
providers:
  models:
    - name: local/fast
      provider_model_id: fast-model
      backend_refs:
        - {provider: vllm, endpoint: http://fast-model:8000/v1}
    - name: local/accurate
      provider_model_id: accurate-model
      backend_refs:
        - {provider: vllm, endpoint: http://accurate-model:8000/v1}

routing:
  modelCards:
    - name: local/fast
      description: Fast general-purpose model
    - name: local/accurate
      description: Higher-quality model

recipes:
  - name: flash
    description: Prefer the lowest-latency eligible backend.
    routing:
      strategy: priority
      decisions:
        - name: fast-path
          description: Serve requests with the fast model.
          priority: 100
          rules:
            operator: AND
            conditions: []

  - name: ultra
    description: Prefer the highest-quality eligible backend.
    routing:
      strategy: priority
      decisions:
        - name: quality-path
          description: Serve requests with the accurate model.
          priority: 100
          rules:
            operator: AND
            conditions: []

entrypoints:
  - model_names: [vllm-sr/mom-v1-flash]
    recipe: flash
    assignments:
      fast-path: {models: [{model: local/fast}]}
  - model_names: [vllm-sr/mom-v1-ultra]
    recipe: ultra
    assignments:
      quality-path:
        models: [{model: local/accurate, reasoning: {enabled: true, effort: high}}]
```

Clients can discover entrypoint names through `/v1/models`. Routed responses
include `x-vsr-selected-recipe`, so operators can confirm which policy handled
a request without exposing the backend selection contract to the client.

## When to Use

Use named entrypoints and recipes when one deployment must expose more than one
routing objective, policy boundary, or rollout track. Even a deployment with
one policy publishes that policy as one Recipe plus one Entrypoint.

Continue with:

- [Entrypoints](entrypoints) for naming, request resolution, discovery, and
  validation rules.
- [Recipes](recipes) for policy isolation, shared infrastructure, lifecycle
  APIs, and limitations.
- [Models, Entrypoints, and Serving](models-entrypoints-serving) for the
  end-to-end catalog, CLI, backend binding, serving, and operations workflow.

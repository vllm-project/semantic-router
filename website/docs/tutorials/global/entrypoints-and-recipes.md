---
title: Entrypoints and Recipes
description: Give clients stable virtual model names backed by isolated routing policies in one Semantic Router deployment.
---

# Entrypoints and Recipes

## Overview

Entrypoints and recipes turn one Semantic Router deployment into a set of
purpose-built virtual models:

- an **entrypoint** is the model name a client requests;
- a **recipe** is the routing policy that handles requests for that name; and
- providers, model endpoints, and shared services remain available to every
  recipe.

## What Problem Does It Solve?

This separation lets an application choose an objective such as low latency,
high quality, or a balanced trade-off without knowing which backend model will
serve the request.

In canonical YAML, `entrypoints` holds the public-name mappings and `recipes`
holds the named routing policies.

## How the pieces fit

```text
request model name -> entrypoint -> recipe -> decision -> algorithm -> backend
```

When the request `model` matches an `entrypoints[].model_names` value, the
Router evaluates only the mapped recipe. The virtual model name is then
replaced by the backend selected from that recipe.

The top-level `routing` block remains the `default` recipe. Requests for
`vllm-sr/auto`, `auto`, or another configured auto alias use that default
policy. If the selected recipe has no matching decision, the Router uses
`providers.defaults.default_model`.

Concrete backend model names are different: they select that model directly
and bypass recipe routing. Use a virtual entrypoint when clients should ask for
an objective, and a concrete model name only when they intentionally need that
exact backend.

## Configuration

The model catalog is shared. Each named recipe owns its signals, projections,
decisions, strategy, algorithms, and route-local plugins.

```yaml
routing:
  modelCards:
    - name: fast-model
    - name: accurate-model

entrypoints:
  - model_names: [vllm-sr/mom-v1-flash]
    recipe: flash
  - model_names: [vllm-sr/mom-v1-ultra]
    recipe: ultra

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
          modelRefs:
            - model: fast-model

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
          modelRefs:
            - model: accurate-model
```

Clients can discover entrypoint names through `/v1/models`. Routed responses
include `x-vsr-selected-recipe`, so operators can confirm which policy handled
a request without exposing the backend selection contract to the client.

## When to Use

Use named entrypoints and recipes when one deployment must expose more than one
routing objective, policy boundary, or rollout track. Keep a single top-level
`routing` profile when all clients should follow the same policy; the existing
auto-model flow needs no extra configuration.

Continue with:

- [Entrypoints](entrypoints) for naming, request resolution, discovery, and
  validation rules.
- [Recipes](recipes) for policy isolation, shared infrastructure, lifecycle
  APIs, and limitations.
- [Models and Recipes](../../installation/models-and-recipes) for the built-in
  catalog, Model Cards, forking, and packaging workflows.

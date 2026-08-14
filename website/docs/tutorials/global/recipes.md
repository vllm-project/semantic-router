---
title: Recipes
description: Define isolated routing policies that share provider models and platform services within one Semantic Router deployment.
---

# Recipes

## Overview

A recipe is a complete routing-policy boundary. It owns the evidence and
control flow used for one objective: signals, projections, decisions, routing
strategy, selection or looper algorithms, and route-local plugins.

## What Problem Does It Solve?

Recipes avoid two poor scaling patterns: mixing unrelated policies into one
decision graph, or duplicating an entire Router deployment for every routing
objective. They keep policy and runtime state separate while reusing expensive
model endpoints and shared services.

## When to Use

Add recipes when consumers need different latency, quality, cost, privacy, or
safety policies. They are also useful for staging a new policy beside the
current one before moving clients to its entrypoint.

Keep one top-level `routing` block when a single policy serves all traffic. The
Router treats that block as the `default` recipe, so existing configurations do
not need to be rewritten.

## Configuration

Provider bindings and model cards stay at the top level. A named recipe refers
to those shared model names from its own routing block:

```yaml
routing:
  modelCards:
    - name: local-model
    - name: general-model

entrypoints:
  - model_names: [vllm-sr/privacy-v1]
    recipe: privacy

recipes:
  - name: privacy
    description: Keep prompts containing sensitive identifiers on the local model.
    routing:
      strategy: priority
      signals:
        pii:
          - name: sensitive-input
            threshold: 0.5
      decisions:
        - name: local-sensitive-route
          description: Keep detected PII on the local backend.
          priority: 200
          rules:
            operator: AND
            conditions:
              - type: pii
                name: sensitive-input
          modelRefs:
            - model: local-model
        - name: general-route
          description: Handle remaining requests with the general backend.
          priority: 100
          rules:
            operator: AND
            conditions: []
          modelRefs:
            - model: general-model
```

Names resolve inside the owning recipe. Two recipes may both define a signal or
decision called `sensitive-input`; neither can reference the other recipe's
definition.

## What is isolated and what is shared

| Recipe-local | Shared by the deployment |
| --- | --- |
| Signals and their thresholds | Provider models and backend endpoints |
| Projections and dependency graph | Top-level `routing.modelCards` |
| Decisions, priorities, and routing strategy | Router-owned classifier and embedding assets |
| Selection and looper policy | API, identity, observability, and transport settings |
| Route-local plugins | External stores and integration services |
| Cache, replay, learning, session, and metric namespaces | Model files and service connections |

Shared infrastructure does not make policy global. For example, a shared PII
model can serve several recipes while each recipe defines its own PII signal,
threshold, and decision behavior.

## Update recipes safely

The management API can validate and change one recipe without replacing the
rest of the document:

| Method and path | Purpose |
| --- | --- |
| `GET /config/router/recipes` | List recipes and their entrypoints; retain the response `ETag`. |
| `POST /config/router/recipes/validate` | Validate a proposed recipe without writing or reloading. |
| `PUT /config/router/recipes/{name}` | Create or replace a recipe and its entrypoints. |
| `DELETE /config/router/recipes/{name}` | Delete an unreferenced named recipe. |

Mutations require the current `ETag` in `If-Match`. A missing precondition
returns `428`; a stale value returns `412`. An accepted mutation validates the
complete configuration, writes it atomically with a backup, triggers Router
runtime activation, and returns a new `ETag`. Activation may continue after a
`202` response; poll `/config/hash` before treating the new policy as active.

The `default` recipe cannot be deleted. Before deleting another recipe, remove
or move every entrypoint that refers to it. See the
[management API reference](../../api/apiserver) for endpoint details.

## Limits and security boundaries

- A named recipe cannot declare `routing.modelCards`; all recipes use the
  shared top-level catalog.
- Signal, projection, and decision references must resolve within the same
  recipe.
- If no decision matches, routing falls back to the deployment's configured
  default provider model, not to another recipe.
- Recipe isolation does not isolate the Router process, network, provider
  credentials, or backing services. Use separate deployments when those must
  be tenant boundaries.
- Route plugins and shared services may persist prompts, responses, and routing
  metadata. Apply retention, access, and encryption policies to every enabled
  store.

Complete maintained examples and their Model Cards live under
[`config/recipes/`](https://github.com/vllm-project/semantic-router/tree/main/config/recipes).

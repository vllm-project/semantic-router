---
title: Recipes
description: Define isolated, model-free routing policies that share Models and platform services within one Semantic Router deployment.
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

Even a deployment with one policy publishes it through an explicit Entrypoint.
A compact file may place that policy in the top-level `routing` profile and
reference the generated `default` Recipe name; named Recipes are clearer when
several policies coexist.

## Configuration

Provider Models and routing Model cards stay outside Recipes. A Recipe is
deliberately model-free; the
Entrypoint supplies all decision assignments. The example assumes
`local/private` and `hosted/general` are declared in `providers.models` and
`routing.modelCards`:

```yaml
entrypoints:
  - model_names: [vllm-sr/privacy-v1]
    recipe: privacy
    assignments:
      local-sensitive-route: {models: [{model: local/private}]}
      general-route: {models: [{model: hosted/general}]}

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
        - name: general-route
          description: Handle remaining requests with the general backend.
          priority: 100
          rules:
            operator: AND
            conditions: []
```

Names resolve inside the owning recipe. Two recipes may both define a signal or
decision called `sensitive-input`; neither can reference the other recipe's
definition.

## What is isolated and what is shared

| Recipe-local | Shared by the deployment |
| --- | --- |
| Signals and their thresholds | Provider Models and backend references |
| Projections and dependency graph | Model semantic metadata and capabilities |
| Decisions, priorities, and routing strategy | Shared classifier and embedding assets |
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
| `GET /management/v1/routing/recipes` | List Recipe resources. |
| `POST /management/v1/routing/recipes` | Create one Recipe revision. |
| `PATCH /management/v1/routing/recipes/{recipeId}` | Create the next revision of a Recipe. |
| `DELETE /management/v1/routing/recipes/{recipeId}` | Delete an unreferenced Recipe using revision CAS. |

Mutations use the Management API's revision and idempotency contract. A Recipe
cannot be deleted while an Entrypoint references it. Publishing the Entrypoint
creates the immutable routing snapshot consumed by every replica. See the
[management API reference](../../api/apiserver) for endpoint details.

## Limits and security boundaries

- A Recipe cannot declare Models, backends, credentials, or Model assignments.
- Signal, projection, and decision references must resolve within the same
  recipe.
- Every Recipe contains at least one decision. Add an explicit empty-condition
  catch-all when unmatched requests should have a fallback route.
- Recipe isolation does not isolate the Router process, network, provider
  credentials, or backing services. Use separate deployments when those must
  be tenant boundaries.
- Route plugins and shared services may persist prompts, responses, and routing
  metadata. Apply retention, access, and encryption policies to every enabled
  store.

Browse the
[complete recipe examples and their Model Cards](https://github.com/vllm-project/semantic-router/tree/main/config/recipes).
For the catalog, CLI, backend binding, and serving workflow, start with
[Models, Entrypoints, and Serving](models-entrypoints-serving).

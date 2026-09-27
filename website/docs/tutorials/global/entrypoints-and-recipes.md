---
title: Virtual Models
description: Give clients stable virtual model names backed by isolated routing policies in one Semantic Router deployment.
---

# Virtual Models

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
`providers.defaults.model`.

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

## Limits for agent clients

`/v1/models` tells a client which virtual names exist and how each one
resolves. It does not report a context window, output limit, or capability for
them, and the model behind a name can change from one request to the next.
This is the entry for `vllm-sr/auto`:

```json
{
  "id": "vllm-sr/auto",
  "object": "model",
  "created": 1790323030,
  "owned_by": "vllm-semantic-router",
  "description": "Intelligent Router for Mixture-of-Models",
  "routing": {
    "resolution": "virtual",
    "selectable": true,
    "default_route": true,
    "recipe": "default"
  }
}
```

Coding agents and other clients that size a request before sending it need
these values in their own configuration. Any turn of a session can reach any
model the recipe can select, including `providers.defaults.model`, so
configure the client with the intersection of their model cards:

| Client setting | Value |
| --- | --- |
| Context window | The smallest `context_window_size` |
| Output limit | The smallest `max_output_tokens` |
| Tool calling | On only if every model declares `tools` |
| Image input | On only if every model declares `vision` or `image_input` |
| Reasoning settings | Sent only if every model declares `reasoning` |

For a recipe that selects among the three models below, configure a
32,768-token context window, an 8,192-token output limit, and tool calling.
Leave image input and reasoning settings off.

```yaml
routing:
  modelCards:
    - name: local-coder
      context_window_size: 32768
      max_output_tokens: 8192
      capabilities: [chat, tools]
    - name: reasoner
      context_window_size: 200000
      max_output_tokens: 64000
      capabilities: [chat, tools, reasoning]
    - name: vision-generalist
      context_window_size: 131072
      max_output_tokens: 16384
      capabilities: [chat, tools, vision]
```

By default, the Router skips a candidate whose declared context window is
smaller than the estimated input, or whose declared capabilities lack a
required input such as images. It does not check output limits, so a request
for 16,384 output tokens can still reach `local-coder`. With
[`candidate_requirements`](../../installation/configuration#recipe-wide-candidate-and-replay-policies)
on the recipe, the Router also checks output limits and tool, reasoning, and
structured-output declarations before scoring. A request that fits only some
candidates goes to one of them, and one that fits none is rejected before
dispatch; see [Request budget errors](../../api/router#request-budget-errors).
A client configured with the intersection keeps every candidate available to
every request.

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
- [Models, Entrypoints, and Serving](models-entrypoints-serving) for the
  end-to-end catalog, CLI, backend binding, serving, and operations workflow.

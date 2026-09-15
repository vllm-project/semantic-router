---
title: Entrypoints
description: Expose stable virtual model names that select a routing recipe through the standard OpenAI-compatible model field.
---

# Entrypoints

## Overview

An entrypoint is a public virtual model name that maps to one recipe. Clients
select it through the normal OpenAI-compatible `model` field, so they do not
need a Router-specific API or header.

## What Problem Does It Solve?

Entrypoints solve a common coupling problem: an application can ask for a
stable objective such as `vllm-sr/mom-v1-flash` while operators change the models,
thresholds, or algorithms behind that objective.

## When to Use

Create an entrypoint when you want to:

- publish latency, quality, cost, safety, or team-specific routing objectives;
- move a client between policy versions without exposing backend model IDs; or
- run several isolated policies in one Router deployment.

Use the configured auto alias when every routed request should use the default
policy. Use a concrete provider model name only when the caller deliberately
wants to bypass signals, decisions, algorithms, and route-local plugins.

## Configuration

Each entrypoint lists one or more aliases and the named recipe they select:

```yaml
entrypoints:
  - model_names:
      - vllm-sr/mom-v1-flash
      - company/fast
    recipe: flash

recipes:
  - name: flash
    description: Low-latency routing for interactive requests.
    routing:
      strategy: priority
      decisions: []
```

Both aliases select the same recipe. A client uses either name like any other
chat-completions model:

```bash
curl http://localhost:8899/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "vllm-sr/mom-v1-flash",
    "messages": [{"role": "user", "content": "Summarize this request."}]
  }'
```

The entrypoint name never reaches a provider. After the recipe chooses a
backend, the Router rewrites the request to that backend's model name.

## Request resolution

| Requested model | Router behavior |
| --- | --- |
| An `entrypoints[].model_names` value | Evaluate only the mapped recipe. |
| `vllm-sr/auto`, `auto`, or another configured auto alias | Evaluate the `default` recipe from top-level `routing`. |
| A configured ReMoM, Fusion, or Flow virtual slug | Run that looper in the `default` recipe. |
| A concrete provider model or LoRA name | Send directly to that backend without recipe routing. |

Entrypoints are listed by `/v1/models` with routing metadata. Successful routed
responses expose `x-vsr-selected-recipe`; Router Replay and Insights can also
filter records by recipe.

## Naming and validation rules

Configuration loading rejects an entrypoint when:

- `model_names` is empty or `recipe` names no configured recipe;
- the same virtual name is claimed by more than one entrypoint; or
- a virtual name collides with a provider model, LoRA, auto alias, or looper
  slug.

Choose names that describe a durable client contract, not the current backend.
Do not put tenant data or secrets in a name: entrypoints appear in model
discovery, response metadata, metrics, and operational records.

An entrypoint is a policy selector, not a security boundary. Recipes share the
Router process and configured infrastructure; use network, compute, and storage
isolation when tenants require stronger separation.

Start with [Models, Entrypoints, and Serving](models-entrypoints-serving) for
the end-to-end CLI workflow, or continue to [Recipes](recipes) for the policy
owned by an entrypoint.

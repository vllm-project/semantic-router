# Entrypoints and Multi-Recipe Routing

## Overview

One router configuration can carry multiple named routing profiles. The top-level `routing` block is the `default` recipe; `recipes` adds named additional profiles, and `entrypoints` maps request-facing virtual model names onto them. A client selects a routing profile simply by setting the request `model` field to an entrypoint name.

## Key Advantages

- One deployment serves several routing policies without duplicating provider, model-catalog, or global system configuration.
- Clients opt into a profile with nothing but the OpenAI-compatible `model` field; no extra headers or endpoints.
- The default profile keeps working unchanged: `vllm-sr/auto`, `auto`, and concrete model names behave exactly as before.

## What Problem Does It Solve?

Before this layer, one router carried exactly one working routing profile. Serving a second policy — a privacy-first profile, a cost-saver profile, a team-specific profile — meant running a second router or overloading one decision graph with unrelated concerns.

`entrypoints` and `recipes` split that cleanly: shared assets stay global, routing profiles become named and selectable per request.

## When to Use

Use entrypoints and recipes when:

- different consumers of one router need different routing policies
- a policy such as privacy handling or cost control should be opt-in per request rather than baked into the default decision graph
- you want to stage a new routing profile next to the production default and switch clients over gradually

Keep a single `routing` block when one profile serves all traffic — the layer is entirely optional.

## Configuration

```yaml
entrypoints:
  - model_names: ["vllm-sr/privacy"]
    recipe: privacy-first

recipes:
  - name: privacy-first
    description: Keep privacy-sensitive prompts on the local model.
    routing:
      signals:
        keywords:
          - name: privacy_terms
            operator: OR
            keywords: ["ssn", "passport number"]
      decisions:
        - name: privacy_route
          rules:
            operator: AND
            conditions:
              - type: keyword
                name: privacy_terms
          modelRefs:
            - model: qwen3-8b
              use_reasoning: false
```

### How requests resolve

- A request model name that matches an entrypoint behaves like an auto-model alias: the router evaluates the selected recipe's decisions and rewrites the body to the concrete model the recipe chooses. The virtual name never reaches a backend.
- If no decision in the recipe matches, the request falls back to `providers.defaults.default_model`, the same way `vllm-sr/auto` does.
- Entrypoint names appear in `/v1/models` next to the auto aliases, using the recipe description as the listing description.
- Request models that match no entrypoint keep the existing behavior: auto aliases run the default profile, concrete model names pass through.

### Sharing rules

- `recipes[].routing` carries `signals`, `projections`, and `decisions` — the same profile shape as the top-level `routing` block — but never `modelCards`. The model catalog and provider bindings stay shared.
- Signal and projection names share one global registry across recipes. Every recipe may reference every registered signal; declaring the same name in two profiles fails validation.
- A recipe named `default` is only valid when the top-level `routing` block carries no profile of its own, so existing single-profile configs cannot silently change meaning.

### Validation

Config load rejects:

- duplicate recipe names, and a `default` recipe conflicting with a non-empty top-level `routing` block
- entrypoints referencing unknown recipes, entrypoints without model names, and the same model name claimed by two entrypoints
- recipe-owned `modelCards`
- signal or projection names declared by more than one profile

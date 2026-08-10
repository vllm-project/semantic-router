# Entrypoints and Multi-Recipe Routing

## Overview

One router can expose several isolated routing policies. The top-level
`routing` block is the `default` recipe, `recipes` adds named recipes, and
`entrypoints` maps request-facing virtual model names to them. A client selects
a recipe by setting the normal OpenAI-compatible `model` field.

A recipe is the complete routing isolation boundary. It owns its signal and
policy definitions, projection graph, decisions, routing strategy, decision
algorithms, and route-local plugins. A request routed through recipe A never
evaluates or mutates routing state owned by recipe B.

## Key Advantages

- one deployment can expose multiple policies without duplicating model and
  provider infrastructure
- clients select policy through the standard `model` field
- policy definitions and runtime state cannot leak across recipe boundaries
- existing single-profile configurations remain the default recipe

## What Problem Does It Solve?

Without recipes, independent consumers either need separate router deployments
or one decision graph containing unrelated policy. Recipes separate those
graphs while keeping expensive infrastructure shared.

## When to Use

Use recipes when one deployment serves consumers with different policies, for
example privacy-first, cost-optimized, or team-specific routing. They also let
you stage a new policy beside the current default and move clients by changing
only the requested model name.

Keep a single top-level `routing` block when one policy serves all traffic.
Existing single-profile configurations continue to work and are normalized as
the `default` recipe.

## Configuration

```yaml
routing:
  strategy: priority
  signals:
    keywords:
      - name: sensitive_input
        operator: OR
        keywords: ["internal ticket"]
  decisions:
    - name: protected_route
      rules:
        operator: AND
        conditions:
          - type: keyword
            name: sensitive_input
      modelRefs:
        - model: qwen3-8b

entrypoints:
  - model_names: ["vllm-sr/privacy"]
    recipe: privacy-first

recipes:
  - name: privacy-first
    description: Keep privacy-sensitive prompts on the local model.
    routing:
      strategy: confidence
      signals:
        keywords:
          # Local names may repeat across recipes. This definition is unrelated
          # to routing.signals.keywords[sensitive_input] above.
          - name: sensitive_input
            operator: OR
            keywords: ["ssn", "passport number"]
        pii:
          - name: restricted_pii
            threshold: 0.5
      decisions:
        - name: protected_route
          rules:
            operator: OR
            conditions:
              - type: keyword
                name: sensitive_input
              - type: pii
                name: restricted_pii
          modelRefs:
            - model: qwen3-8b
              use_reasoning: false
```

The two `sensitive_input` signals and two `protected_route` decisions are valid:
their fully qualified identities are `(default, sensitive_input)` and
`(privacy-first, sensitive_input)`, and likewise for the decisions. References
remain local, so a decision in `privacy-first` cannot reference a signal or
projection declared only by `default`.

## Request resolution

| Requested model | Behavior |
| --- | --- |
| An `entrypoints[].model_names` value | Select that entrypoint's recipe, evaluate only that recipe, then rewrite the request to the selected backend model. |
| `vllm-sr/auto`, `auto`, or another configured auto alias | Select the `default` recipe. |
| A direct ReMoM, Fusion, or Flow virtual slug | Select the looper in the `default` recipe. |
| A concrete backend model or LoRA name | Pass through directly. Do not evaluate recipe signals, decisions, route-local plugins, cache, learning, or session routing state. |

If no decision in the selected recipe matches, the router uses
`providers.defaults.default_model`. The virtual entrypoint name never reaches a
backend.

Entrypoints appear in `/v1/models` with stable routing metadata. Routed
responses expose `x-vsr-selected-recipe`, and Router Replay/Insights can be
filtered by recipe.

## Isolation and sharing

Recipe-local:

- signal definitions, including PII, jailbreak, and authorization role bindings
- projection partitions, scores, mappings, and their dependency graph
- decisions, priorities, `strategy`, decision algorithms, and route-local plugins
- response-cache namespaces, replay identities, learning/session state,
  handoff penalties, and routing metric labels

Shared infrastructure:

- `routing.modelCards` and `providers` model/backend bindings
- model files, embedding/classifier runtimes, external services, and stores
- transport/service configuration such as identity header extraction, replay
  storage, and observability backends
- router defaults inherited by a recipe when the recipe does not override a
  supported field

Shared infrastructure does not make policy global. For example,
`global.services.authz` configures identity and credential resolution, while
the authorization policy facts used for routing live in each recipe's
`routing.signals.role_bindings`. PII and jailbreak model assets may be shared,
but their rule declarations and thresholds are recipe-local.

Metadata and learned classifier declarations follow the same isolation rule:

```yaml
recipes:
  - name: risk-aware
    routing:
      signals:
        metadata:
          - name: premium_tenant
            key: tier
            predicate:
              in: [gold, platinum]
        classifiers:
          - name: request_risk
            type: local
            model_path: models/request-risk
            labels: [SAFE, RISKY]
            use_cpu: true
      decisions:
        - name: guarded_route
          description: Route risky premium traffic to the guarded backend.
          priority: 50
          rules:
            operator: AND
            conditions:
              - type: metadata
                name: premium_tenant
              - type: classifier
                name: request_risk
                label: RISKY
                predicate:
                  gte: 0.5
                on_error: no_match
          modelRefs:
            - model: qwen3-8b
```

Another recipe may reuse `premium_tenant` or `request_risk` with a different
definition; references resolve only within the owning recipe.

## Validation

Configuration loading rejects:

- duplicate recipe names or a `default` recipe conflicting with a non-empty
  top-level routing profile
- entrypoints referencing unknown recipes, empty entrypoints, duplicate virtual
  model claims, or names colliding with concrete models, LoRAs, auto aliases, or
  looper slugs
- recipe-owned `modelCards`
- duplicate signal, projection, or decision names within one recipe
- any decision or projection reference that cannot be resolved inside its own
  recipe

The same local name in different recipes is intentionally allowed.

## Lifecycle management

Use the recipe endpoints when an operator needs to stage, replace, or retire a
single policy without rewriting the rest of the canonical document. Reads
return an `ETag`; every mutation requires the matching value in `If-Match`.
That optimistic-concurrency contract prevents a stale dashboard tab or
automation job from overwriting a newer config.

```bash
# Read the active collection and retain the ETag response header.
curl -i http://localhost:8080/config/router/recipes

# Validate the exact mutation without writing, backing up, or reloading.
curl -X POST http://localhost:8080/config/router/recipes/validate \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "privacy-first",
    "routing": {"strategy": "priority", "decisions": []},
    "entrypoints": ["vllm-sr/privacy"]
  }'

# Create or replace atomically. Replace <etag> with the current response value.
curl -X PUT http://localhost:8080/config/router/recipes/privacy-first \
  -H 'Content-Type: application/json' \
  -H 'If-Match: <etag>' \
  -d '{
    "routing": {"strategy": "priority", "decisions": []},
    "entrypoints": ["vllm-sr/privacy"]
  }'
```

A successful mutation validates the complete document, creates a backup,
writes atomically, reloads Router and Envoy, and publishes a new `ETag`. A
missing precondition returns `428`; a stale `ETag` returns `412`. The API
rejects deletion of `default` and deletion of any named recipe still referenced
by an entrypoint. Detach the entrypoint in a guarded `PUT`, then issue the
guarded `DELETE`.

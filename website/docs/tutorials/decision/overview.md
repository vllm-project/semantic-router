# Decisions

## Overview

Signals tell the Router what it detected. Decisions turn those detections into
a route policy:

- which route matched
- which models are candidates
- whether reasoning is enabled
- which plugins run after the route is chosen

## Key Advantages

- Keeps route policy readable even when multiple signals must cooperate.
- Makes boolean logic explicit and reviewable.
- Separates route matching from deployment bindings, algorithms, and plugins.

## What Problem Does It Solve?

Without a decision layer, signal outputs do not tell the router how to react. Teams end up scattering route logic across ad hoc if-statements, model defaults, and plugin wiring.

Decisions solve that by turning named signals into clear route policies with stable priorities and candidate models.

## When to Use

Use a decision when:

- a route should activate from one or more signals
- the same model policy should be reused across several signal combinations
- route priority matters
- plugins or algorithms should attach to a matched route instead of the whole router

## Configuration

In v0.3, decisions live under `routing.decisions`:

```yaml
routing:
  decisions:
    - name: business_route
      description: Route business requests to the business model.
      priority: 110
      rules:
        operator: AND
        conditions:
          - type: domain
            name: business
      modelRefs:
        - model: qwen2.5:3b
          use_reasoning: false
```

Classifier failures evaluate as `Unknown`, not `False`. `NOT Unknown` remains
`Unknown`; `False AND Unknown` is `False`, and `True OR Unknown` is `True`.
When the final result is still unknown, `rules.on_unknown` chooses `no_match`,
`match`, or `fail_request`. If omitted, existing generic-classifier
`on_error` and prompt-guard `on_error` behavior is retained.

Decision matching stays separate from:

- `providers.models[]`, which carries deployment bindings
- `decision.algorithm`, which chooses among multiple candidate models
- `decision.plugins`, which post-processes a matched route

Choose the smallest shape that expresses the policy clearly:

| Decision shape | Best for | Guide |
|----------------|----------|-------|
| Single condition | One decisive signal | [Single Condition](./single) |
| `AND` | Several conditions that must all match | [AND Decisions](./and) |
| `OR` | One route shared by several alternative conditions | [OR Decisions](./or) |
| `NOT` | An explicit exclusion or safety guard | [NOT Decisions](./not) |
| Composite | Nested combinations of `AND`, `OR`, and `NOT` | [Composite Decisions](./composite) |
| Retention directives | Cache or session side effects after a decision matches | [Retention Directives](./retention) |

Add [Algorithm](../algorithm/overview) when `modelRefs` contains more than one candidate, and add [Plugin](../plugin/overview) when the route needs post-selection behavior.

## Operational Boundaries

- Every leaf must reference a signal or projection output declared in the same
  recipe.
- Higher `priority` wins when more than one decision matches. Keep an explicit
  unconditional fallback or configure `providers.defaults.default_model`.
- Decision names and route diagnostics can become operational metadata; avoid
  secrets or personal identifiers in names and descriptions.
- Boolean logic is policy, not authentication. Use trusted identity through
  the `authz` service and signal for access-sensitive routes.

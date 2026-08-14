# Decision Tutorials

## Overview

These tutorials mirror the maintained boolean examples under
`config/fragments/decision/`.

Signals tell the router what it detected. Decisions tell the router what to do with those detections:

- which route matched
- which models are candidates
- whether reasoning is enabled
- which plugins run after the route is chosen

## Key Advantages

- Keeps route policy readable even when multiple signals must cooperate.
- Makes boolean logic explicit and reviewable.
- Separates route matching from deployment bindings, algorithms, and plugins.
- Maps directly to reusable fragment directories under `config/fragments/decision/`.

## What Problem Does It Solve?

Without a decision layer, signal outputs do not tell the router how to react. Teams end up scattering route logic across ad hoc if-statements, model defaults, and plugin wiring.

Decisions solve that by turning named signals into clear route policies with stable priorities and candidate models.

## When to Use

Use `decision/` when:

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

Decision matching stays separate from:

- `providers.models[]`, which carries deployment bindings
- `decision.algorithm`, which chooses among multiple candidate models
- `decision.plugins`, which post-processes a matched route

Use the case-shape catalog below in the same order as the fragment tree:

| Decision shape | Fragment example | Best for | Tutorial |
|----------------|------------------|----------|----------|
| `single` | `config/fragments/decision/single/domain-business.yaml` | one decisive signal | [Single Condition](./single) |
| `and` | `config/fragments/decision/and/urgent-business.yaml` | multiple required signals | [AND Decisions](./and) |
| `or` | `config/fragments/decision/or/business-or-law.yaml` | shared route across alternatives | [OR Decisions](./or) |
| `not` | `config/fragments/decision/not/exclude-jailbreak.yaml` | explicit exclusion or safety guard | [NOT Decisions](./not) |
| `composite` | `config/fragments/decision/composite/priority-safe-escalation.yaml` | nested real-world policies | [Composite Decisions](./composite) |
| `retention` | `routing.decisions[].emits[]` | post-decision cache/session side effects | [Retention Directives](./retention) |

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
- Maintained examples:
  [`config/fragments/decision/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/decision).

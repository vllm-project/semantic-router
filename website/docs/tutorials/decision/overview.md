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

Decision matching stays separate from:

- `providers.models[]`, which carries deployment bindings
- `decision.algorithm`, which chooses among multiple candidate models
- `decision.plugins`, which post-processes a matched route

### Rule operators

A rule node's `operator` combines its `conditions`:

- `AND` — every condition must match.
- `OR` — at least one condition must match.
- `NOT` — negates a single nested condition.

The operator is matched case-insensitively (`and`, `or`, `not` all work) but is not trimmed, so it must carry no surrounding whitespace. Always write it explicitly: an omitted operator is evaluated as `OR` by the router while the CLI and the DSL emit `AND`, so the tree does not mean the same thing on every surface.

A decision matches unconditionally when its `rules` block is omitted entirely, or when the root is `AND` with no conditions.

The config loader rejects a rule tree the evaluator cannot honour, naming both the decision and the offending node (for example `decision 'business_route': rules.conditions[0]: ...`):

- an operator outside `AND`/`OR`/`NOT`, which would otherwise fall through to the evaluator's default branch and widen the rule to `OR`
- a `NOT` with any number of children other than one, which would otherwise never match
- a node that is both a leaf (`type`/`name`) and a combination (`operator`/`conditions`), which would otherwise drop its conditions
- a node that carries leaf fields without a `type`, which would otherwise either never match or, worse, match every request
- a leaf without a `name`, which references no signal at all
- a leaf that declares `conditions`, which would otherwise be ignored
- a combination with no conditions anywhere but the root, which would otherwise never match under `OR` and match everything under `AND`

> **Note:** this operator set is intentionally distinct from keyword-signal operators (`routing.signals.keywords[].operator`, which also accepts `NOR`). A decision rule only accepts `AND`, `OR`, and `NOT`.

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

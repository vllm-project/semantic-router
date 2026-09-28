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

Each `rules` node is either a leaf (`type` and `name`) or a combination
(`operator` and `conditions`). The operator must be `AND`, `OR`, or `NOT`;
case and surrounding whitespace are normalized, and an omitted operator on a
node with conditions means `AND`. `NOT` is strictly unary and takes exactly one
child condition; nest `NOT` around `OR` or `AND` for NOR or NAND. Config
validation rejects any other operator, a `NOT` with zero or several children,
a node that mixes leaf and combination fields, and a childless combination
anywhere except a root `AND`, which is the explicit match-all form. Errors name
the decision and the node path, for example
`decision "billing": rules.conditions[1]: NOT requires exactly one child condition, got 2`.

Classifier failures evaluate as `Unknown`, not `False`. `NOT Unknown` remains
`Unknown`; `False AND Unknown` is `False`, and `True OR Unknown` is `True`.
When the final result is still unknown, `rules.on_unknown` chooses `no_match`,
`match`, or `fail_request`. If omitted, existing generic-classifier
`on_error` and prompt-guard `on_error` behavior is retained, and the router
warns at startup about classifier conditions that set neither. Applied policies
appear in the `x-vsr-applied-unknown-policy` response header and the
`llm_decision_unknown_total{decision, policy}` metric; the `fail_request` 503
message names the fix.

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

## Selection Order

Matching and ranking are separate steps. Every decision whose rules evaluate to
true becomes a candidate, and ranking then picks one of them.

`decision.tier` is the hard precedence boundary: a lower tier wins, and a
decision that omits `tier` is tier 0, so it ranks ahead of every decision that
sets one. Tier is compared first whenever one matched decision carries a tier
above 0. `routing.strategy` then decides how the decisions inside that tier are
ordered, and it means the same thing where no decision sets a tier at all.

| `routing.strategy` | Keys, in order |
| --- | --- |
| `priority` (the default) | tier ascending, catch-all last, priority descending, confidence descending, name ascending |
| `confidence` | tier ascending, catch-all last, confidence descending, priority descending, name ascending |

Confidence ranks a comparable pool only. A pool is one tier under tiered
ranking and the whole candidate set otherwise. Every leaf is either policy or
evidence. Keyword rules, `NOT` guards, predicates, conversation and metadata
rules, and projection outputs restate policy the operator already wrote, so
they decide whether a decision matches and carry no weight in its confidence.
Only a signal that reports a measurement is evidence, and selection ranks on
it only where the quantity it reports is established.

| Evidence signal | Score kind |
| --- | --- |
| `domain`, `classifier`, `safety`, `preference` | classifier probability |
| `embedding`, `reask`, `kb` | vector similarity |
| `complexity`, `jailbreak` | depends on the configured backend |

A decision is comparable when exactly one evidence leaf produced its score,
since a mean over several leaves falls as a decision gains evidence. A pool
ranks by confidence when every member that is not a catch-all is comparable
and all of those scores are the same kind. Otherwise the pool ranks by
priority, and the router warns at startup about a pool that mixes kinds. Three
cases are never comparable: a score whose kind depends on the backend that
produced it, a tree that reports one kind or another depending on which branch
of an `OR` matched, and a match an `on_error` policy manufactured. Inside an
`OR`, a branch that reported a score outranks one that did not, so an extra
matching gate adds support without removing evidence. Name ascending is the
final tie-break, so ranking never depends on map or file order.

The eval API reports how one request was ranked under `decision_ranking`: the
strategy that ran, the tier the winner came from, whether that pool was
comparable and which decision made it incomparable, and the key that separated
the winner from the decision behind it.

A catch-all ranks after every real match under either strategy, whatever
priority it carries, so an unconditional fallback stays a fallback. The
[Decision Ranking Semantics](../../proposals/decision-ranking-semantics)
proposal records how this contract was settled.

## Operational Boundaries

- Every leaf must reference a signal or projection output declared in the same
  recipe.
- Ranking follows [Selection Order](#selection-order). Keep an explicit
  unconditional fallback or configure `providers.defaults.model`.
- Decision names and route diagnostics can become operational metadata; avoid
  secrets or personal identifiers in names and descriptions.
- Boolean logic is policy, not authentication. Use trusted identity through
  the `authz` service and signal for access-sensitive routes.

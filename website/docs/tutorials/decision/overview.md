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

Without a decision layer, signal outputs do not tell the router how to react.
Teams end up scattering route logic across ad hoc if-statements, model
defaults, and plugin wiring.

Decisions solve that by turning named signals into clear route policies with
stable priorities and candidate models.

## When to Use

Use a decision when:

- a route should activate from one or more signals
- the same model policy should be reused across several signal combinations
- route priority matters
- plugins or algorithms should attach to a matched route instead of the whole router

## Configuration

Decisions live under `recipes[].decisions`:

```yaml
recipes:
  - name: default
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

### Rule trees

- **Nodes.** Each `rules` node is either a leaf (`type` and `name`) or a
  combination (`operator` and `conditions`).
- **Operators.** `AND`, `OR` or `NOT`; case and surrounding whitespace are
  normalized, and a node with conditions but no operator means `AND`.
- **`NOT` is unary.** It takes exactly one child; nest it around `OR` or `AND`
  for NOR or NAND.
- **Match-all.** A childless combination is valid only as a root `AND`, the
  explicit match-all form.

Validation rejects anything else, such as a node that mixes leaf and
combination fields. Each error names the decision and the node path:

```text
decision "billing": rules.conditions[1]: NOT requires exactly one child condition, got 2
```

### Unknown signals

A signal that failed or answered late is unknown. By default, its condition
evaluates as `False`. Two settings change that:

| Setting | Effect |
| --- | --- |
| `rules.on_unknown` | Keeps unknown conditions `Unknown`: `NOT Unknown` stays `Unknown`, `False AND Unknown` is `False`, and `True OR Unknown` is `True`. A result that is still unknown resolves to `no_match`, `match` or `fail_request`. |
| `on_error` on a `systemone` condition | Resolves that one condition to `match` or `no_match`. It can't be combined with `rules.on_unknown`. |

- The Router warns at startup about a `systemone` condition that sets neither.
- Applied policies appear in the `on_unknown` member of `x-vsr-trace` and in
  the `llm_decision_unknown_total{decision, policy}` metric.
- `fail_request` returns a 503 with the code
  [`decision_unresolved`](../../api/router.md#routing-errors), and the
  Router's log line names the fix.

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
true becomes a candidate, and ranking then picks one of them:

1. **Tier first.** `decision.tier` is a hard precedence boundary: a lower tier
   wins. A decision without `tier` is tier 0, ahead of every decision that sets
   one. Tier counts whenever a matched decision has a tier above 0.
2. **Then the recipe's strategy** orders the decisions inside that tier, or
   all of them when no decision sets a tier.
3. **Catch-alls last.** A catch-all ranks after every real match under either
   strategy, whatever its priority, so an unconditional fallback stays a
   fallback.

| `recipes[].strategy` | Keys, in order |
| --- | --- |
| `priority` (the default) | tier ascending, catch-all last, priority descending, confidence descending, name ascending |
| `confidence` | tier ascending, catch-all last, confidence descending, priority descending, name ascending |

Name ascending is the final tie-break, so ranking never depends on map or file
order.

### Confidence

Confidence ranks only a comparable pool: one tier under tiered ranking, and
the whole candidate set otherwise.

- **Evidence comes from model-judged leaves.** `systemone`, `domain`,
  `safety`, `jailbreak`, `complexity` and `turn_relation` leaves report the
  probability of their answer.
- **Policy leaves carry no weight.** Keyword rules, `NOT` guards, predicates,
  conversation and metadata rules, and projection outputs decide whether a
  decision matches, but add nothing to its confidence.
- **One score per decision.** A decision is comparable when exactly one
  evidence leaf produced its score, because a mean over several leaves falls
  as a decision gains evidence. A match that an `on_error` policy manufactured
  is never comparable.
- **Otherwise, priority.** A pool ranks by confidence only when every member
  that is not a catch-all is comparable.

Inside an `OR`, a branch that reported a score outranks one that did not, so
an extra matching gate adds support without removing evidence.

### Read the ranking

The eval API reports how one request was ranked under `decision_ranking`:

- the strategy that ran, and the tier the winner came from;
- whether that pool was comparable, and which decision made it incomparable;
- the decision behind the winner, as `runner_up`, and the key that separated
  them;
- a `reason` stating the values compared, such as `priority 150 > 100`.

The [Decision Ranking Semantics](../../proposals/decision-ranking-semantics)
proposal explains the design of this contract.

## Operational Boundaries

- Every leaf must reference a signal or projection output declared in the same
  recipe.
- Ranking follows [Selection Order](#selection-order). Keep an explicit
  unconditional fallback or configure `providers.defaults.model`.
- Decision names and route diagnostics can become operational metadata; avoid
  secrets or personal identifiers in names and descriptions.
- Boolean logic is policy, not authentication. Use trusted identity through
  the `authz` service and signal for access-sensitive routes.

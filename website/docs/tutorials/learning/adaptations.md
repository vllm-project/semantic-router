# Adaptation

## Overview

Adaptation is online model-choice learning. It runs after the matched decision
and base selector, then proposes a model from an allowed candidate set.

## Key Advantages

- Improves model choice without rewriting semantic decisions on the request path.
- Keeps the default search space narrow with `candidate_set: decision`.
- Can share evidence across a route tier when `candidate_set: tier` is enabled.
- Uses protection before random sampling and before final model switches.
- Writes compact headers and detailed replay diagnostics for offline analysis.

## What Problem Does It Solve?

Static recipes encode what the operator knows at deploy time. In production,
the router also observes model fit, overuse, provider failures, latency, cache
reuse, and effective cost. Adaptation turns that bounded evidence into an
online model proposal while the recipe remains the policy source of truth.

## When to Use

- A decision has multiple candidate models and runtime outcomes can improve the
  choice.
- Related decisions share a tier and should learn from each other's model
  evidence.
- You want online exploration, but only when protection says the current agent
  state is safe to explore.
- You want offline evals to seed model experience without changing the recipe
  immediately.

## Configuration

The first strategy is `routing_sampling`:

```yaml
global:
  router:
    learning:
      enabled: true
      adaptation:
        enabled: true
        strategy: routing_sampling
        candidate_set: decision
```

## Candidate Sets

| Value | Candidate models |
| --- | --- |
| `decision` | Models from the matched decision's `modelRefs`. |
| `tier` | Union of `modelRefs` from decisions with the same `decision.tier`. |
| `global` | All deployed models in the recipe's model/provider inventory. |

`decision` is the safest default. `tier` lets related routes share candidates.
`global` is broadest and can propose deployed models that do not appear in the
matched decision's `modelRefs`, so it uses stricter cost and reliability guards.

## Routing Sampling

`routing_sampling` scores each candidate from model experience:

- offline or neutral quality seed
- `good_fit`, `underpowered`, `overprovisioned`, and `failed` outcomes
- latency evidence
- cache reuse evidence
- effective input cost
- reliability evidence

When protection allows exploration, the strategy can sample from the candidate
posterior. When protection suppresses exploration, it scores by deterministic
posterior mean.

Protection still gets the final say. A sampled or mean-selected candidate must
clear the switch guard before it becomes the final model.

## Diagnostics

Headers stay compact:

```http
x-vsr-learning-methods: adaptation,protection
x-vsr-learning-actions: adaptation=keep_base,protection=hold_current
x-vsr-learning-reasons: adaptation=base_best,protection=cache_cost_high
```

Router Replay stores candidate scores, posterior mean, sampled value, base
model, proposal model, final model, candidate set, strategy, and reason.

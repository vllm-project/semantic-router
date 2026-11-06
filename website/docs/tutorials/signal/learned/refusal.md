# Refusal Signal

## Overview

`refusal` asks the decision model whether the model's answer declines the
request. It reads the answer, so a refusal rule exists only once the model has
answered: a response plugin reads it, never a decision's rules.

It maps to config/fragments/signal/refusal and is declared under
`routing.signals.refusal`.

## Key Advantages

- Detects a declined answer without a separate classifier.
- Uses the same decision model and the same call as the other response-stage
  signals.
- A predicate sets the operating point when the default does not fit.

## What Problem Does It Solve?

An answer that refuses a benign request is a failed answer even when the
request routed correctly. `refusal` names that outcome so a response plugin
can act on it.

## When to Use

Use `refusal` when:

- a response plugin should react to answers that decline the request
- you measure how often a route's model refuses

## Configuration

```yaml
routing:
  signals:
    refusal:
      - name: declined
        description: The answer declines the request or refuses to help.
        predicate:
          gte: 0.5
```

| Field | Meaning |
| --- | --- |
| `name` | The rule's name |
| `description` | What the rule detects |
| `predicate` | The bound on the probability of the `refusal` option; without one, the threshold calibrated for the answering model, else 0.5 |
| `deployment` | The deployment that answers the rule; else `model_bindings.refusal`, else the decision model |
| `timeout_ms` | Bound of the question |

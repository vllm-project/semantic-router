# Refusal Signal

## Overview

`refusal` asks the decision model whether the model's answer declines the request.
It asks the `refusal` [preset](../../../system-one/reference/presets#preset-catalog),
"Did the assistant refuse?", with the options `compliance` and `refusal`, at
the response stage. It reads the answer, so a refusal rule exists only once
the model has answered: a response plugin reads it, never a decision's rules.

## Key Advantages

- Detects a declined answer without a separate classifier.
- Uses the same decision model, and the same call, as the other
  response-stage signals.
- A predicate sets the operating point when the default does not fit.

## What Problem Does It Solve?

An answer that refuses a benign request is a failed answer even when the
request routed correctly. `refusal` names that outcome so a response plugin
can act on it. [`escalate`](../../algorithm/looper/escalate) asks the same
question to move on from a candidate that refused (`accept.refusal: false`).

## When to Use

Use `refusal` when:

- a [`response_guard`](../../plugin/response-guard) should annotate or block
  answers that decline the request;
- you measure how often a route's model refuses.

## Configuration

```yaml
recipes:
  - name: default
    signals:
      refusal:
        - name: declined
          description: The answer declines the request or refuses to help.
          predicate:
            gte: 0.5
    decisions:
      - name: assistant
        priority: 100
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: answer-model
        plugins:
          - type: response_guard
            configuration:
              rules:
                - when: {type: refusal, name: declined}
                  action: annotate
```

| Field | Meaning |
| --- | --- |
| `name` | The rule's name, which guard rules use as `{type: refusal, name: ...}` |
| `description` | What the rule detects |
| `predicate` | The least probability of `refusal`; by default the threshold calibrated for the answering model, else 0.5 |
| `deployment` | The deployment that answers the rule; else the `refusal` binding, else the decision model |
| `timeout_ms` | The answer's deadline |

The question sends the request and the answer to the deployment. vela3-fast
and the Vela 3.0 d3 sizes have a trained form of it; every other model
answers it zero-shot. See a complete example:
[`config/fragments/signal/refusal/refusal.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/refusal/refusal.yaml).

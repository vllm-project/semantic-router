# Complexity Signal

## Overview

`complexity` rates how difficult a request is to answer well: easy, medium or hard.
It asks the `complexity`
[preset](../../../system-one/reference#preset-catalog), "How difficult is
this request to answer well?", over `easy` (straightforward retrieval,
rewriting or formatting), `medium` (several steps of reasoning, or a moderate
amount of code or analysis) and `hard` (advanced multi-step reasoning or
specialized expertise).

## Key Advantages

- Sends hard requests to a reasoning model and easy ones to a fast one.
- Needs no example banks and no thresholds to fit.
- One question per request answers every complexity rule.

## What Problem Does It Solve?

Routing every request to the strongest model wastes it on easy work; routing
every request to the cheapest fails the hard ones. `complexity` tells them
apart before the model is chosen.

## When to Use

Use `complexity` when a decision should pick a stronger model, or turn on
reasoning, for hard requests only. Combine it with
[domains](./domain) for "hard math" or "easy code" routes. For a finer scale,
write a [`systemone`](../systemone) `score` question of your own.

## Configuration

```yaml
recipes:
  - name: default
    signals:
      domains:
        - name: math
      complexity:
        - name: difficulty
        - name: needs_reasoning
          description: Multi-step reasoning or synthesis-heavy prompts.
          labels: [hard]
    decisions:
      - name: hard-math
        priority: 200
        rules:
          operator: AND
          conditions:
            - {type: domain, name: math}
            - {type: complexity, name: difficulty, label: hard}
        modelRefs:
          - model: reasoner
      - name: easy-anything
        priority: 100
        rules:
          operator: AND
          conditions:
            - {type: complexity, name: difficulty, label: easy}
        modelRefs:
          - model: fast-model
```

| Field | Meaning |
| --- | --- |
| `name` | The rule's name, which `{type: complexity, name: ...}` conditions use |
| `labels` | The levels the rule matches, among `easy`, `medium` and `hard`; a rule without `labels` matches nothing itself, and its conditions name a level with `label` |
| `deployment`, `timeout_ms` | The deployment that answers, and the answer's deadline |

A rule matches when the level the model chooses is one of its `labels`; a
condition with a `label` matches when the model chose that level. A
condition can add a `predicate` on the level's probability.

### The model that answers

The decision model answers every complexity rule, unless the `complexity`
binding or the rule's `deployment` names another; see
[Choose a deployment](../../../system-one/deployments). No model has a
trained form of the question yet, so every model answers it zero-shot:
check it on your own traffic before it gates a costly route. It reads the
current user turn, truncated to the model's input.

Configurations written for earlier releases scored example banks against
thresholds (`hard`, `easy`, `threshold`, `composer`, ...) and named a level
as `difficulty:hard`. `vllm-sr config migrate` rewrites both: the rule keeps
its name, and conditions name the level with `label`. See a complete example:
[`config/fragments/signal/complexity/escalation.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/complexity/escalation.yaml).

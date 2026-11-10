# Domain Signal

## Overview

`domains` routes on the request's subject area, one of 14 that the decision model chooses.
Each rule names the subject areas it matches. It asks the `domain`
[preset](../../../system-one/reference/presets#preset-catalog): "Which subject area
is this request about?", over `biology`, `business`, `chemistry`,
`computer science`, `economics`, `engineering`, `health`, `history`, `law`,
`math`, `other`, `philosophy`, `physics` and `psychology`.

## Key Advantages

- Routes by topic without a keyword list per topic.
- One question per request answers every domain rule.
- Groups several subject areas under one route name.
- Feeds `static` model scores per domain.

## What Problem Does It Solve?

Routes are often organized around subject areas: a math model, a legal
reviewer, a coding assistant. Keyword lists for each are brittle and miss
paraphrases; the decision model reads the request and names its subject.

## When to Use

Use `domains` when:

- routes are organized around subject areas;
- lexical matching is too brittle for the topic boundary;
- the same topic boundary should feed several decisions.

For subjects outside the 14, write a [`systemone`](../systemone) `choice` of
your own.

## Configuration

```yaml
recipes:
  - name: default
    signals:
      domains:
        - name: math
          description: Mathematics and quantitative reasoning.
        - name: stem
          description: Science and engineering.
          labels: [physics, chemistry, biology, engineering]
        - name: computing
          labels: [computer science]
          predicate: {gte: 0.6}
    decisions:
      - name: stem-route
        priority: 100
        rules:
          operator: OR
          conditions:
            - {type: domain, name: math}
            - {type: domain, name: stem}
        modelRefs:
          - model: science-model
```

| Field | Meaning |
| --- | --- |
| `name` | The rule's name, which `{type: domain, name: ...}` conditions use |
| `labels` | The subject areas the rule matches; by default `[name]`, so a rule named after a subject area needs none |
| `predicate` | The least probability of the chosen subject area; by default the model's calibrated threshold (0.41 on vela3-fast), else 0.5 |
| `model_scores` | Per-candidate scores the [`static`](../../algorithm/selection/static) selector ranks by for requests in this domain |
| `deployment`, `timeout_ms` | The deployment that answers, and the answer's deadline |

A rule matches when the subject area the model chooses is one of its
`labels` and its probability meets the predicate. A rule whose name is not a
subject area must set `labels`; the configuration is refused otherwise, with
the closest subject area suggested. A request about none of your rules'
subject areas matches no domain rule: give a rule `labels: [other]` to catch
the requests the model files under `other`.

### The model that answers

The decision model answers every domain rule, unless the `domain` binding or
the rule's `deployment` names another; see
[Choose a deployment](../../../system-one/deploy/deployments). vela3-fast, the Vela
3.0 d3 sizes and Vela 2.0 have trained forms of the question; every other
System One model answers it zero-shot. The question reads the current user
turn, truncated to the model's input.

Through the Router, the routing preview (`POST /api/v1/routing/preview`)
reports the matched domain rules. See a complete example:
[`config/fragments/signal/domain/mmlu.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/domain/mmlu.yaml).

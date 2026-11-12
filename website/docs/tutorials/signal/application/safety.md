# Safety Signal

## Overview

`safety` detects harmful content in the request or the model's answer, optionally in named harm categories.
A rule without `labels` asks the `harm`
[preset](../../../system-one/reference/presets#preset-catalog), "Is this request
harmful?", with the options `safe` and `unsafe`. A rule with `labels` asks
which of the 34 [harm categories](../../../system-one/reference/presets#harm-categories)
apply, and matches the ones it lists.

## Key Advantages

- Separates content risk from prompt attacks, which `jailbreak` detects.
- Routes on specific harm categories, not only on safe or unsafe.
- Checks the request before routing and the answer after it.

## What Problem Does It Solve?

Content risk and prompt attacks need different labels. A benign discussion of
a harmful topic can be safe, while a harmful request need not try to override
any instruction. `safety` reads content risk; `jailbreak` reads the attack.

## When to Use

Use a rule without labels for a broad content policy, and a rule with labels
when only some categories should select a decision. Validate predicates on
your application's languages and input lengths before a rule refuses
traffic.

## Configuration

```yaml
recipes:
  - name: default
    signals:
      safety:
        - name: unsafe-content
        - name: violent-or-criminal
          labels:
            - Violence
            - Criminal Planning/Confessions
            - Guns and Illegal Weapons
            - S1
            - S2
        - name: unsafe-answer
          stage: response
          predicate: {gte: 0.6}
    decisions:
      - name: handle-content-risk
        priority: 300
        rules:
          operator: AND
          on_unknown: fail_request
          conditions:
            - {type: safety, name: unsafe-content}
        modelRefs:
          - model: safety-capable-model
```

| Field | Meaning |
| --- | --- |
| `name` | The rule's name, which `{type: safety, name: ...}` conditions and guard rules use |
| `labels` | Harm categories the rule matches; without them, the rule matches an unsafe request |
| `predicate` | Without labels, the least probability of `unsafe`: by default the model's calibrated threshold (0.52 on vela3-fast) at the request stage, else 0.5. With labels, the least probability of a listed category; without one, the model's own selection |
| `stage` | `request` (default) or `response`, which reads the model's answer |
| `min_coverage` | The share of a long input's windows, 0 to 1, that must have been read to decide on them; below it the rule is unknown. Off by default |
| `deployment`, `timeout_ms` | The deployment that answers, and the answer's deadline |

Every rule without labels shares one question per stage, and every rule with
labels shares the 34-category question; each applies its own labels and
predicate. A positive answer can come from a person seeking help in a
crisis: it does not imply malicious intent. Choose the response policy in
the decision, and keep prompt-attack decisions beside content-risk ones with
explicit priorities.

The question reads its whole input: an input longer than one of the
deployment's `scan_window`s is read in overlapping windows, the tail first,
then the head, then the middle by halves, and any window found unsafe, or in a
listed category, flags the input (see
[Long inputs](../../../system-one/deploy/performance.md#long-inputs)). Like
`jailbreak`, it also reads what `global.router.routing_input` removes from the
routing text. When the deadline ends before every window is read, the rule
decides on the windows read, once they are its `min_coverage` share of them if
it sets one; below that share it is unknown. A model that answers no window
leaves the rule unknown, and the decision's `rules.on_unknown` decides:
`fail_request` above returns HTTP 503.

### Harm categories

The categories are the 21 AEGIS 2.0 categories and the 13 MLCommons hazard
codes vela3-fast was trained on; the
[reference](../../../system-one/reference/presets#harm-categories) lists them. vela3-fast
answers all 34 with one trained set question. Every other model answers the
same labels zero-shot, one yes/no question per category, in the same call.
A topic mention is not automatically harmful; measure the categories you rely
on with your own data.

### At the response stage

A `stage: response` rule asks about the model's answer. Only a
[`response_guard`](../../plugin/response-guard) reads it; a decision that reads
one is refused at load.

### The model that answers

The decision model answers every safety rule, unless the `safety` binding or
the rule's `deployment` names another; see
[Choose a deployment](../../../system-one/deploy/deployments). vela3-fast, the Vela
3.0 d3 sizes and Vela 2.0 have trained forms of the safe or unsafe question.

See a complete example:
[`config/fragments/signal/safety/content-safety.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/safety/content-safety.yaml).

# Fact Check Signal

## Overview

`fact_check` detects requests whose answer relies on factual claims that should be verified.
It asks the `fact_check`
[preset](../../../system-one/reference#preset-catalog), "Does answering this
request require checking facts?", with the options `FACT_CHECK_NEEDED` and
`NO_FACT_CHECK_NEEDED`. It flags the need; it does not check any fact.

## Key Advantages

- Separates factual verification from domain routing.
- Lets decisions choose retrieval, a stronger model or an answer check for
  evidence-sensitive traffic.
- Keeps verification policy visible in the routing graph instead of inside
  later plugins.

## What Problem Does It Solve?

Not every request needs the same level of grounding. Treating all traffic the
same over-constrains creative requests and under-protects factual ones.
`fact_check` tells the two apart before routing.

## When to Use

Use `fact_check` when:

- factual requests need retrieval, stricter routing or a grounded answer
  check;
- creative or rewriting requests should skip expensive verification;
- a [`response_guard`](../../plugin/response-guard) rule should check
  grounding only when the request needed facts.

## Configuration

```yaml
recipes:
  - name: default
    signals:
      fact_check:
        - name: needs_fact_check
          description: Requests with factual claims that should be verified.
        - name: no_fact_check_needed
          description: Creative or rewriting requests.
          labels: [NO_FACT_CHECK_NEEDED]
    decisions:
      - name: grounded
        priority: 150
        rules:
          operator: AND
          conditions:
            - {type: fact_check, name: needs_fact_check}
        modelRefs:
          - model: grounded-model
```

| Field | Meaning |
| --- | --- |
| `name` | The rule's name, which `{type: fact_check, name: ...}` conditions use |
| `labels` | `[FACT_CHECK_NEEDED]` (the default) or `[NO_FACT_CHECK_NEEDED]` |
| `predicate` | The least probability of the rule's option; for `FACT_CHECK_NEEDED`, by default the model's calibrated threshold (0.9999 on vela3-fast), else 0.5 |
| `deployment`, `timeout_ms` | The deployment that answers, and the answer's deadline |

A rule's name does not choose its option: a rule without `labels` matches
`FACT_CHECK_NEEDED`. Each option is matched on its own probability, so a
request can match neither rule when the model is unsure.

vela3-fast's fact-check adapter puts its probabilities close to 0 and 1, so
its calibrated threshold is close to 1. On another model, write a
`predicate` you have measured.

### The model that answers

The decision model answers every fact-check rule, unless the `fact_check`
binding or the rule's `deployment` names another; see
[Choose a deployment](../../../system-one/deployments). vela3-fast, the Vela
3.0 d3 sizes and Vela 2.0 have trained forms of the question. It reads the
current user turn.

See a complete example:
[`config/fragments/signal/fact-check/needs-verification.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/fact-check/needs-verification.yaml).

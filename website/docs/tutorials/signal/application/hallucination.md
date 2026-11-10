# Hallucination Signal

## Overview

`hallucination` finds the claims in the model's answer that its grounding context does not support.
It asks the `hallucination`
[preset](../../../system-one/reference#preset-catalog), a span question
over the answer, with the label `unsupported`, at the response stage. The
grounding context is the documents RAG retrieved and the request's tool
results.

## Key Advantages

- Separates detecting an unsupported claim from what the Router does about it.
- Returns the spans of the unsupported claims, not only a verdict.
- Reports an answer that could not be checked as unknown, never as verified.
- Checks only the answers worth grounding, when combined with `fact_check`.

## What Problem Does It Solve?

A model given evidence can still answer past it: a wrong number, an invented
name, a detail the context never states. `hallucination` reads the answer
against that evidence and publishes what it finds as a signal that a
[`response_guard`](../../plugin/response-guard) acts on.

## When to Use

Use `hallucination` when:

- answers grounded in tool results or retrieved documents must be checked
  before they are trusted;
- a decision should annotate or block unsupported answers;
- Router Replay should record whether an answer was checked, and with what
  result.

## Configuration

```yaml
recipes:
  - name: default
    signals:
      fact_check:
        - name: needs_fact_check
      hallucination:
        - name: ungrounded
    decisions:
      - name: grounded-qa
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
                - when:
                    operator: AND
                    conditions:
                      - {type: fact_check, name: needs_fact_check}
                      - {type: hallucination, name: ungrounded}
                  action: annotate
                  details: true
              on_unknown: annotate
```

| Field | Meaning |
| --- | --- |
| `name` | The rule's name, which guard rules use as `{type: hallucination, name: ...}` |
| `predicate` | The least probability of an unsupported span; by default the model's own span thresholds decide |
| `deployment`, `timeout_ms` | The deployment that answers, and the answer's deadline |

A rule matches when the model finds an unsupported span. The guard rule above
checks only the answers to requests that need facts: an `AND` stops at a
false condition, so the question is not asked when `needs_fact_check` does
not match.

### Stage and outcomes

The question is asked after the model has answered, so a decision cannot
read it: a decision that does, directly or through a projection, is refused
at load. Only a `response_guard` reads it, from the recipe the request
resolved to. The rule is unknown when the request carried no grounding
context, when the model gave no answer, or when the response had no text to
check; the guard's `on_unknown` decides what an unknown rule does.

A streamed answer is checked once the stream ends. The guard above holds the
stream until then, because its rule can write a note into the answer
(`details: true`): the client receives the answer once it has been checked,
with the note at its head when the rule matched. A guard whose rules only flag
answers in `x-vsr-warnings`, or only record them, keeps the stream live and
records the result when it ends. See
[Streamed Answers](../../plugin/response-guard.md#streamed-answers).

### The model that answers

The decision model answers every hallucination rule, unless the
`hallucination` binding or the rule's `deployment` names another; see
[Choose a deployment](../../../system-one/deployments). The question needs
spans: vela3-fast (the default), the Vela 3.0 d3 sizes and Vela 2.0 answer it
with their own hallucination question. A rule that resolves to a deployment
without span answers, such as a d3 model, fails to load with
`span_required`.

The question sends the request, the answer and the grounding context to the
deployment. See a complete example:
[`config/fragments/signal/hallucination/grounded-answer.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/hallucination/grounded-answer.yaml).

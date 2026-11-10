# User Feedback Signal

## Overview

`user_feedbacks` reads what the current user turn says about the previous answer.
It asks the `feedback`
[preset](../../../system-one/reference#preset-catalog), "What feedback does
this user turn give about the previous answer?", over five options: `SAT`,
`NEED_CLARIFICATION`, `WRONG_ANSWER`, `WANT_DIFFERENT` and `NO_FEEDBACK`.

## Key Advantages

- Turns corrections and complaints into routing facts.
- Escalates to a stronger model only when the user says the answer failed.
- Tells feedback from an ordinary follow-up (`NO_FEEDBACK`).

## What Problem Does It Solve?

A user who says "that's wrong" or "that didn't work" is telling the Router
the last model was not good enough. Without a signal, the next turn goes to
the same model; with it, a decision can escalate.

## When to Use

Use `user_feedbacks` when:

- a correction should send the next turn to a stronger model;
- a request for clarification should get a more detailed answer;
- you record satisfaction or dissatisfaction per route.

## Configuration

```yaml
recipes:
  - name: default
    signals:
      user_feedbacks:
        - name: wrong_answer
          description: The user says the previous answer is wrong.
          labels: [WRONG_ANSWER]
        - name: need_clarification
          labels: [NEED_CLARIFICATION]
        - name: unhappy
          labels: [WRONG_ANSWER, WANT_DIFFERENT]
    decisions:
      - name: escalate-after-correction
        priority: 300
        rules:
          operator: AND
          conditions:
            - {type: user_feedback, name: wrong_answer}
        modelRefs:
          - model: strong-model
```

| Field | Meaning |
| --- | --- |
| `name` | The rule's name, which `{type: user_feedback, name: ...}` conditions use |
| `labels` | The feedback options the rule matches; a rule without `labels` matches nothing itself, and its conditions name one with `label` |
| `predicate` | The least probability of the chosen option; by default the model's calibrated threshold (0.47 on vela3-fast), else 0.5 |
| `deployment`, `timeout_ms` | The deployment that answers, and the answer's deadline |

A rule's name does not choose its feedback: name it in `labels`. A
`NO_FEEDBACK` answer matches only a rule that lists it.

The Router asks the question only for a user turn with text that follows an
assistant answer, and the question reads that answer with the turn. First
turns, tool-result continuations and assistant prefills do not ask it. A new question late in a conversation can still read
as feedback; validate that case before feedback changes the selected model.

### The model that answers

The decision model answers every feedback rule, unless the `user_feedback`
binding or the rule's `deployment` names another; see
[Choose a deployment](../../../system-one/deployments). vela3-fast, the Vela
3.0 d3 sizes and Vela 2.0 have trained forms of the question.

See a complete example:
[`config/fragments/signal/user-feedback/escalation.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/user-feedback/escalation.yaml).

# Action Signal

## Overview

`action` labels each request with the operation it asks for: `generate`,
`explain`, `fix`, `refactor`, `test`, or `other`. Declare the actions you route
on under `routing.signals.actions`, and reference one from a decision with
`type: action`.

The classifier reads the latest user message and matches English phrases that
name an action. It needs no model.

## Key Advantages

- Separates requests that share a topic but ask for different work, such as an
  explanation and an edit of the same code.
- Gives every request exactly one action, so decisions keyed on two different
  actions never match the same request.
- Adds no model inference, so it is cheap enough to combine with any other
  signal.

## What Problem Does It Solve?

Coding assistants send requests such as "explain this function", "fix this
failing test", and "rename these variables". The domain signal files all three
under the same subject. An explanation only reads code, while a fix or a
refactor edits files and often runs a tool loop, so a deployment may want a
small model for the first and a stronger one for the others.

## When to Use

Use `action` when:

- coding or agent traffic should split between read-only and editing work
- one topic needs different models for writing code, fixing it, and testing it
- follow-ups such as "yes, go ahead" should reach a cheaper route

## Configuration

```yaml
routing:
  signals:
    actions:
      - name: explain
        description: Describe code or answer a question without editing anything.
      - name: fix
        description: Repair a bug or a failing behavior.
```

Rule names come from a fixed vocabulary:

| Action | The request asks to |
| ------ | ------------------- |
| `generate` | write new code, configuration, or scripts |
| `explain` | describe code or answer a question without editing anything |
| `fix` | repair a bug or a failing behavior |
| `refactor` | change existing code without changing what it does |
| `test` | write or extend tests |
| `other` | none of the above, for example "yes, go ahead" or "review this PR" |

The classifier ignores text inside backticks, so code and logs pasted in a
fence do not count. It then finds every phrase that names an action. The
earliest phrase decides, and a longer phrase wins when two start at the same
place, so "write unit tests" is `test` rather than `generate`. A message with
no action phrase is `other`.

The routing preview reports the share of matched phrases that name the chosen
action under `signal_values`. "Fix this and add a test" is `fix` with
`action:fix` at `0.5`; a message that names one action scores `1`. The share is
a lexical measure, not a model probability, so the signal metrics mark
confidence as unavailable.

## Example Decision

```yaml
routing:
  decisions:
    - name: explain_code
      priority: 100
      rules:
        operator: AND
        conditions:
          - type: domain
            name: computer science
          - type: action
            name: explain
      modelRefs:
        - model: small-code-model
```

## Measuring Accuracy

[`bench/data/action_test_data.json`](https://github.com/vllm-project/semantic-router/blob/main/bench/data/action_test_data.json)
holds labeled prompts in the same `text` and `true_label` shape as the domain
set. It includes requests that ask for two actions, follow-ups with no action,
and pasted logs full of action words. When a prompt asks for two actions, the
label is the one that decides the work: an edit over an explanation, and
otherwise the first one requested.

Score a running Router that declares all six actions:

```bash
python tools/calibration/recipe/signal_labeled_set.py \
  --router-url http://localhost:8080 \
  --dataset bench/data/action_test_data.json
```

The script sends each prompt through the routing preview API and prints
per-action precision and recall, a confusion matrix, and the prompts it missed.

## Dependencies and Limitations

The phrases are English only, and only the latest user message is read, so a
follow-up keeps no memory of the action before it. A request that mixes actions
gets the earliest one: "explain this, then fix it" is `explain`. Logs pasted
outside a code fence can contain action words that decide the result. See a
complete example:
[`config/fragments/signal/action/coding-actions.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/action/coding-actions.yaml).

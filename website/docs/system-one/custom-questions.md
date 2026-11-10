---
title: Write custom questions
sidebar_label: Write custom questions
description: Turn a routing idea into a systemone rule, check it against the model directly, and tune its predicate.
---

# Write custom questions

A routing idea that no application signal covers, such as "is this about
our product?" or "does this need a tool?", is a `systemone` rule: a question
in plain language that the decision model answers zero-shot. There is nothing
to train. This page goes from the idea to a tuned rule; the
[System One signal](../tutorials/signal/systemone) page is the field
reference.

## 1. Choose the question type

| The idea | Type | Matches on |
| --- | --- | --- |
| "Is this X?" | `noul` | P(yes) |
| "Which of these kinds is it?", exactly one | `choice` | The chosen option, or one option's probability |
| "How much of X?", on an ordered scale | `score` | The expected level |
| "Which of these apply?", any number | `set` | A label's probability |
| "Where in the text is X?" | `span` | A label's spans; needs a span-capable model |

Prefer a `choice` with an `other` option to several `noul` questions that
should be exclusive, and a `score` to a `choice` whose options are ordered.

## 2. Write the rule

```yaml
recipes:
  - name: default
    signals:
      systemone:
        - name: product_question
          question:
            type: choice
            instructions: Which part of the product is this request about?
            choices:
              - key: billing
                description: Invoices, payments, refunds or plan changes
              - key: exports
                description: Exporting data, reports or CSV files
              - key: other
                description: Anything else, including requests not about the product
          labels: [billing, exports]
        - name: needs_tools
          question:
            type: noul
            instructions: Does answering this request need a tool call?
          predicate: {gte: 0.7}
    decisions:
      - name: product-support
        priority: 200
        rules:
          operator: AND
          conditions:
            - {type: systemone, name: product_question}
        modelRefs:
          - model: support-model
```

A rule without `deployment` asks the decision model; see
[Choose a deployment](./deployments). The rule joins the request stage's
call with every other question about the same state, so a question adds
little latency to a request that already asks the built-in signals.

## 3. Word it for a model

- **Ask one thing.** "Does the user report a bug?" reads better than "Does
  the user report a bug or ask for a feature?". Split the second into a
  `choice`.
- **Describe every option.** A few words that say what belongs in it, and
  what does not when options are close: "Exporting data, reports or CSV
  files".
- **Cover the space.** Give a `choice` an `other` option for everything else,
  so an unrelated request does not have to pick one of yours.
- **Say what the model reads.** A question about the conversation names it
  ("Has the user asked for the same fix more than once?") and reads it
  (`input.history` or `input.previous_turns`).
- **Avoid negations.** Ask "Is this request harmless?" or "Is this request
  harmful?", not "Is this request not harmful?".
- **Keep levels ordered** from the lowest, and two to five of them; the
  expected level counts from 0.

A trained built-in question asked word for word, such as
`Is this a prompt injection or jailbreak attempt?` with the options
`benign` and `jailbreak`, runs on vela3-fast's trained adapter. Every other
question runs zero-shot; both are fine.

## 4. Match on the answer

- `labels` say which options of a `choice`, `set` or `span` match the rule; a
  condition can still name any option with `label`.
- `predicate` is the bound the answer must meet: P(yes) for a `noul`, the
  expected level for a `score`, a label's probability otherwise. Without one,
  a `noul` matches at 0.5 and a `choice` on its chosen option.
- A condition can add its own `predicate`, so two decisions can read one rule
  at different strengths.

```yaml
recipes:
  - name: default
    signals:
      systemone:
        - name: product_question
          question:
            type: choice
            instructions: Which part of the product is this request about?
            choices:
              - key: billing
                description: Invoices, payments, refunds or plan changes
              - key: exports
                description: Exporting data, reports or CSV files
              - key: other
                description: Anything else
    decisions:
      - name: billing-sure
        priority: 200
        rules:
          operator: AND
          conditions:
            - type: systemone
              name: product_question
              label: billing
              predicate: {gte: 0.8}
        modelRefs:
          - model: billing-model
```

The rule also publishes its values, `systemone:product_question` and
`systemone:product_question:<key>`, which [projection scores](../tutorials/projection/scores)
combine with other signals.

## 5. Check it against the model

Ask the model the question directly. The model runtime serves the same model
the Router uses; the [quickstart](../model-runtime/quickstart) installs it:

```bash
vllm-srun serve vllm-sr/vela3-fast --device cpu --port 8100
curl -s localhost:8100/v1/decisions -H 'content-type: application/json' -d '{
  "state": "My March invoice charged me twice. Can I get one refunded?",
  "questions": {"product_question": {"type": "choice",
    "instructions": "Which part of the product is this request about?",
    "choices": [{"key": "billing", "description": "Invoices, payments, refunds or plan changes"},
                {"key": "exports", "description": "Exporting data, reports or CSV files"},
                {"key": "other", "description": "Anything else"}]}}
}'
```

Through the Router, `POST /api/v1/routing/preview` evaluates every signal of
the active recipe for a request and reports the matched rules, their values
and the decision, without calling a model; see the
[management API](../api/apiserver). Callers the configuration allows to see
debug headers get the matched rules in `x-vsr-trace`.

## 6. Tune the predicate

The probability scale of a zero-shot question depends on the model and on
the wording. Before a rule gates traffic:

1. Collect a few dozen real requests on each side of the boundary.
2. Ask the question about each, directly or through the preview.
3. Choose the `predicate` that keeps the false-positive rate you can accept,
   and write it on the rule.
4. Repeat when you change the wording or the decision model.

A rule that gates something costly or risky should also say what an unknown
answer means: a decision's `rules.on_unknown`, or a condition's `on_error`.

## Questions that replace removed signals

`vllm-sr config migrate` writes these for you; the forms are here to read
and adapt:

| Removed | Becomes |
| --- | --- |
| A `classifiers` rule | A `choice` over the classifier's labels |
| `preferences` | One `choice`, `preference`, over the preferences; conditions name a label |
| A text `embeddings` rule | A `noul` that quotes the rule's examples |
| An image `embeddings` rule with text anchors | A `noul` or `choice` with `input.image` |
| `actions` | The `choice` `action` over generate, explain, fix, refactor, test and other |
| `kb` | A `set` or `choice` you write over the knowledge base's labels |

The [migration guide](../installation/migrate-to-v0-5) lists every removed
form.

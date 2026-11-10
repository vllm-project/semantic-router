# System One Signal

## Overview

`systemone` asks the decision model a typed question you write, and turns the answer into a routing fact.
You write the question in plain language: pick one of a few options
(`choice`), yes or no (`noul`), a level on a scale (`score`), which of these
labels apply (`set`), or where in the text each label is (`span`). It is the
primitive the [application signals](./overview#application-signals) compile
into, and the [System One](../../system-one/overview) chapter explains the
model behind it.

## Key Advantages

- A new question works as soon as you write it; there is no classifier to train.
- Answers come with probabilities, so routes can require a confident answer.
- A rule reads what its question needs: the current turn, the earlier turns,
  tool results, the request's images, or the model's answer.
- Questions about the same request share one call with the request's other
  questions to the same deployment.

## What Problem Does It Solve?

Fixed-label classifiers answer only the questions they were trained for. A
System One model answers "does this need step-by-step reasoning?" or "is this
about our product?" from the question text alone.

## When to Use

Use it for a judgment about the request that no
[application signal](./overview#application-signals) already makes. Prefer a
heuristic signal for structural facts: length, keywords, input media,
identity. [Write custom questions](../../system-one/use/custom-questions) walks
from an idea to a tuned rule.

## Configuration

```yaml
recipes:
  - name: default
    signals:
      systemone:
        - name: needs_tools
          question:
            type: noul
            instructions: Does answering this request need a tool call?
          predicate:
            gte: 0.7
        - name: request_kind
          question:
            type: choice
            instructions: What kind of request is this?
            choices:
              - key: code
                description: Writing, reviewing or debugging code
              - key: math
                description: Mathematics or quantitative reasoning
              - key: chat
                description: Anything else
          labels: [code, math]
        - name: difficulty
          deployment: scorer
          question:
            type: score
            instructions: How difficult is this request?
            levels: [Trivial, Moderate, Hard]
          predicate:
            gte: 1.5
    decisions:
      - name: hard-code
        priority: 200
        rules:
          operator: AND
          on_unknown: no_match
          conditions:
            - type: systemone
              name: request_kind
              label: code
            - type: systemone
              name: difficulty
        modelRefs:
          - model: large-coder
global:
  model_catalog:
    deployments:
      scorer:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Kai-0.6B
```

A rule without `deployment` asks the decision model,
`global.model_catalog.system.decision_model` (vela3-fast unless you choose
another); [Choose a deployment](../../system-one/deploy/deployments) covers local,
attached and remote models. The deployment must answer the rule's question
type, or the configuration is refused when it loads. vela3-fast answers
`choice`, `noul` and `span` questions, and `set` questions as one `noul` per
label, but no `score` question, so `difficulty` asks a Decision 2.0
deployment.

Configurations written for earlier releases name this signal `decision`
(`recipes[].signals.decision` and `type: decision` conditions). They still
load, with a `renamed_signal_type` warning, and `vllm-sr config migrate`
rewrites them to `systemone`.

### Rule fields

| Field | Meaning |
| --- | --- |
| `name` | The rule's name, which conditions and projections use. |
| `question` | The typed question: `type`, `instructions`, and its `choices`, `levels` or `labels`. |
| `labels` | The options a `choice`, `set` or `span` rule matches. A condition can still name any option. |
| `ignore_labels` | `set` and `span` only: labels that never count toward the rule. |
| `predicate` | The probability or level the answer must meet (`gt`, `gte`, `lt`, `lte`). |
| `deployment` | The deployment that answers; without one, the decision model. |
| `stage` | `request` (default) or `response`. |
| `input` | What the rule reads besides the current user turn; see [Read more than the current turn](#read-more-than-the-current-turn). |
| `coverage` | `truncate` (default) reads what fits the model's input; `full` reads all of it. |
| `on_unscanned` | `coverage: full` only: what an answer that could not read the whole input means: `match`, `no_match` or `unknown` (default). |
| `timeout_ms` | The answer's deadline, up to 60000 ms; else the deployment's, else 1000 ms. |

### What a rule matches

A condition `{type: systemone, name: <rule>}` matches when the rule does; a
condition with a `label` matches when that option or label does:

| Question type | The rule matches when | A condition's `label` matches when |
| --- | --- | --- |
| `noul` | the probability of yes meets `predicate` (default `gte: 0.5`) | (takes no label) |
| `score` | the expected level meets `predicate` (required; levels count from 0) | (takes no label) |
| `choice` | the chosen option is one of `labels`, and its probability meets `predicate` when one is set; a rule without `labels` matches only through a label | the label is the chosen option, and its probability meets `predicate` when one is set |
| `set` | a counted label matches | the label's probability meets `predicate`; without one, the model selected the label |
| `span` | a counted label matches | a span with the label has a probability that meets `predicate`; without one, the model found a span with the label |

A `set` or `span` rule counts `labels`, or every label of the question
without them, less `ignore_labels`. A condition may add its own `predicate`;
for a condition with a `label`, it reads that label's value. The
configuration is refused when a `label` or an entry of `labels` or
`ignore_labels` is not an option of the question, when a condition names an
ignored label, and when an unlabelled condition names a `choice` rule
without `labels`.

The rule publishes these values, which a condition's `predicate` and a
[projection score](../projection/scores) read:

| Question type | `systemone:<name>` | `systemone:<name>:<key>` |
| --- | --- | --- |
| `noul` | P(yes) | — |
| `score` | the expected level | — |
| `choice` | P(chosen) | P(key) for every option |
| `set` | the highest P among the counted labels | P(key) for every label |
| `span` | the highest probability among the counted labels | the label's most probable span (0 when none) |

A projection score reads them with `value_source: raw`: `name: <question>`
reads `systemone:<name>`, and `name: <question>:<key>` reads one option or
label of a `choice`, `set` or `span` question. The question is asked whenever
a used projection reads it:

```yaml
recipes:
  - name: default
    signals:
      systemone:
        - name: difficulty
          deployment: scorer
          question:
            type: score
            instructions: How much reasoning does a strong expert need to answer well?
            levels: [none, a little, multi-step, expert]
          predicate:
            gte: 2
        - name: needs
          question:
            type: set
            instructions: What does a good answer need?
            labels:
              - key: deliberation
                description: a derivation, proof or careful step-by-step check
              - key: tools
                description: calling external tools or functions
    projections:
      scores:
        - name: effort
          method: weighted_sum
          inputs:
            - type: systemone
              name: difficulty
              weight: 0.3
              value_source: raw
            - type: systemone
              name: needs:deliberation
              weight: 0.4
              value_source: raw
      mappings:
        - name: effort_band
          source: effort
          method: threshold_bands
          outputs:
            - name: effort_high
              gte: 0.9
    decisions:
      - name: deliberate
        priority: 200
        rules:
          operator: AND
          conditions:
            - type: projection
              name: effort_high
        modelRefs:
          - model: large-reasoner
global:
  model_catalog:
    deployments:
      scorer:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Kai-0.6B
```

### Read more than the current turn

A rule reads the current user turn. `input` adds named parts of the
conversation to what the model reads:

```yaml
recipes:
  - name: default
    signals:
      systemone:
        - name: escalation
          question:
            type: noul
            instructions: Has the user asked for the same fix more than once?
          input:
            history: true
            previous_turns: 6
            tool_results: true
```

- `history`: the earlier user turns.
- `previous_turns` (0 to 32): the last turns of the conversation, user and
  assistant, as a transcript.
- `tool_results`: the request's tool results.
- `turn: false`: leave the current user turn out, so a request-stage rule
  reads only the earlier user turns or the tool results it names, as its
  request. It needs `history` or `tool_results`, takes no `previous_turns`
  or media, and is not asked, so it does not match, while those hold no
  text.
- `image` and `video`: the current turn's images and videos. The deployment
  must read that input, or the configuration is refused with
  `input_modality_unsupported`. No deployment reads audio yet, so
  `input.audio` is always refused.
- `answer` and `context`: the model's answer and its grounding context, for
  a `stage: response` rule only.

Rules that read the same parts of the request share a call; when rules read
different parts, the call is a batch with an entry per text they read.

### Read the whole input

A rule reads what fits the model's input and drops the rest
(`coverage: truncate`). With `coverage: full` the model reads all of it, in
windows when it is longer than one input. When the answer could not cover the
whole input (the scan budget ran out, the model could not tell its coverage,
or the input exceeds the model's length), `on_unscanned` decides what the
rule reports: `match`, `no_match`, or `unknown` (default), which leaves the
decision's failure policy to decide:

```yaml
recipes:
  - name: default
    signals:
      systemone:
        - name: account_ids
          question:
            type: span
            instructions: Which spans are account or order numbers?
            labels:
              - key: account_number
                description: a customer account or order number
          coverage: full
          on_unscanned: match
```

### Questions about the answer

`stage: response` asks the question about the model's answer, with
`input.answer` and `input.context` naming what it reads. A response plugin
such as [`response_guard`](../plugin/response-guard) reads such a rule; a
decision's rules cannot, because a decision routes the request before an
answer exists, and the configuration is refused if one does.

### Set and span questions

A `set` question names labels and asks which apply; a `span` question asks
where in the request each label occurs. Both take `labels` instead of
`choices`, each with a `key` and an optional `description`, and conditions on
them name a label or, to match any counted label, none:

```yaml
recipes:
  - name: default
    signals:
      systemone:
        - name: support_topics
          question:
            type: set
            instructions: Which topics does the request mention?
            labels:
              - key: billing
                description: payments, invoices or refunds
              - key: shipping
                description: deliveries, tracking or returns
              - key: other
                description: anything else
          ignore_labels: [other]
        - name: account_ids
          question:
            type: span
            instructions: Which spans are account or order numbers?
            labels:
              - key: account_number
                description: a customer account or order number
    decisions:
      - name: billing-with-account
        priority: 200
        rules:
          operator: AND
          conditions:
            - type: systemone
              name: support_topics
              label: billing
            - type: systemone
              name: account_ids
        modelRefs:
          - model: support-model
```

- `threshold` (0 to 1) replaces the model's own decision threshold for the
  question. Without it the model applies its calibrated threshold, which is
  what decides `selected` labels and found spans.
- `head` (`span` only) names the span head that answers on models with two:
  `router` (trained on personal data, unsupported claims and toxic spans) or
  `broad` (open extraction). Without it the model chooses by its own rule.
  Only Vela 2.0 0.8B, 4B and 9B have a broad head; a rule that names
  `head: broad` on another built-in model (vela3-fast, the other Vela 3.0
  models, Vela 2.0 0.3B) fails to load with `span_head_unsupported`. A
  `systemone` deployment's heads are unknown at load; its API refuses a head
  it lacks.
- A rule's `predicate` replaces the model's selection: `gte: 0.8` on a `set`
  rule matches the labels whose probability is at least 0.8.
- A model without native `set` support answers a set as one `noul` question
  per label, in the same call.
- A `span` rule on a deployment that answers no span questions is refused
  with `span_required`. The default decision model returns spans; the d3
  family and Decision 1.0 and 2.0 do not.

### When the model gives no answer

While the model is loading, overloaded or slower than `timeout_ms`, the rule
is unknown. `rules.on_unknown` on the decision, or `on_error: match |
no_match` on a condition, decides what an unknown answer means. Truncating to
the deployment's scan budget shortens only what the rule reads; it does not
shorten the request sent to the selected backend.

To see which rules matched, use the routing preview
(`POST /api/v1/routing/preview`), or, for callers allowed debug headers, the
`signals` member of the `x-vsr-trace` response header, which lists each
matched rule as `systemone:<name>`.

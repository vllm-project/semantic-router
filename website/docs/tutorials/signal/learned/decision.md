# Decision Signal

## Overview

`decision` asks a decision model a typed question about the request and turns the answer into a routing fact.
You write the question in plain language: pick one of a few options
(`choice`), yes or no (`noul`), or a level on a scale (`score`). Models that
answer them, such as Vela 2.0, also take `set` questions (which of these
labels apply?) and `span` questions (where in the text is each label?). The
model runs in the [built-in model runtime](model-runtime/overview.md),
which the router starts for you.

## Key Advantages

- A new question works as soon as you write it; there is no classifier to train.
- Answers come with probabilities, so routes can require a confident answer.
- All questions of one request to the same model travel in one call,
  including the PII question when the [`pii` signal](tutorials/signal/learned/pii.md#vela-20) uses that model.
- A late or failed answer makes the signal unknown; the request still goes through.

## What Problem Does It Solve?

Fixed-label classifiers answer only the questions they were trained for. A
decision model answers "does this need step-by-step reasoning?" or "is this
about our product?" from the question text alone.

## When to Use

Use it for questions that need judgment about the whole request. Prefer
heuristic signals for structural facts (length, keywords, modality) and the
specialized learned signals (domain, PII, jailbreak) where they already answer
your question.

## Configuration

A question that names no `deployment` asks the Router's decision model,
`global.model_catalog.system.decision_model` (Vela 2.0 0.3B unless you
[choose a size](model-runtime/choose-a-model.md#choose-a-size)). It joins the
call that answers the built-in signals, so one model answers every question
the Router asks of a request in one call:

```yaml
routing:
  signals:
    decision:
      - name: needs_tools
        question:
          type: noul
          instructions: Does answering this request need a tool call?
        predicate:
          gte: 0.7
```

The binding uses `{deployment: primary}` and can select Vela or Decision
1.0/2.0. The model must support every requested question type. The
[`decision` selection algorithm](tutorials/algorithm/selection/decision.md)
uses the same default binding, so the resource can also choose a backend.

To ask another model, such as a Decision 2.0 model, name it as a
`model_runtime` deployment and give each question its `deployment`:

```yaml
global:
  model_catalog:
    deployments:
      decision-kai:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Kai-0.6B
        device: auto

routing:
  signals:
    decision:
      - name: needs_reasoning
        deployment: decision-kai
        question:
          type: noul
          instructions: Does answering this request need multi-step reasoning?
        predicate:
          gte: 0.7
        timeout_ms: 1000
      - name: request_kind
        deployment: decision-kai
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
      - name: difficulty
        deployment: decision-kai
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
          - type: decision
            name: request_kind
            label: code
          - type: decision
            name: needs_reasoning
      modelRefs:
        - model: large-coder
```

| Question type | Matches when | Value a route can read |
| --- | --- | --- |
| `noul` | the probability of yes meets `predicate` (default `gte: 0.5`) | `decision:<name>` = P(yes) |
| `score` | the expected level meets `predicate` (required; levels count from 0) | `decision:<name>` = expected level |
| `choice` | a condition's `label` is the chosen option, and its probability meets `predicate` when one is set | `decision:<name>:<key>` = P(key), `decision:<name>` = P(chosen) |
| `set` | the probability of the condition's `label` meets `predicate`; without one, the model selected the label | `decision:<name>:<key>` = P(key), `decision:<name>` = the highest P |
| `span` | a span with the condition's `label` has a probability that meets `predicate`; without one, the model found a span with the label | `decision:<name>:<key>` = the label's most probable span (0 when none), `decision:<name>` = the highest probability any word reached |

A condition may add its own `predicate`; for a `choice`, `set` or `span`
condition with a `label`, it reads that label's value.

A [projection score](tutorials/projection/scores.md) reads the same values
with `value_source: raw`: `name: <question>` reads `decision:<name>`, and
`name: <question>:<key>` reads one option or label of a `choice`, `set` or
`span` question. The question is asked whenever a used projection reads it:

```yaml
routing:
  signals:
    decision:
      - name: difficulty
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
          - type: decision
            name: difficulty
            weight: 0.3
            value_source: raw
          - type: decision
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
```

### Set and span questions

A `set` question names labels and asks which apply; a `span` question asks
where in the request each label occurs. Both take `labels` instead of
`choices`, each with a `key` and an optional `description`, and conditions on
them name a label:

```yaml
global:
  model_catalog:
    deployments:
      vela2:
        provider: model_runtime
        artifact: vllm-sr/Vela-2.0-0.3B
        device: cpu

routing:
  signals:
    decision:
      - name: support_topics
        deployment: vela2
        question:
          type: set
          instructions: Which topics does the request mention?
          labels:
            - key: billing
              description: payments, invoices or refunds
            - key: shipping
              description: deliveries, tracking or returns
      - name: account_ids
        deployment: vela2
        question:
          type: span
          instructions: Which spans are account or order numbers?
          labels:
            - key: account_number
              description: a customer account or order number
          head: router

  decisions:
    - name: billing-with-account
      priority: 200
      rules:
        operator: AND
        conditions:
          - type: decision
            name: support_topics
            label: billing
          - type: decision
            name: account_ids
            label: account_number
      modelRefs:
        - model: support-model
```

- `threshold` (0 to 1) replaces the model's own decision threshold for the
  question. Without it the model applies its calibrated threshold, which is
  what decides `selected` labels and found spans.
- `head` (`span` only) names the span head that answers on models with two:
  `router` (trained on PII, unsupported claims and toxic spans) or `broad`
  (open extraction). Without it the model chooses by its own rule.
- A rule's `predicate` replaces the model's selection: `gte: 0.8` on a `set`
  rule matches the labels whose probability is at least 0.8.
- The model answers each `set` label under `<name>.<label>` in the same call,
  so no other decision signal on the deployment may have that name.

Only models that declare these question types answer them. When the router
prepares its configuration, at startup or on a reload, it checks every used
`set` or `span` question against its deployment's model and fails with the
rule's name if the model answers only `choice`, `noul` and `score` (Decision
1.0 and 2.0).

While the model is loading, overloaded or slower than `timeout_ms`, the signal
is unknown. `rules.on_unknown` on the decision, or `on_error: match | no_match`
on a condition, decides what an unknown answer means. Every question a request
asks one deployment, the built-in signals' included, goes in one call; once it
is sent, each question waits for it as long as the latest of them, since the
request waits for that call anyway. Matched decision signals
are listed in the `x-vsr-matched-decision-model` response header.

To choose a model, size and hardware, or to run the model on your own GPU
server, see [Decision models](../../../model-runtime/guides/decisions.md).

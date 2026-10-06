# Decision Signal

## Overview

`decision` asks a decision model a typed question about the request and turns the answer into a routing fact.
You write the question in plain language: pick one of a few options
(`choice`), yes or no (`noul`), or a level on a scale (`score`). The model
runs in the [built-in model runtime](model-runtime/overview.md),
which the router starts for you.

## Key Advantages

- A new question works as soon as you write it; there is no classifier to train.
- Answers come with probabilities, so routes can require a confident answer.
- All decision questions of one request travel in one call to the model.
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

Name the model as a `model_runtime` deployment, then ask it questions:

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

A condition may add its own `predicate`; for a `choice` condition with a
`label`, it reads that option's probability.

While the model is loading, overloaded or slower than `timeout_ms`, the signal
is unknown. `rules.on_unknown` on the decision, or `on_error: match | no_match`
on a condition, decides what an unknown answer means. Matched decision signals
are listed in the `x-vsr-matched-decision-model` response header.

To choose a model, size and hardware, or to run the model on your own GPU
server, see [Decision models](../../../model-runtime/guides/decisions.md).

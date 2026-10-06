---
title: Decision models
description: Ask a decision model your own routing questions in plain language, and let it choose the model that answers.
---

# Decision models

A decision model answers questions you write in plain language about a
request, without training a classifier for each one. The router uses it in two
places:

- a [`decision` signal](tutorials/signal/learned/decision.md) asks a
  question and routes on the answer;
- the [`decision` selection algorithm](tutorials/algorithm/selection/decision.md)
  asks which of a route's models should answer.

## Kinds of questions

| Type | Asks | The answer |
| --- | --- | --- |
| `choice` | Which of these options fits? | The chosen option and the probability of each option |
| `noul` | Is this true? | The probability of yes |
| `score` | How much, on an ordered scale? | The expected level and the probability of each level |
| `set` | Which of these labels apply? (Vela 2.0) | Every label above its threshold |
| `span` | Where in the text is ...? (Vela 2.0) | Labelled spans of the text |

All questions of one request that go to the same model travel in one call and
are answered together.

## Ask a question

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

Write instructions the way you would ask a colleague, and describe each option
in a few words. Short, concrete options give the most reliable answers.

## Let it choose the model

```yaml
routing:
  decisions:
    - name: code-route
      priority: 100
      rules:
        operator: AND
        conditions:
          - type: decision
            name: request_kind
            label: code
      modelRefs:
        - model: qwen3-8b
        - model: qwen3-32b
      algorithm:
        type: decision
        decision:
          deployment: decision-kai
          instructions: Which model should answer this request?
          candidates:
            qwen3-8b: Fast general model for routine code
            qwen3-32b: Strong reasoning model for hard code
          timeout_ms: 1000
```

The model's probability for each candidate becomes its selection score. If the
model is not ready or answers late, the first model in `modelRefs` answers.

## Which decision model

Decision 2.0 is the default family; Kai-0.6B runs on a CPU and the larger
sizes are more accurate on a GPU. Decision 1.0 models answer the same
questions. Vela 2.0 also answers `set` and `span` questions and has ready-made
questions for PII, hallucination and toxicity; it is a private preview and
needs a Hugging Face token with access. See
[Choose a model](model-runtime/choose-a-model.md#decision-models).

## Check it

Ask the model the same question directly with `/v1/decisions`; see the
[Quickstart](model-runtime/quickstart.md#3-send-a-request). Through the router,
`x-vsr-matched-decision-model` lists the decision signals that matched and
`x-vsr-selected-model` the model the selector chose.

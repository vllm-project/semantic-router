# Decision Model Selection

## Overview

`decision` asks a decision model which of a routing decision's `modelRefs` should answer the request.
You describe each candidate in a sentence; the model reads the request and
gives every candidate a probability, and the most likely one answers.

## Key Advantages

- Chooses with a model that reads the whole request.
- Reuses the Router's decision model by default, so choosing loads no other model.
- Reports a probability per candidate, visible in selection traces.
- Falls back to the first `modelRef` whenever the model cannot answer in time.

## What Problem Does It Solve?

A fixed order ignores the request, and asking a chat model to pick needs a
generation round trip and output parsing. A decision model answers the same
question in one pass, with probabilities and no text generation.

## When to Use

Use it when a decision has two or more candidates whose strengths you can
describe in a sentence each, and the right choice depends on what the request
asks. Prefer `static` when the order never changes and `multi_factor` when the
choice is about cost, latency or load.

## Configuration

Without a `deployment`, the Router's decision model chooses
(`global.model_catalog.system.decision_model`, Vela 2.0 0.3B unless you
[choose a size](../../../model-runtime/choose-a-model.md#choose-a-size)). It is the
model that already answers the request's built-in signals and its `decision`
questions that name no deployment, so choosing the model needs no second copy
of it:

```yaml
global:
  model_catalog:
    deployments:
      primary:
        provider: model_runtime
        artifact: vllm-sr/Vela-2.0-4B
        device: rocm
    system:
      decision_model:
        deployment: primary

routing:
  signals:
    decision:
      - name: request_kind
        question:
          type: choice
          instructions: What kind of request is this?
          choices:
            - key: code
              description: Writing, reviewing or debugging code
            - key: chat
              description: Anything else
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
          use_reasoning: false
        - model: qwen3-32b
          use_reasoning: true
      algorithm:
        type: decision
        decision:
          instructions: Which model should answer this request?
          candidates:
            qwen3-8b: Fast general model for routine code
            qwen3-32b: Strong reasoning model for hard code
          timeout_ms: 1000
```

The selector asks after the decision matches, in its own call to the same
deployment. The default binding may select Vela or Decision 1.0/2.0;
the selected resource must support the choice task. An explicit deployment
on the selector overrides the default binding.

To let another model choose, such as a Decision 2.0 model, name a
`model_runtime` deployment:

```yaml alternative
global:
  model_catalog:
    deployments:
      decision-kai:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Kai-0.6B

routing:
  signals:
    decision:
      - name: request_kind
        deployment: decision-kai
        question:
          type: choice
          instructions: What kind of request is this?
          choices:
            - key: code
              description: Writing, reviewing or debugging code
            - key: chat
              description: Anything else
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
          use_reasoning: false
        - model: qwen3-32b
          use_reasoning: true
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

A named deployment must use `provider: model_runtime`. The decision needs 2–255
different `modelRefs`, and `candidates` may describe only those models; a
model without a description uses its configured description. When the model
is not ready, answers late, is overloaded or returns an invalid answer, the
router records a selection fallback and uses the first `modelRef`. The chosen
model is reported in the `x-vsr-selected-model` response header. Routing
Preview asks the decision model the same question and reports its choice as
the selected model, without calling a backend.

See [Decision models](../../../model-runtime/guides/decisions.md) to choose a
model and where it runs.

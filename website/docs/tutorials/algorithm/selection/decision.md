# Decision Model Selection

## Overview

`decision` asks a decision model which of a routing decision's `modelRefs`
should answer the request. The question is a Choice whose options are the
candidate model names, described by `candidates` or by the models'
configured descriptions. The model's probabilities become the candidates'
selection scores, and the most likely candidate is selected.

## Key Advantages

- chooses among candidates with a model that reads the whole request
- returns a probability per candidate, visible in selection traces
- falls back to the first `modelRef` whenever the runtime is not ready or answers late

## What Problem Does It Solve?

Static ordering ignores the request, and helper-LLM selection needs a
generation round trip with a structured-output parser. A decision model
answers the same question in one forward pass, with calibrated probabilities
and no text generation.

## When to Use

Use it when a decision has two or more candidates whose strengths can be
described in a sentence each, and the routing question depends on the
content of the request. Prefer `static` when the order never changes and
`multi_factor` when the choice is about cost, latency or load.

## Configuration

```yaml
global:
  model_catalog:
    deployments:
      decision-kai:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Kai-0.6B
        revision: 881bee413681d80ebeac86afcda8b4138dae516e

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

The deployment must use `provider: model_runtime`, the decision needs 2–255
unique `modelRefs`, and `candidates` may describe only those models. Any
failure (runtime not ready, timeout, overload, an invalid answer) is recorded
as a selection fallback and the first `modelRef` is used.

---
title: Hallucination checks
description: Check a model's answer against the context it was given and mark the parts the context does not support.
---

# Hallucination checks

Vela 1.0 Halu reads the context a request carried (tool results, retrieved
documents), the user's question and the model's answer, and marks the spans
of the answer that the context does not support. The
[`hallucination` plugin](tutorials/plugin/hallucination.md) then adds a
warning header, a note in the response body, or only records the result.

It checks support, not truth: an answer can be supported by wrong context, and
an answer with no context to check is reported as unverified instead.

## Turn it on

Declare the observation as a signal and enforce it with the plugin on the
routes that should be checked. Vela FactCheck decides first whether a request
makes claims worth checking.

```yaml
routing:
  signals:
    fact_check:
      - name: needs_fact_check
        description: Requests that make factual claims.
    hallucination:
      - name: ungrounded_claims
        description: Claims the context does not support.
  decisions:
    - name: grounded-answers
      priority: 100
      rules:
        operator: AND
        conditions:
          - type: fact_check
            name: needs_fact_check
      modelRefs:
        - model: answer-model
      plugins:
        - type: hallucination
          configuration:
            enabled: true
            hallucination_action: header
            unverified_factual_action: header
            include_hallucination_details: true
global:
  model_catalog:
    modules:
      hallucination_mitigation:
        enabled: true
```

The router runs FactCheck and Halu on the CPU. Halu reads up to 8,192 tokens of
context, question and answer together; longer context is shortened from its
end so the answer is always checked.

## Choose where it runs

The binding is `hallucination_detector` and reads text spans of the answer
(`token_spans.v1`):

```yaml
global:
  model_catalog:
    deployments:
      vela-halu:
        provider: model_runtime
        artifact: vllm-sr/Vela-1.0-Encoder-307M-Halu
        device: cpu
        input:
          max_tokens: 8192
          overflow: reject
    bindings:
      hallucination_detector:
        deployment: vela-halu
        contract: token_spans.v1
```

### On Vela 2.0

Vela 2.0 marks unsupported claims with its router span head, as a ready-made
question about the answer. Bind `hallucination_detector` to a Vela 2.0
deployment instead:

```yaml alternative
global:
  model_catalog:
    deployments:
      vela2:
        provider: model_runtime
        artifact: vllm-sr/Vela-2.0-0.3B
        device: cpu
    bindings:
      hallucination_detector:
        deployment: vela2
        contract: token_spans.v1
```

The model reads the whole answer and applies its own calibrated threshold, so
the detector's `threshold` applies to Vela 1.0 Halu only. The same
deployment can answer the request's [decision questions](model-runtime/guides/decisions.md) and
its [PII question](model-runtime/guides/pii.md#on-vela-20).

## Check it

```bash
vllm-sr serve vllm-sr/Vela-1.0-Encoder-307M-Halu --device cpu --port 8100
curl -s localhost:8100/v1/classify -H 'content-type: application/json' -d '{
  "input": [{"context": "The Eiffel Tower is 330 metres tall and stands in Paris.",
             "question": "How tall is the Eiffel Tower?",
             "answer": "The Eiffel Tower is 450 metres tall."}]
}'
```

The result lists the unsupported `spans` of the answer, here the height, with
their character offsets in the answer. Through the router, the response
carries the hallucination warning headers and `x-vsr-matched-hallucination`.

Earlier releases could add an NLI explanation per span. That explainer is
retired; see [Migrate](model-runtime/migrate.md#features-that-were-retired).

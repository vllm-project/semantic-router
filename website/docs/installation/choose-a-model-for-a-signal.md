---
title: Choose a model for a signal
sidebar_label: Choose a model for a signal
description: Which deployment answers each signal, how to change it globally, per recipe or per rule, and what the Router checks when it loads the configuration.
---

# Choose a model for a signal

Every model the Router asks is declared once, by name, under
`global.model_catalog.deployments`. Everything that uses a model writes that
name. This page shows where to write it. The
[model catalog reference](model-catalog.md) lists every deployment field.

## The rule

Which deployment answers a signal is settled at one of three levels. The most
specific level wins.

| Level | Where you write the deployment name | Applies to |
| --- | --- | --- |
| Rule | `deployment:` on the rule | Rules that ask their own question: `decision`, `classifiers`, `safety` (and its `hazard` part), `complexity`, and the decision selector `algorithm.decision` |
| Signal type | `global.model_catalog.bindings.<type>`, or `routing.model_bindings.<type>` in a recipe | Signal types whose rules are labels or thresholds over one shared answer: `domain`, `jailbreak`, `pii`, `fact_check`, `user_feedback`, `modality`, `hallucination`, `preference`, `reask` |
| Default | `global.model_catalog.system.decision_model` | Every question-style signal and selector the two levels above do not cover |

A binding in `recipes[].routing.model_bindings` applies only to that recipe
and wins over the global binding for the same signal type.

Embeddings, reranking and the hazard head ask no question. They are named
where they are used:

| Resource | Where you name it | Default |
| --- | --- | --- |
| Embeddings | `global.model_catalog.embeddings.default` | `vela-embedding` |
| Multimodal embeddings | `global.model_catalog.embeddings.multimodal` | `omni-nano` |
| Reranking | `rerank.deployment` in the `rag` plugin | none |
| Hazard head | `hazard.deployment` on a safety rule | `hazard` |

Four deployments are always declared, and the Router starts one only when
something resolves to it: `primary` (`vllm-sr/vela3-fast` on CPU, the default
decision model), `hazard` (Vela 1.0 Hazard), `vela-embedding` (Vela 1.0
Embedding) and `omni-nano` (Vela 1.0 Omni Nano). Declare a deployment with
the same name to change one.

## S1. All defaults: nothing written

```yaml title="s1-defaults"
global:
  model_catalog:
    system:
      decision_model: {deployment: primary}
```

`decision_model: {deployment: primary}` is the default, so leaving it out
changes nothing.

| What | Resolves to |
| --- | --- |
| Every question-style signal and selector | `primary`, by name |
| Hazard head | `hazard` |
| Embeddings | `vela-embedding` |
| Multimodal embeddings | `omni-nano` |

## S2. Everything on a remote System One model

```yaml title="s2-remote-systemone"
global:
  model_catalog:
    deployments:
      d3-remote:
        provider: systemone
        endpoint: https://app.vllm-sr.ai/v1/systemone
        model: vllm-sr/d3-ultra
        auth: {api_key_env: VSR_APP_KEY}      # keys come from an env var or a file, never inline
        timeout_ms: 1500
        capabilities: {presets: [pii, halu]}  # used if the API's /models lists none
    system:
      decision_model: {deployment: d3-remote}
```

- Every built-in question-style signal, custom decision signal and selector
  asks `d3-remote`. The Router checks at load that the deployment answers each
  question type the configuration asks.
- In one request stage, every question to the same deployment travels in one
  call. A question the API reports as failed is asked once more on its own.
- Thresholds come from the model that answers: its model card, or the
  `threshold` you write. A remote model without calibrated thresholds uses
  0.5, and the load reports an `uncalibrated_threshold` warning.
- PII and hallucination need spans. A model that answers neither span
  questions nor the `pii`/`halu` presets answers them yes/no, and the Router
  counts each such signal in `vsr_signal_span_verdict_fallbacks_total`. What
  the API's `/models` lists for the model decides, else the declared
  `capabilities`; when those already lack spans, the load reports a
  `span_fallback` warning. A masking plugin needs PII spans, so it refuses
  such a deployment.
- Embeddings, reranking and the hazard head stay on their local deployments.

## S3. Mixed: PII local, everything else remote

```yaml title="s3-mixed"
global:
  model_catalog:
    deployments:
      local-vela: {provider: model_runtime, artifact: vllm-sr/vela3-fast, device: cpu}
      d3-remote:
        provider: systemone
        endpoint: https://app.vllm-sr.ai/v1/systemone
        model: vllm-sr/d3-ultra
        auth: {api_key_env: VSR_APP_KEY}
    system:
      decision_model: {deployment: d3-remote}
    bindings:
      pii: local-vela
```

`pii` resolves to `local-vela`; every other signal resolves to `d3-remote`.
Binding keys are signal types, as they are named in `routing.signals`; values
are deployment names.

## S4. A decision signal and the domain signal share the default, then one of each is changed

```yaml title="s4-type-and-rule-overrides"
global:
  model_catalog:
    deployments:
      kai:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Kai-0.6B
        revision: cd49ea3813fd8ba0928a9a23ef6c9a0f2f0cd764
        device: auto
      d3-remote:
        provider: systemone
        endpoint: https://app.vllm-sr.ai/v1/systemone
        model: vllm-sr/d3-ultra
        auth: {api_key_env: VSR_APP_KEY}
    system:
      decision_model: {deployment: kai}
routing:                           # or recipes[].routing, which affects only that recipe
  model_bindings:
    domain: d3-remote              # signal type: every domain rule shares one answer
  signals:
    domains: [{name: math, description: Mathematics, mmlu_categories: [math]}]
    decision:
      - name: needs_reasoning      # no deployment: asks the default, kai
        question: {type: noul, instructions: "Does answering need multi-step reasoning?"}
      - name: support_team
        deployment: d3-remote      # rule: only this question goes remote
        question:
          type: choice
          instructions: Which team?
          choices: [{key: billing}, {key: engineering}]
```

`domain` resolves to `d3-remote`, `needs_reasoning` to `kai` and
`support_team` to `d3-remote`. The same block under `recipes[].routing`
changes only that recipe:

```yaml title="s4-recipe-scope"
global:
  model_catalog:
    deployments:
      kai:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Kai-0.6B
        revision: cd49ea3813fd8ba0928a9a23ef6c9a0f2f0cd764
        device: auto
      d3-remote:
        provider: systemone
        endpoint: https://app.vllm-sr.ai/v1/systemone
        model: vllm-sr/d3-ultra
        auth: {api_key_env: VSR_APP_KEY}
    system:
      decision_model: {deployment: kai}
entrypoints:
  - {model_names: [vllm-sr/support], recipe: support}
recipes:
  - name: support
    routing:
      model_bindings:
        domain: d3-remote          # only the support recipe asks d3-remote about domains
      signals:
        domains: [{name: math, description: Mathematics, mmlu_categories: [math]}]
```

In the `support` recipe, `domain` resolves to `d3-remote`; everywhere else it
resolves to `kai`.

## S5. Rules that name their own deployment

```yaml title="s5-rule-level"
routing:
  signals:
    classifiers:
      - {name: risk, deployment: content-risk, labels: [low, high]}
    safety:
      - name: harmful
        deployment: d3-remote
        threshold: 0.5
        hazard: {deployment: hazard, labels: [violence, weapons], categories: [violence], threshold: 0.6}
  decisions:
    - name: pick-model
      modelRefs: [{model: general}, {model: small}]
      algorithm: {type: decision, decision: {deployment: kai, instructions: Pick the cheapest capable model.}}
```

`content-risk`, `d3-remote` and `kai` are declared deployments, here a
`classify` service, a System One API and a Decision model. `risk` asks
`content-risk`, `harmful` asks `d3-remote`, its hazard part asks `hazard`, and
the `pick-model` selector asks `kai`.

## Remote and attached models

The connection block (`endpoint`, `model`, `auth`, `headers`, `timeout_ms`,
`max_retries`, `tls`, `limits`) is the same for every remote service and for a
model runtime the Router attaches to. The
[model catalog reference](model-catalog.md#examples) shows a remote embedding
model, a remote reranker, a remote classifier and chat judge, a System One
gateway with its own header, an attached runtime and a request backend.

## What the Router checks at load

The Router checks every name and every pairing when it loads the
configuration; nothing falls back silently. `vllm-sr config validate --config
config.yaml` runs the same checks.

| Mistake | What the Router reports |
| --- | --- |
| A binding or rule names a deployment that is not declared | The binding or rule, the unknown name and the declared names |
| A deployment cannot answer the signal it is given, such as `bindings.pii: remote-embed` or an `embeddings.default` that serves no embeddings | The signal, the deployment and what it can answer |
| A binding key is not a signal type, such as `bindings.piii` | The key and the closest signal type |
| `deployment:` on a rule whose type shares one answer, such as `domains[].deployment` | That the type is bound by `model_bindings.<type>` |
| A binding for a type whose rules ask their own question, such as `bindings.classifier` | That the deployment goes on each rule |
| A key written inline, such as `auth.api_key` or an `Authorization` header | That keys come from `api_key_env` or `api_key_file` |
| A systemone deployment without spans for `pii` or `hallucination` | A `span_fallback` warning; a masking plugin that needs those spans is refused |

Configurations written in the earlier forms still load, with a
`deprecated_model_field` warning for each field.
[Migrate model resources](model-resource-migration.md) shows each earlier form
next to the form on this page.

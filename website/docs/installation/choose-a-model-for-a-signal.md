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
| Rule | `deployment:` on the rule | Every application rule (`domains`, `fact_check`, `user_feedbacks`, `jailbreak`, `safety`, `pii`, `hallucination`, `refusal`, `turn_relation`, `complexity`, `modality`), every `systemone` rule, and the System One selector `algorithm.systemone` |
| Signal type | `global.model_catalog.bindings.<type>`, or `routing.model_bindings.<type>` in a recipe | The rules of an application signal type that name no deployment: `domain`, `fact_check`, `user_feedback`, `jailbreak`, `safety`, `pii`, `hallucination`, `refusal`, `turn_relation`, `complexity`, `modality` |
| Default | `global.model_catalog.system.decision_model` | Every question the two levels above do not cover |

A binding in `recipes[].routing.model_bindings` applies only to that recipe
and wins over the global binding for the same signal type.

Embeddings ask no question and answer no signal; they serve recall only. They
are named where they are used:

| Resource | Where you name it | Default |
| --- | --- | --- |
| Embeddings | `global.model_catalog.embeddings.default` | `vela-embedding` |
| Multimodal embeddings | `global.model_catalog.embeddings.multimodal` | off |

Two deployments are always declared, and the Router starts one only when
something resolves to it: `primary` (`vllm-sr/vela3-fast` on CPU, the default
decision model) and `vela-embedding` (Vela 1.0 Embedding). Declare a
deployment with the same name to change one.

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
| Embeddings | `vela-embedding` |
| Multimodal embeddings | off |

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

- Every built-in question-style signal, custom `systemone` signal and selector
  asks `d3-remote`. The Router checks at load that the deployment answers each
  question type the configuration asks.
- In one request stage, every question to the same deployment travels in one
  call. A question the API reports as failed is asked once more on its own.
- A rule's default predicate comes from the model that answers: the
  threshold its model card calibrates, or the `predicate` the rule writes. A
  model without calibrated thresholds, local or remote, uses 0.5, and the
  load reports an `uncalibrated_preset` warning naming the rule.
- PII and hallucination rules need spans. A `pii` or `hallucination` rule
  that resolves to a deployment whose declared `capabilities` answer neither
  span questions nor the `pii`/`halu` presets fails to load
  (`span_required`). A masking plugin needs PII spans as well, so it refuses
  such a deployment.
- Embeddings stay on their local deployments.

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

## S4. A systemone signal and the domain signal share the default, then one of each is changed

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
    domains: [{name: math, description: Mathematics, labels: [math]}]
    systemone:
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
        domains: [{name: math, description: Mathematics, labels: [math]}]
```

In the `support` recipe, `domain` resolves to `d3-remote`; everywhere else it
resolves to `kai`.

## S5. Rules that name their own deployment

```yaml title="s5-rule-level"
routing:
  signals:
    domains:
      - {name: math, labels: [math]}
      - {name: law, labels: [law], deployment: kai}
    safety:
      - name: harmful
        deployment: d3-remote
        predicate: {gte: 0.5}
  decisions:
    - name: pick-model
      modelRefs: [{model: general}, {model: small}]
      algorithm: {type: systemone, systemone: {deployment: kai, instructions: Pick the cheapest capable model.}}
```

`d3-remote` and `kai` are declared deployments, here a System One API and a
Decision model. `harmful` asks `d3-remote`, `law` and the `pick-model`
selector ask `kai`, and `math` asks the default, `primary`. Rules of one type
on one deployment still share one question.

## Remote and attached models

The connection block (`endpoint`, `model`, `auth`, `headers`, `timeout_ms`,
`max_retries`, `tls`, `limits`) is the same for every remote service and for a
model runtime the Router attaches to. The
[model catalog reference](model-catalog.md#examples) shows a remote embedding
model, a remote jailbreak guard, a System One gateway with its own header, an
attached runtime and a request backend.

## What the Router checks at load

The Router checks every name and every pairing when it loads the
configuration; nothing falls back silently. `vllm-sr config validate --config
config.yaml` runs the same checks.

| Mistake | What the Router reports |
| --- | --- |
| A binding or rule names a deployment that is not declared | The binding or rule, the unknown name and the declared names |
| A deployment cannot answer the signal it is given, such as `bindings.pii: remote-embed` or an `embeddings.default` that serves no embeddings | The signal, the deployment and what it can answer |
| A binding key is not a signal type, such as `bindings.piii` | The key and the closest signal type |
| A deployment that cannot answer the questions a rule asks, such as a `classify` deployment named by a `domains` rule | The rule, the deployment and the providers that can answer it |
| A key written inline, such as `auth.api_key` or an `Authorization` header | That keys come from `api_key_env` or `api_key_file` |
| A `pii` or `hallucination` rule on a deployment without spans | `span_required`; a masking plugin that needs those spans is refused as well |

Configurations written in the earlier forms still load, with a
`deprecated_model_field` warning for each field.
[Migrate model resources](model-resource-migration.md) shows each earlier form
next to the form on this page.

---
title: Prompt attacks and unsafe content
sidebar_label: Prompt attacks and unsafe content
description: Detect jailbreaks and prompt injection, and unsafe content, with the decision model's jailbreak and harm presets.
---

# Prompt attacks and unsafe content

Two separate questions protect your models:

- **Is the request an attack?** The jailbreak preset detects prompt injection
  and jailbreak attempts. It backs the
  [`jailbreak` signal](tutorials/signal/learned/jailbreak.md).
- **Is the content unsafe?** The harm preset scores whether a request or an
  answer is unsafe and, with labels, in which of 34 harm categories. It backs
  the [`safety` signal](tutorials/signal/learned/safety.md).

An unsafe request is not necessarily an attack, and an attack can be politely
worded, so most deployments use both.

## Turn it on

```yaml
recipes:
  - name: default
    signals:
      jailbreak:
        - name: prompt_attack
          predicate: {gte: 0.5}
      safety:
        - name: unsafe-content
          predicate: {gte: 0.5}
    decisions:
      - name: block-attacks
        priority: 300
        rules:
          operator: AND
          conditions:
            - type: jailbreak
              name: prompt_attack
        modelRefs:
          - model: refusal-model
      - name: handle-content-risk
        priority: 290
        rules:
          operator: AND
          conditions:
            - type: safety
              name: unsafe-content
        modelRefs:
          - model: safety-capable-model
```

Without a binding or a rule's own `deployment`, jailbreak and safety rules ask
the decision model, vela3-fast by default; a rule without a `predicate` uses
the model's calibrated threshold. Re-evaluate rule predicates when changing
models. See
[Choose a model](../choose-a-model#built-in-signals) and
[Choose a model for a signal](../../installation/choose-a-model-for-a-signal.md).

## Choose a model and where it runs

| Feature | Where its deployment is named | Default |
| --- | --- | --- |
| Jailbreak | `bindings.jailbreak`, or `deployment` on the rule | The decision model (`vllm-sr/vela3-fast`) |
| Safety | `bindings.safety`, or `deployment` on the rule | The decision model (`vllm-sr/vela3-fast`) |

A safety rule with `labels` matches content that is harmful in one of those
harm categories, each on its own probability. See the
[safety signal guide](tutorials/signal/learned/safety.md) for the categories
and complete configurations.

To ask another model, declare its deployment and name it on the rule:

```yaml
global:
  model_catalog:
    deployments:
      vela2:
        provider: model_runtime
        artifact: vllm-sr/Vela-2.0-0.8B
        device: cpu
recipes:
  - name: default
    signals:
      safety:
        - name: unsafe-content
          deployment: vela2
          labels: [Harassment]
```

To run the jailbreak check on a GPU, describe a deployment and bind it:

```yaml
global:
  model_catalog:
    deployments:
      vela-gpu:
        provider: model_runtime
        artifact: vllm-sr/vela3-fast
        device: rocm:0
    bindings:
      jailbreak: vela-gpu
```

## When a check cannot finish

A model that is not ready, a timeout or an input over the limit makes the
signal unknown. Decide what that means per route with `rules.on_unknown`
(`no_match` or `fail_request`). An input the model did not read in full (over
its input under `reject`, over its
[scan cap](model-runtime/reference.md#long-inputs), truncated, or not scanned
by the signals' deadline) matches a jailbreak rule as `unscanned`, so padding a
prompt cannot carry an attack past it:

```yaml
recipes:
  - name: default
    signals:
      jailbreak:
        - name: prompt_attack
    decisions:
      - name: block-attacks
        priority: 300
        rules:
          operator: AND
          on_unknown: fail_request
          conditions:
            - type: jailbreak
              name: prompt_attack
        modelRefs:
          - model: refusal-model
```

With `fail_request`, a request that could not be checked is refused.

## Check it

Through the router, `x-vsr-matched-jailbreak` and `x-vsr-matched-safety` list
the rules that matched.

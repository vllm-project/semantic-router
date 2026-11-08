---
title: Classify requests
description: Route on a request's domain, its need for fact checking, user feedback and the requested output modality, or on labels of your own question.
---

# Classify requests

Classifying signals give every request a label: its subject, whether it needs
fact checking, how the user reacts to the last answer, or whether they ask for
an image. Routes match on those labels.

| Signal | Preset | Labels |
| --- | --- | --- |
| [`domain`](tutorials/signal/learned/domain.md) | domain | 14 domains: `biology`, `business`, `chemistry`, `computer science`, `economics`, `engineering`, `health`, `history`, `law`, `math`, `other`, `philosophy`, `physics`, `psychology` |
| [`fact_check`](tutorials/signal/learned/fact-check.md) | fact_check | `FACT_CHECK_NEEDED`, `NO_FACT_CHECK_NEEDED` |
| [`user_feedback`](tutorials/signal/learned/user-feedback.md) | feedback | `SAT`, `NEED_CLARIFICATION`, `WRONG_ANSWER`, `WANT_DIFFERENT`, `NO_FEEDBACK` |
| [`modality`](tutorials/signal/learned/modality.md) | modality | `text`, `image`, `text_image` and the other outputs |

Without a binding, these signals ask the decision model, vela3-fast by default.
See [Choose a model](../choose-a-model#by-task) and
[Choose a model for a signal](../../installation/choose-a-model-for-a-signal.md).

## Turn it on

Add the signal and use it in a route. A domain rule matches the domains its
`labels` name, by default its own name:

```yaml
recipes:
  - name: default
    signals:
      domains:
        - name: math
          description: Mathematics and quantitative reasoning.
        - name: computing
          description: Programming and computer science.
          labels: [computer science]
    decisions:
      - name: math-route
        priority: 100
        rules:
          operator: AND
          conditions:
            - type: domain
              name: math
        modelRefs:
          - model: math-model
```

## Choose where it runs

To pick the device or another model, describe a deployment and bind the
signal to it. The binding keys are the signal types `domain`, `fact_check`,
`user_feedback` and `modality`; the deployment must answer the presets'
questions, as the decision model does:

```yaml
global:
  model_catalog:
    deployments:
      vela-gpu:
        provider: model_runtime
        artifact: vllm-sr/vela3-fast
        device: rocm:0
    bindings:
      domain: vela-gpu
```

## Check it

Through the router, send a request with the `x-vsr-debug: true` header:
`x-vsr-matched-domains` lists the domains that matched and
`x-vsr-selected-decision` names the route.

## Use your own labels

Labels of your own are a [systemone](tutorials/signal/learned/systemone.md)
question: a choice over them that the decision model answers. Route on one of
its labels:

```yaml
recipes:
  - name: default
    signals:
      systemone:
        - name: ticket_topic
          question:
            type: choice
            instructions: Which desk should handle the ticket?
            choices:
              - key: billing
              - key: shipping
              - key: other
    decisions:
      - name: billing-desk
        priority: 100
        rules:
          operator: AND
          conditions:
            - type: systemone
              name: ticket_topic
              label: billing
              predicate:
                gte: 0.5
        modelRefs:
          - model: support-model
```

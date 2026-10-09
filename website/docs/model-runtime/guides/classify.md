---
title: Classify requests
description: Route on a request's domain, its need for fact checking, user feedback and the requested output modality, or on labels of your own classifier.
---

# Classify requests

Classifiers give every request a label: its subject, whether it needs fact
checking, how the user reacts to the last answer, or whether they ask for an
image. Routes match on those labels.

| Signal | Model | Labels |
| --- | --- | --- |
| [`domain`](tutorials/signal/learned/domain.md) | Vela 1.0 Domain | 14 domains: `biology`, `business`, `chemistry`, `computer science`, `economics`, `engineering`, `health`, `history`, `law`, `math`, `other`, `philosophy`, `physics`, `psychology` |
| [`fact_check`](tutorials/signal/learned/fact-check.md) | Vela 1.0 FactCheck | `FACT_CHECK_NEEDED`, `NO_FACT_CHECK_NEEDED` |
| [`user_feedback`](tutorials/signal/learned/user-feedback.md) | Vela 1.0 Feedback | satisfied, need clarification, wrong answer, want different, no feedback |
| [`modality`](tutorials/signal/learned/modality.md) | Vela 1.0 Modality | `AR` (text), `DIFFUSION` (image), `BOTH` |
| [`classifier`](tutorials/signal/learned/classifier.md) | your own model | your labels |

The table lists specialist models. With no task override, the built-in domain,
fact-check, feedback, and modality signals use the default Vela 2.0 judgment
deployment. The specialist bindings below select Vela 1.0 explicitly. See
[Choose a model](../choose-a-model#by-task).

## Turn it on

Add the signal and use it in a route. The router runs the right model for you
on the CPU; nothing else is needed:

```yaml
routing:
  signals:
    domains:
      - name: math
        description: Mathematics and quantitative reasoning.
        mmlu_categories: [math]
      - name: computer science
        description: Programming and computer science.
        mmlu_categories: [computer science]
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

To pick the device, the input limit or another model, describe a deployment
and bind the feature to it. The specialist binding names are `domain_classifier`,
`fact_check_classifier`, `feedback_detector` and `modality_detector`; their
native classifier contract reads label probabilities (`label_distribution.v1`):

```yaml
global:
  model_catalog:
    deployments:
      vela-domain:
        provider: model_runtime
        artifact: vllm-sr/Vela-1.0-Encoder-307M-Domain
        device: cpu
        input:
          max_tokens: 2048
          overflow: truncate
    bindings:
      domain_classifier:
        deployment: vela-domain
        contract: label_distribution.v1
```

`overflow: truncate` classifies the first 2,048 tokens of a long request and
reports that it did; `reject` (the default) leaves the signal unknown for
input over the limit instead.

## Check it

Ask the model directly. Start it on its own, or use any runtime that serves it:

These worker-level examples run inside an environment containing `vllm-srun`
(such as the Router image). Classify, embeddings, rerank and bundle are worker
APIs; the instance frontend publishes System One and decision requests.

```bash
vllm-srun serve vllm-sr/Vela-1.0-Encoder-307M-Domain --device cpu --port 8100
curl -s localhost:8100/v1/classify -H 'content-type: application/json' \
  -d '{"input": ["What is the derivative of x squared?", "Fix this segfault in my C code."]}'
```

Each result has the most likely `label` and the `probabilities` of all labels
in the order of `labels`. Through the router, send a request with the
`x-vsr-debug: true` header: `x-vsr-matched-domains` lists the domains that
matched and `x-vsr-selected-decision` names the route.

## Use your own classifier

A Hugging Face ModernBERT or mmBERT sequence classifier works as a `classifier`
signal with your own labels. Pin its revision, bind it to the rule and route on
its labels:

```yaml
routing:
  signals:
    classifiers:
      - name: ticket_topic
        type: local
        labels: [billing, shipping, other]
  decisions:
    - name: billing-desk
      priority: 100
      rules:
        operator: AND
        conditions:
          - type: classifier
            name: ticket_topic
            label: billing
            predicate:
              gte: 0.5
      modelRefs:
        - model: support-model
  model_bindings:
    classifier.ticket_topic:
      deployment: ticket-topics
      contract: label_distribution.v1
global:
  model_catalog:
    deployments:
      ticket-topics:
        provider: model_runtime
        artifact: your-org/ticket-topic-classifier
        revision: 0123456789abcdef0123456789abcdef01234567
        device: cpu
        input:
          max_tokens: 512
          overflow: reject
```

List `labels` in the model's own label order (its `id2label`). The router
reads the model's labels from the runtime and refuses to start if they do not
match the rule. A model with independent
labels and a published operating point uses `label_scores.v1` instead; see
[classifier signals](tutorials/signal/learned/classifier.md).

# Classifier Signal

## Overview

`classifier` exposes reusable label scores from a local native sequence classifier
or a configured external LLM. Decisions test a declared label with a required
numeric predicate.

Specialized domain, PII, jailbreak, fact-check, KB, and preference signals
remain the preferred interfaces for their respective domains.

## Key Advantages

- integrates arbitrary sequence-classification heads without adding domain logic
- constrains LLM classifiers to declared labels and deterministic JSON output
- computes one label map that multiple decisions can gate at different scores

## What Problem Does It Solve?

Some trained classifiers do not belong to the built-in signal taxonomies. The
classifier signal exposes those labels and scores to decisions without mixing
classification with route outcomes.

## When to Use

Use this signal for a genuine reusable classification head or a prompted LLM
labeler. Prefer embedding/KB signals for reference-phrase similarity and
preference signals for response-style routing.

## Configuration

```yaml
document:
  signals:
    classifiers:
      - name: phishing
        type: local
        model_path: models/phishing-email
        labels: [BENIGN, PHISHING]
        use_cpu: true

  decisions:
    - name: phishing-local
      description: Keep suspected phishing requests on the local model.
      priority: 200
      rules:
        operator: AND
        conditions:
          - type: classifier
            name: phishing
            label: PHISHING
            predicate:
              gte: 0.5
            on_error: no_match
```

LLM classifiers reference a named `global.model_catalog.external` entry and
add `instructions`. The runtime fixes temperature, output schema, token bounds,
and exact-label validation. Classifier leaves are the only decision predicates
that accept `on_error`; failures expose the bounded
`classifier_evaluation_failed` code in eval/replay diagnostics.

This condition-level `on_error` (`no_match` or `match`) decides what the
predicate evaluates to when the classifier fails. It is a different key from
`prompt_guard.on_error` (`allow` or `block`), which decides whether a guardrail
backend failure counts as unverified content for every rule that backend
serves. See [Safety models and policy](../../global/safety-models-and-policy.md).

Local classifiers use `model_path`. One binary local classifier is supported
per Router process, and its decision predicates use `gte: 0.5` or higher on the
winning-label confidence. Restart the Router after changing the model or label
order. A management API update that requires this restart returns
`RESTART_REQUIRED`.

The local path processes request text inside the Router. An `llm` classifier
sends that text to its configured external model, so choose the provider and
retention policy accordingly. Labels and thresholds must be evaluated as one
versioned contract. See a complete example:
[`config/fragments/signal/classifier/label-score.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/classifier/label-score.yaml).

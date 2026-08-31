# Classifier Signal

## Overview

`classifier` exposes reusable label scores from a local native sequence classifier,
a remote sequence classifier, or a configured external LLM. Decisions test a
declared label with a required numeric predicate.

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
routing:
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
        on_unknown: no_match
        conditions:
          - type: classifier
            name: phishing
            label: PHISHING
            predicate:
              gte: 0.5
      modelRefs:
        - model: local-small
          use_reasoning: false
```

LLM classifiers reference a named `global.model_catalog.external` entry and
add `instructions`. The runtime fixes temperature, output schema, and
exact-label validation. Classifier leaves are the only decision predicates
that accept `on_error`; failures expose the bounded
`classifier_evaluation_failed` code in eval/replay diagnostics.

On failure, the decision tree evaluates this leaf as `Unknown` until the full
AND/OR/NOT expression is known. Root-level `rules.on_unknown` then chooses
`no_match`, `match`, or `fail_request`. `no_match` and `match` resolve only
their own decision; `fail_request` is global fail-closed: it rejects the whole
request with a 503 even when another decision matches cleanly, regardless of
priority. When `rules.on_unknown` is omitted, condition-level `on_error`
(`no_match` or `match`) preserves the previous generic-classifier result.
Setting `rules.on_unknown` disables every condition-level `on_error` in that
tree, so the Router rejects a configuration that sets both.
`prompt_guard.on_error` (`allow` or `block`) remains the compatibility
default for jailbreak rules. Diagnostics include both the signal error and any
terminal policy that was applied. See
[Safety models and policy](../../global/safety-models-and-policy.md).

`sequence_classifier` classifiers also reference a named external model, but
use the shared `http_classify` contract and preserve its full label distribution.
The response must contain exactly the declared labels, with scores that sum to
approximately `1.0`; sigmoid multi-label outputs and label subsets are rejected.
They require at least two labels and do not accept `instructions`, `model_path`,
or `use_cpu`.

Local classifiers use `model_path`. One binary local classifier is supported
per Router process, and its decision predicates use `gte: 0.5` or higher on the
winning-label confidence. Restart the Router after changing the model or label
order. A management API update that requires this restart returns
`RESTART_REQUIRED`.

The local path processes request text inside the Router. Both `llm` and
`sequence_classifier` send that text to their configured external model, so
choose the provider and retention policy accordingly. Labels and thresholds
must be evaluated as one versioned contract. See complete examples for
[`llm`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/classifier/label-score.yaml)
and
[`sequence_classifier`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/classifier/sequence-label-score.yaml).

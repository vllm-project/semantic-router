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
classifier signal provides a narrow label-score seam while preserving the
separation between fact extraction and decision control logic.

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
        conditions:
          - type: classifier
            name: phishing
            label: PHISHING
            predicate:
              gte: 0.5
            on_error: no_match
      modelRefs:
        - model: local-small
          use_reasoning: false
```

LLM classifiers reference a named `global.model_catalog.external` entry and
add `instructions`. The runtime fixes temperature, output schema, token bounds,
and exact-label validation. Classifier leaves are the only decision predicates
that accept `on_error`; failures expose the bounded
`classifier_evaluation_failed` code in eval/replay diagnostics.

Local classifiers use `model_path`; paths under `models/` participate in the
normal model registry/download flow. One binary local generic classifier is
supported per process, and its decision predicates use `gte: 0.5` or higher on
the winning-label confidence. Changing its model or label order requires a
router restart so an in-flight config reload cannot swap process-global native
state. Management API updates that attempt this mutation return
`RESTART_REQUIRED` rather than applying a partial runtime snapshot.

The local path processes request text inside the Router. An `llm` classifier
sends that text to its configured external model, so choose the provider and
retention policy accordingly. Labels and thresholds must be evaluated as one
versioned contract. Maintained example:
[`config/fragments/signal/classifier/label-score.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/classifier/label-score.yaml).

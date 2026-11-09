# Domain Signal

## Overview

`domain` classifies the request topic family. Define domain rules under
`routing.signals.domains`.

The decision model answers it unless a `domain` binding names another
deployment; see [Choose a model for a signal](../../../installation/choose-a-model-for-a-signal.md).
`global.model_catalog.modules.classifier.domain` holds its settings, such as
its threshold.

## Key Advantages

- Routes by topic without hard-coding every phrase into keyword lists.
- Keeps domain policy reusable across multiple decisions.
- Supports stable category families that are easy to audit.
- Works well as the first learned signal in a routing graph.

## What Problem Does It Solve?

Keyword routing breaks down once prompts are paraphrased or when domain boundaries are broader than a handful of phrases.

`domain` solves that by mapping topic classification into named routing signals that decisions can compose with complexity, safety, or plugin logic.

## When to Use

Use `domain` when:

- routes are organized around topic families
- lexical matching is too brittle
- the same topic boundary should feed several decisions
- you want a stable learned classifier before adding more specialized signals

## Configuration

```yaml
routing:
  signals:
    domains:
      - name: business
        description: Business and management related queries.
        mmlu_categories: [business]
      - name: law
        description: Legal questions and law-related topics.
        mmlu_categories: [law]
      - name: psychology
        description: Psychology and mental health topics.
        mmlu_categories: [psychology]
      - name: health
        description: Health and medical information queries.
        mmlu_categories: [health]
      - name: other
        description: General fallback traffic.
        mmlu_categories: [other]
```

Keep domain names stable because decisions reference those names directly.

Only declared domains match. A classifier label matches the domain that lists
it in `mmlu_categories`, or the domain named after it. A label no domain lists
counts as `other`, so the domain that lists `other`, as above, is the fallback
for every topic you didn't declare; without one, such a request matches no
domain.

### Local and remote models

By default the decision model (vela3-fast) answers the domain signal in the
[model runtime](../../../model-runtime/guides/classify.md). The runtime reads
the model's architecture from the package, so there is no local selector to
set; `vllm-sr config migrate` removes the earlier `variant`, `use_modernbert`
and `use_mmbert_32k` keys.

A remote domain classifier is a `classify` deployment that `domain` is bound
to. The Router posts the request text to `<endpoint>/classify` and reads the
full label distribution, which feeds domain matching and routing decisions.

```yaml
global:
  model_catalog:
    deployments:
      domain-service:
        provider: classify
        endpoint: https://domain-classifier.example.com
        auth: {api_key_env: DOMAIN_SERVICE_KEY}
        timeout_ms: 5000
    bindings:
      domain: domain-service
```

## Dependencies and Limitations

Domain classification processes the request text on the deployment that
answers the signal. Declare `other` as the fallback, and re-evaluate labels and
thresholds when that deployment changes. See a complete example:
[`config/fragments/signal/domain/mmlu.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/domain/mmlu.yaml).

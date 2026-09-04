# Domain Signal

## Overview

`domain` classifies the request topic family. Define domain rules under
`routing.signals.domains`.

The detector uses the classifier configured under
`global.model_catalog.modules.classifier` and its model bindings under
`global.model_catalog.system`.

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

### Local and remote classifier selection

With no `backend`, category/domain classification keeps its existing local
model behavior. Use `variant: candle`, `variant: modernbert`, or
`variant: mmbert32k` for an explicit local selector; the deprecated
`use_modernbert` and `use_mmbert_32k` keys remain readable for compatibility
but are normalized to `variant` in canonical output. An agreeing canonical and
legacy selector is accepted; contradictory active selectors are rejected.

A remote category classifier uses the shared backend block. Its `model` is an
explicit name from `global.model_catalog.external[]`, and that catalog entry
must have `model_role: classification`. Category currently accepts only the
`http_classify` protocol and the `label_distribution.v1` contract so the full
label distribution continues to feed domain matching and routing decisions.

```yaml
global:
  model_catalog:
    external:
      - name: domain-service
        model_role: classification
        llm_endpoint:
          address: domain-classifier.default.svc
          port: 8080
        llm_model_name: domain-intent-v1
    modules:
      classifier:
        domain:
          backend:
            protocol: http_classify
            contract: label_distribution.v1
            model: domain-service
            deadline_ms: 5000
```

## Dependencies and Limitations

Domain classification uses the configured classifier module and processes the
request text. Treat `other` as a fallback, and re-evaluate labels and thresholds
when the classifier changes. See a complete example:
[`config/fragments/signal/domain/mmlu.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/domain/mmlu.yaml).

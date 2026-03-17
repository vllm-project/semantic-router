# Domain Signal

## Overview

`domain` classifies the request topic family. It maps to `config/signal/domain/` and is declared under `routing.signals.domains`.

This family is learned: the router uses the domain-classification path under `global.model_catalog.modules.classifier` and the stable domain system model bindings in `global.model_catalog.system`.

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

Source fragment family: `config/signal/domain/`

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

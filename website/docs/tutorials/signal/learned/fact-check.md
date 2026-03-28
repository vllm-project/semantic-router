# Fact Check Signal

## Overview

`fact-check` decides whether a prompt should be treated as evidence-sensitive traffic. It maps to `config/signal/fact-check/` and is declared under `routing.signals.fact_check`.

This family is learned: it relies on the fact-check classification path under `global.model_catalog.modules.hallucination_mitigation.fact_check`.

## Key Advantages

- Separates factual verification from general domain routing.
- Helps decisions choose safer plugins or stronger models for evidence-sensitive traffic.
- Keeps verification policy visible instead of burying it inside later plugins.
- Exposes both positive and negative labels such as `needs_fact_check` and `no_fact_check_needed`.

## What Problem Does It Solve?

Not every prompt needs the same level of factual grounding. If the router treats all traffic the same, creative prompts can be over-constrained while factual prompts can be under-protected.

`fact-check` solves that by detecting which prompts should trigger evidence-aware routing behavior.

## When to Use

Use `fact-check` when:

- factual claims need stricter routing or plugins
- creative prompts should bypass expensive verification paths
- hallucination mitigation depends on an early routing signal
- you want factuality handled as routing policy instead of post-hoc repair

## Configuration

Source fragment family: `config/signal/fact-check/`

```yaml
routing:
  signals:
    fact_check:
      - name: needs_fact_check
        description: Queries with factual claims that should be verified against evidence.
      - name: no_fact_check_needed
        description: Creative or opinion-heavy prompts that do not need factual verification.
```

Define only the labels your decisions will reference. The learned classifier decides which one fires.

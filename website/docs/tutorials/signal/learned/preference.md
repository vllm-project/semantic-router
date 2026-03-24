# Preference Signal

## Overview

`preference` infers user response-style preferences from examples and classifier settings. It maps to `config/signal/preference/` and is declared under `routing.signals.preferences`.

This family is learned: it uses the preference-classification path under `global.model_catalog.modules.classifier.preference`.

If `global.model_catalog.modules.classifier.preference.use_contrastive` is omitted, vSR now defaults it to `true`. That means a profile like `deploy/recipes/balance.yaml` can rely on preference signals without adding a separate global classifier block unless it wants to disable contrastive mode explicitly.

## Key Advantages

- Personalizes routing without hard-coding user state into decisions.
- Keeps preference detection separate from route outcomes.
- Supports example-driven style detection such as terse vs detailed answers.
- Reuses one preference policy across multiple decisions.

## What Problem Does It Solve?

Users often want different response styles even when they ask about the same topic. If those preferences are only handled downstream, routing cannot choose the most suitable model or plugin stack.

`preference` solves that by exposing inferred style preferences as named routing inputs.

## When to Use

Use `preference` when:

- some users prefer terse answers while others want high detail
- route behavior should adapt to stable style preferences
- you want preference detection to stay reusable across several decisions
- user style signals should influence model choice, plugin choice, or both

## Configuration

Source fragment family: `config/signal/preference/`

```yaml
routing:
  signals:
    preferences:
      - name: terse_answers
        description: Users who prefer short, direct responses.
        examples:
          - keep it concise
          - bullet points only
          - answer in one paragraph
        threshold: 0.7
```

Treat the examples as training anchors for the preference detector, not as literal keyword rules.

```yaml
global:
  model_catalog:
    modules:
      classifier:
        preference:
          use_contrastive: false # optional override; default is true
```

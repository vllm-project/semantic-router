# Preference Signal

## Overview

`preference` infers response-style preferences from examples and classifier
settings. Define preference rules under `routing.signals.preferences`.

This family is learned: it uses the preference-classification path under `global.model_catalog.modules.classifier.preference`.

`global.model_catalog.modules.classifier.preference.use_contrastive` defaults
to `true`. Set it to `false` only when you intentionally want the alternative
classifier path.

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

```yaml
document:
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
          prototype_scoring:
            enabled: true
            cluster_similarity_threshold: 0.9
            max_prototypes: 8
            best_weight: 0.75
            top_m: 2
            margin_threshold: 0.05
```

In contrastive mode, the router embeds each preference rule's descriptions and examples, compresses them into representative prototypes when `prototype_scoring` is enabled, and compares the incoming request against those prototypes. `margin_threshold` lets you reject ambiguous winners instead of forcing a weak preference match.

## Dependencies and Limitations

Preference rules use the shared embedding/classifier path and infer style only
from the available request context. They should not be treated as durable user
consent or identity. See a complete example:
[`config/fragments/signal/preference/power-user.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/preference/power-user.yaml).

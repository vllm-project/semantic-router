# Operational Signals

## Overview

Operational signals adapt route behavior to request shape, user context, or interaction history.

This page aligns to:

- `config/signal/complexity/`
- `config/signal/context/`
- `config/signal/language/`
- `config/signal/modality/`
- `config/signal/preference/`
- `config/signal/user-feedback/`

## Key Advantages

- Turns request shape into routing policy instead of hard-coded heuristics.
- Supports personalization without baking user state into decisions.
- Helps one route switch models, reasoning, or plugins based on runtime context.
- Makes operational adaptation explicit and reusable.

## What Problem Does It Solve?

Two requests may belong to the same domain but still need different treatment because one is long-context, one is multimodal, one comes from a power user, or one follows negative feedback.

Operational signals solve that by exposing runtime context as named inputs to the decision layer.

## When to Use

Use operational signals when:

- route choice depends on context length or complexity
- language or modality should steer model selection
- the router should react to user preferences or prior feedback
- the same domain route needs different behavior under different runtime conditions

## Configuration

Configure the relevant family under `routing.signals`.

### Complexity

Source fragment family: `config/signal/complexity/`

```yaml
routing:
  signals:
    complexity:
      - name: needs_reasoning
        threshold: 0.7
```

Use this when some requests should enable stronger reasoning or a different algorithm.

### Context

Source fragment family: `config/signal/context/`

```yaml
routing:
  signals:
    context:
      - name: long_context
        token_threshold: 16000
```

Use this when long-context traffic should go to a different model family.

### Language and Modality

Source fragment families: `config/signal/language/`, `config/signal/modality/`

```yaml
routing:
  signals:
    language:
      - name: multilingual
        allowed_languages: [en, zh]
    modality:
      - name: image_request
        modalities: [image]
```

Use these when routing depends on locale or input/output modality.

### Preference and User Feedback

Source fragment families: `config/signal/preference/`, `config/signal/user-feedback/`

```yaml
routing:
  signals:
    preferences:
      - name: power_user
        preference_keys: [reasoning, detail]
    user_feedbacks:
      - name: escalation_feedback
        min_rating_delta: 1
```

Use these when routing should adapt to saved user preferences or previous feedback.

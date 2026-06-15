# Context Signal

## Overview

`context` detects requests that need a larger effective context window. It maps to `config/signal/context/` and is declared under `routing.signals.context`.

This family is heuristic: it routes from token-window requirements rather than classifier inference.

## Key Advantages

- Keeps long-context routing explicit instead of burying it in model defaults.
- Prevents short prompts from paying the cost of oversized context models.
- Reuses one context threshold across multiple decisions.
- Works well alongside domain or complexity signals.

## What Problem Does It Solve?

Two prompts can ask about the same topic but require very different context windows. If routing only looks at domain, long documents can land on models that truncate or fail.

`context` solves that by making context-window needs a first-class routing input.

## When to Use

Use `context` when:

- some routes need 32K, 128K, or larger context support
- long-document traffic should use a different model family
- you want short requests to stay on cheaper or faster models
- routing depends on context size rather than topic alone

## Configuration

Source fragment family: `config/signal/context/`

```yaml
routing:
  signals:
    context:
      - name: long_context
        min_tokens: 32K
        max_tokens: 256K
        description: Requests that need a larger effective context window.
```

Use `context` when the router should switch candidates based on prompt length or expected context demand.

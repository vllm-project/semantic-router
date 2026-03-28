# Language Signal

## Overview

`language` detects the request language and exposes it as a routing signal. It maps to `config/signal/language/` and is declared under `routing.signals.language`.

This family is heuristic in the tutorial taxonomy because it uses a lightweight language detector instead of router-owned classifier models.

## Key Advantages

- Lets multilingual traffic route without duplicating decisions per locale.
- Keeps language handling explicit in the routing graph.
- Works well with modality, context, and model-family constraints.
- Avoids paying for a domain classifier when only locale matters.

## What Problem Does It Solve?

If language is ignored, multilingual traffic can land on models that are weak for the detected locale or on plugins that assume English-only behavior.

`language` solves that by turning detected locale into a reusable routing input.

## When to Use

Use `language` when:

- different languages need different model families
- multilingual support is partial or tiered
- downstream tools or prompts depend on locale
- you want a clean split between language detection and route outcomes

## Configuration

Source fragment family: `config/signal/language/`

```yaml
routing:
  signals:
    language:
      - name: zh
        description: Chinese-language requests.
      - name: es
        description: Spanish-language requests.
```

The rule names should match the language codes you want decisions to reference, such as `zh`, `es`, or `en`.

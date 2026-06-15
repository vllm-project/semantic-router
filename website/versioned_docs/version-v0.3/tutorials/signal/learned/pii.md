# PII Signal

## Overview

`pii` detects sensitive personal data in requests. It maps to `config/signal/pii/` and is declared under `routing.signals.pii`.

This family is learned: it uses the router-owned PII detection path configured through `global.model_catalog.system.pii_classifier`.

## Key Advantages

- Makes privacy-sensitive routing explicit.
- Lets decisions block, downgrade, or isolate risky traffic before it reaches a backend.
- Supports allowlists for low-risk identifier types.
- Keeps privacy policy reusable across routes and plugins.

## What Problem Does It Solve?

Without a dedicated PII signal, privacy-sensitive traffic can reach the wrong model or plugin stack before detection happens. Ad hoc filters also make policy harder to audit.

`pii` solves that by turning personal-data detection into a reusable routing input.

## When to Use

Use `pii` when:

- prompts may contain regulated or sensitive personal data
- some PII types are acceptable but others must trigger a safer route
- privacy-sensitive traffic needs different plugins or backends
- route policy depends on early PII detection

## Configuration

Source fragment family: `config/signal/pii/`

```yaml
routing:
  signals:
    pii:
      - name: restricted_pii
        threshold: 0.85
        include_history: true
        pii_types_allowed:
          - EMAIL_ADDRESS
        description: Sensitive prompts where only low-risk identifiers may pass through.
```

When `pii_types_allowed` is empty, any detected PII can cause the signal to match.

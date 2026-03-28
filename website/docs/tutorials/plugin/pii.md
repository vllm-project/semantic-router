# PII

## Overview

`pii` is a route-local plugin for applying PII screening or redaction policies after a decision matches.

It aligns to `config/plugin/pii/redact.yaml`.

## Key Advantages

- Keeps PII policy local to the routes that need it.
- Makes allowed PII types explicit instead of implicit.
- Complements global PII model loading with route-local enforcement.

## What Problem Does It Solve?

Some routes may allow limited identifiers, while others should redact nearly everything. `pii` makes that route-local policy explicit without turning every route into the strictest one.

## When to Use

- one route needs route-specific PII handling
- PII should be redacted, blocked, or narrowed by allowed type
- the route should reuse the global PII model but not globalize enforcement

## Configuration

Use this fragment under `routing.decisions[].plugins`:

```yaml
plugin:
  type: pii
  configuration:
    enabled: true
    threshold: 0.85
    pii_types_allowed:
      - EMAIL_ADDRESS
```

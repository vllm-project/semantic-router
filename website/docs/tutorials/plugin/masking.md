# Masking

## Overview

`masking` is a route-local plugin that detects personal data (emails, phone numbers, names, and other configured entity types) in the provider-bound request and replaces each occurrence with a numbered placeholder, such as `[EMAIL_ADDRESS_0]`, before the request leaves the router.

## What Problem Does It Solve?

Some routes send traffic to a model provider that should never see raw personal data — a third-party API, a lower-trust tier, or a route subject to a data-handling policy. `masking` rewrites the content immediately before dispatch so the provider only ever sees placeholders, without requiring every caller to redact input themselves.

## When to Use

- a route dispatches to a provider that must not receive personal data
- the same value should map to the same placeholder throughout one request, and different values should map to different placeholders
- a missing or unavailable PII classifier should block the request rather than silently send it unmasked

## Configuration

Add the plugin under `routing.decisions[].plugins`:

```yaml
plugins:
  - type: masking
    configuration:
      enabled: true
      threshold: 0.85
      entity_types:
        - EMAIL_ADDRESS
        - PHONE_NUMBER
      placeholders:
        EMAIL_ADDRESS: "[EMAIL_{index}]"
        PHONE_NUMBER: "[PHONE_{index}]"
```

`entity_types` is an include list: only listed types are masked, and an empty list masks every detected type. Each `placeholders` template must contain `{index}`, since dropping it would collapse distinct values onto the same placeholder. If the PII classifier is unavailable, requests on a masking-enabled route fail closed rather than dispatching unmasked content. See a complete example:
[`config/fragments/plugin/masking/basic.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/masking/basic.yaml).

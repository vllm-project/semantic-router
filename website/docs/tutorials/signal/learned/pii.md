# PII Signal

## Overview

`pii` detects sensitive personal data in requests. Define PII rules under
`routing.signals.pii`.

It uses the PII detector configured through
`global.model_catalog.system.pii_classifier`.

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

To scan textual results returned by tools, opt in with `source: tool_result`:

```yaml
routing:
  signals:
    pii:
      - name: tool_result_pii
        source: tool_result
        threshold: 0.85
        pii_types_allowed: []
        description: Keep tool results containing sensitive data on a protected route.
```

This source is evaluated from the protocol-neutral tool-result representation,
so the rule works across the supported chat, responses, and messages formats.
It scans textual tool-result content only; tool calls and non-text content are
not included. Tool-result scope is independent from prompt/history scope, so
it does not automatically scan either of those inputs. Omitting `source` (or
leaving it empty) preserves the legacy prompt and optional history behavior.

## Dependencies and Limitations

The PII classifier processes the configured source scope. It is a routing
control, not a substitute for redaction, encryption, access control, or data
loss prevention. Calibrate thresholds by entity type. See a complete example:
[`config/fragments/signal/pii/strict.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/pii/strict.yaml).

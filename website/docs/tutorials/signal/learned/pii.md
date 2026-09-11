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

## Remote backend (token_spans.v1)

With no `backend`, PII detection keeps its local model. A remote PII classifier
uses the shared backend block: `model` names an entry in
`global.model_catalog.external[]` with `model_role: classification`, the
protocol is `http_classify`, and the contract is `token_spans.v1`. The service
receives `{"inputs": "<request text>"}` and answers with entity spans whose
`start`/`end` are Unicode code-point offsets into that exact string, a `label`
from the configured PII mapping, a `score` in `[0, 1]`, and the span `text`,
which must equal the slice it points at. The HuggingFace token-classification
spellings `entity_group` and `word` are accepted as aliases. A bare JSON list of
spans or an envelope `{"spans": [...], "truncated_at": n, "model": "..."}` are
both valid; the envelope's `model`, when present, must equal the catalog
entry's `llm_model_name`.

The router rejects the whole response, rather than part of it, when a span is
outside the text, overlaps itself, carries an unknown or outside label, has a
score out of range, has conflicting alias values, or when the body is not a span
list. A declared `truncated_at` keeps the spans before the cut and marks the
rest of the content as unscored. What a rejected or partial response does to a
PII rule is `on_error`: `allow` (default) treats the unread content as not
matching, `block` matches it as `classification_error`, so unverified text
cannot pass as clean.

Spans returned before a declared cut are real detections under both policies. A
rule that matched on one of them stays a genuine match even when the rest of
its content was never read, so a decision using `rules.on_unknown: no_match`
still sees it; only a match that exists solely because the scan failed is
unknown to the decision engine. The detection API says the same thing with
`scan_incomplete: true` beside its entities, so `has_pii: false` after a
truncation reads as "nothing in the part that was read" rather than as a clean
scan.

```yaml
global:
  model_catalog:
    external:
      - name: pii-service
        model_role: classification
        llm_endpoint:
          address: pii-spans.default.svc
          port: 8080
        llm_model_name: pii-spans-v1
    modules:
      classifier:
        pii:
          backend:
            protocol: http_classify
            contract: token_spans.v1
            model: pii-service
            deadline_ms: 5000
          on_error: block
```

`PIIDetected`, `PIIEntities`, `MatchedPIIRules` and the masked text are the
same whether the spans came from the local model or from a remote backend;
overlapping and nested spans are merged before masking.

## Dependencies and Limitations

The PII classifier processes the prompt and optional history. It is a routing
control, not a substitute for redaction, encryption, access control, or data
loss prevention. Calibrate thresholds by entity type. See a complete example:
[`config/fragments/signal/pii/strict.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/pii/strict.yaml).

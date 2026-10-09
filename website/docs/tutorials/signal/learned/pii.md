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
`threshold` is optional: a rule without one takes every span the PII model
reports, and a Vela 2.0 model reports only spans above its size's calibrated
threshold.

## Complete local scans

The default PII model, Vela 2.0 0.3B, reads each text item whole, up to its
8,192-token input, and finds spans with its router span head. When the module
runs Vela 1.0 PII, it scans each text item up to 32,768 tokens, including
special tokens. Each forward uses at most 512 tokens, with 255 content tokens
of overlap. The model tokenizer defines the windows; character estimates
and text re-tokenization at window boundaries do not determine coverage.

The [model runtime](../../../model-runtime/guides/pii.md) reports spans as
character offsets into the original text, chooses one observation per token by
its surrounding context, then decodes BIO entities once. Overlap does not
double-count input usage or entity confidence. This guarantees coverage of
admitted tokens, not detection accuracy or the quality of a single 32K forward.

An explicit module budget, backend, window, or recipe binding keeps its own
policy. For example, a deployment with `input: {max_tokens: 8192, overflow: reject}`
still rejects an oversized input. For a checkpoint and graph qualified for 32K
forwards, a 64K document budget can use:

```yaml
global:
  model_catalog:
    modules:
      classifier:
        pii:
          max_sequence_length: 65536  # Complete text budget, including special tokens.
          window: {size: 32768, overlap: 256}
```

With a named binding, declare `input.overflow: window` and a positive
`input.max_tokens` on its deployment; that limit replaces the module budget.
The same `window` block supplies the geometry. The document budget can exceed
the loaded model's single-forward capacity; each window must fit that capacity
and the document budget. Unsupported adapters, missing window geometry, and
oversized windows are errors. Window size includes the classifier tokenizer's
special tokens; overlap counts content only. These token counts need not match
the downstream generative model's tokenizer.

A text beyond the document limit or a failed window produces a classifier error,
not a successful partial scan. Existing `on_error` and decision `rules.on_unknown`
policies determine its routing effect. Remote backends and explicitly selected
truncation retain the partial-result behavior described below.

## Vela 2.0

Bind `pii_classifier` to a Vela 2.0 deployment and the signal asks the model's
ready-made PII question, answered by its router span head, instead of a
separate PII model. The PII question then travels in the same call as the
deployment's [`decision`](tutorials/signal/learned/decision.md) questions about the same text:

```yaml
global:
  model_catalog:
    deployments:
      vela2:
        provider: model_runtime
        artifact: vllm-sr/Vela-2.0-0.3B
        device: cpu
    bindings:
      pii_classifier:
        deployment: vela2
        contract: token_spans.v1
```

- The model finds the same 17 entity types as Vela 1.0 PII and names them in
  its spans, so the binding takes no `mapping_path`.
- It reads the whole text itself, a long one in windows, so the router sends
  each text in one piece: the deployment sets no `input` and the PII module no
  `window`.
- The model's calibrated threshold decides which spans it reports. A rule's
  `threshold` and `pii_types_allowed` then apply to those spans as they do for
  Vela 1.0; the model's span probability is the mean over the span's words.
- `head` may only be `router`, the span head that answers the PII question.

Existing configurations keep their Vela 1.0 PII binding.

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
cannot pass as clean. Content the model did not read at all (an input over the
model's input or [scan cap](../../../model-runtime/reference.md#long-inputs),
a truncated one, or one not scanned by the signals' deadline) matches as
`unscanned` whatever `on_error` says, so a long request routes as private;
set `classifier.pii.on_unscanned: allow` to leave it to `on_error`.

The PII mapping cannot declare `classification_error` as an entity label.
Aliases with `B-`, `I-`, or `E-` prefixes, including stacked prefixes, are also
reserved and rejected at mapping load time in either mapping direction.

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

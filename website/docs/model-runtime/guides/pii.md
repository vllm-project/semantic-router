---
title: Detect PII
description: Find personal information in requests with exact character spans, and route or block on it.
---

# Detect PII

Vela 1.0 PII finds personal information in a request and returns where it is:
the type, the exact characters and a probability for every span. A
[`pii` signal](tutorials/signal/learned/pii.md) matches when it finds a
type you do not allow, and a route can then keep the request on a private
model or refuse it.

The model knows 17 types: `AGE`, `CREDIT_CARD`, `DATE_TIME`, `DOMAIN_NAME`,
`EMAIL_ADDRESS`, `GPE` (places), `IBAN_CODE`, `IP_ADDRESS`, `NRP`
(nationality, religion or political group), `ORGANIZATION`, `PERSON`,
`PHONE_NUMBER`, `STREET_ADDRESS`, `TITLE`, `US_DRIVER_LICENSE`, `US_SSN` and
`ZIP_CODE`.

## Turn it on

```yaml
routing:
  signals:
    pii:
      - name: personal_data
        threshold: 0.85
        pii_types_allowed: [EMAIL_ADDRESS]
  decisions:
    - name: keep-private
      priority: 200
      rules:
        operator: AND
        conditions:
          - type: pii
            name: personal_data
      modelRefs:
        - model: private-model
```

The router runs Vela PII on the CPU. It reads the whole request, up to 32,768
tokens, in overlapping 512-token windows, so a name near the end of a long
document is found as reliably as one at the start.

## Choose where it runs

The binding is `pii_classifier` and reads text spans (`token_spans.v1`):

```yaml
global:
  model_catalog:
    deployments:
      vela-pii:
        provider: model_runtime
        artifact: vllm-sr/Vela-1.0-Encoder-307M-PII
        device: cpu
        input:
          max_tokens: 32768
          overflow: window
    bindings:
      pii_classifier:
        deployment: vela-pii
        contract: token_spans.v1
    modules:
      classifier:
        pii:
          window:
            size: 512
            overlap: 255
```

Keep `overflow: window` for PII: the model reads the whole request in
512-token windows that overlap by 255 tokens. `truncate` would scan only the
beginning, and `reject` makes the signal unknown for longer requests.

### On Vela 2.0

Vela 2.0 finds the same 17 types with its router span head, as a ready-made
question. Bind `pii_classifier` to a Vela 2.0 deployment and the PII question
travels in the same call as the [decision questions](model-runtime/guides/decisions.md) the
request asks that deployment about the same text:

```yaml alternative
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

The model reads the whole request itself, so the deployment takes no `input`
and the PII module no `window`. See the
[`pii` signal](tutorials/signal/learned/pii.md#vela-20) for how rule
thresholds apply to its spans.

## When the scan cannot finish

If the model is not ready or a scan fails, the signal is unknown. The PII
module's `on_error` decides what that means: `allow` (default) treats the
unread text as clean, and `block` matches it as `classification_error`, so
text that could not be checked cannot pass as clean.

```yaml
global:
  model_catalog:
    modules:
      classifier:
        pii:
          on_error: block
```

## Check it

```bash
vllm-sr serve vllm-sr/Vela-1.0-Encoder-307M-PII --device cpu --port 8100
curl -s localhost:8100/v1/classify -H 'content-type: application/json' \
  -d '{"input": ["Hi, I am Tom Baker, write to tom.baker@example.com."]}'
```

Each span has its `label`, `start` and `end` (in characters, end exclusive),
the `text` and its `probability`. Through the router, `x-vsr-matched-pii`
lists the PII rules that matched.

On Vela 2.0, ask the ready-made question on `/v1/decisions`; the spans are in
`spans.pii`:

```bash
curl -s localhost:8100/v1/decisions -H 'content-type: application/json' -d '{
  "state": "Hi, I am Tom Baker, write to tom.baker@example.com.",
  "questions": {"pii": {"preset": "pii"}}
}'
```

---
title: Detect PII
description: Find personal information in requests with exact character spans, and route or block on it.
---

# Detect PII

The pii preset finds personal information in a request and returns where it
is: the type, the exact characters and a probability for every span. A
[`pii` signal](tutorials/signal/learned/pii.md) matches when it finds a type
it counts, and a route can then keep the request on a private model or refuse
it.

The preset knows 17 types: `AGE`, `CREDIT_CARD`, `DATE_TIME`, `DOMAIN_NAME`,
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
        predicate: {gte: 0.85}
        ignore_labels: [EMAIL_ADDRESS]
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

Without a binding, the Router asks the decision model, vela3-fast by default,
for PII; see [Choose a model](../choose-a-model#built-in-signals). A rule's
`predicate` is explicit policy: calibrate this example's value for the model
you select, or omit it to use the model's calibrated threshold. The
`ignore_labels` are the types the rule does not count.

## Choose where it runs

The binding key is `pii`, and the signal reads text spans, so its deployment
must answer span questions; one that cannot is refused when the configuration
loads. Vela 2.0 finds the same 17 types with its router span head, as a
ready-made question. Bind `pii` to a Vela 2.0 deployment and the PII question
travels in the same call as the
[System One questions](model-runtime/guides/decisions.md) the request asks that
deployment about the same text:

```yaml
global:
  model_catalog:
    deployments:
      vela2:
        provider: model_runtime
        artifact: vllm-sr/Vela-2.0-0.3B
        device: cpu
    bindings:
      pii: vela2
```

Vela 2.0 uses its own full-input checks and supported scan budget. An `input`
override can set the deployment's scan budget; see
[Long inputs](../reference.md#long-inputs).

## When the scan cannot finish

Text the model did not read (over its input or
[scan cap](model-runtime/reference.md#long-inputs), truncated, or not scanned
by the signals' deadline) counts as personal data, so text that could not be
checked cannot pass as clean. If the model is not ready or a scan fails, the
signal is unknown; decide what that means per route with `rules.on_unknown`:

```yaml
routing:
  signals:
    pii:
      - name: personal_data
  decisions:
    - name: keep-private
      priority: 200
      rules:
        operator: AND
        on_unknown: fail_request
        conditions:
          - type: pii
            name: personal_data
      modelRefs:
        - model: private-model
```

## Check it

These worker-level examples run inside an environment containing `vllm-srun`
(such as the Router image). Classify, embeddings, rerank and bundle are worker
APIs; the instance frontend publishes System One and decision requests.

```bash
vllm-srun serve vllm-sr/Vela-1.0-Encoder-307M-PII --device cpu --port 8100
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

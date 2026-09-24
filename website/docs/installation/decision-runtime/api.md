---
title: Call the API
description: Choose the single-state or shared-question batch endpoint and handle backpressure.
---

The runtime serves one named model. Both inference routes require that exact
model ID; unknown fields and mismatched question answers are rejected. The
[Decision Runtime API reference](../../api/decision-runtime.md) gives full
request and response examples for Noul, Choice, and Score questions.

## One state, one or more questions

Send `POST /v1/systemone` when several questions describe the **same** state:

```bash
curl -sS http://127.0.0.1:8001/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "llm-semantic-router/Decision-1.0-Kai-0.6B",
    "state": "Please refund the duplicate charge.",
    "questions": {
      "refund_requested": {
        "type": "noul",
        "instructions": "Does the customer explicitly request a refund?"
      }
    }
  }'
```

The `answers.refund_requested.noul` value is the probability of true, not a
preselected action threshold. The response also includes `model` and `usage`.

## Several states, shared questions

Send `POST /v1/decision/batches` when the same question map applies to
multiple identified states. It returns ordered `results`, each with the
caller-supplied state ID, answers, and usage, plus aggregate usage. This is a
Decision-specific extension—not `/v1/systemone/batch`—and changes HTTP
overhead, not answer semantics. State IDs must be unique, and state count ×
question count cannot exceed 1,024 decisions.

## Handle admission and diagnostics

Single-request JSON is capped at 256 KiB, batch JSON at 2 MiB, and expanded
rendered input at 16 MiB. Expect `413` for oversized input, `422` for an
invalid request or model, `529` with `Retry-After` when the runtime is full,
and `503` when the backend is unavailable. Honor backpressure; blind retries
can keep the queue saturated.

Use `GET /ready` to test resident readiness, `GET /v1/models` to confirm the
pinned model, and `GET /openapi.json` for the exact live schema. Successful
inference includes artifact identity response headers; `GET /api/status` and
`GET /metrics` expose detailed identity, scheduling, and telemetry. Protect
those diagnostic routes when publishing an inference endpoint.

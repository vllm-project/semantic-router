---
title: Decision Runtime API reference
sidebar_label: Decision Runtime
description: Request fields, answer shapes, limits, and errors for Decision Runtime.
---

# Decision Runtime API reference

Start with the [quickstart](../installation/decision-runtime/overview.md) and
the [cURL and SDK examples](../installation/decision-runtime/api.md) if you
are making your first request. This page is the field-level reference for the
runtime's HTTP API.

Each `decision serve` instance serves **one model** on its own port. Every inference
request must use that instance's exact model ID; the runtime does not choose a
default. `GET /v1/models` returns the ID to send.

| Endpoint | Use it when | Status |
| --- | --- | --- |
| `POST /v1/systemone` | One state has one or more questions. | SystemOne single-state format |
| `POST /v1/systemone/batches` | Several states have the same question set. | Decision Runtime extension using SystemOne questions and answers |

The SystemOne path and JSON shape support the official
[TypeSafe SDKs](../installation/decision-runtime/api.md#use-the-official-systemone-sdks)
when configured with this runtime's base URL and full model ID. Local
validation requires `instructions` for all three question types and rejects
unknown fields;
Choice and Score confidence use Decision's own calculation.
The `/v1/systemone/batches` path groups several evaluations under one shared
question map. Its `states` and `results` envelope is a Decision Runtime
extension; the official SystemOne SDKs call only the single-state endpoint.

## Single state: `POST /v1/systemone`

```json
{
  "model": "llm-semantic-router/Decision-1.0-Kai-0.6B",
  "state": "Please refund the duplicate charge.",
  "questions": {
    "refund_requested": {
      "type": "noul",
      "instructions": "Does the customer explicitly request a refund?"
    }
  }
}
```

| Request field | Required | Meaning |
| --- | --- | --- |
| `model` | Yes | Exact model ID served by this instance. |
| `state` | Yes | Nonblank text or structured JSON content to evaluate. |
| `questions` | Yes | Map from your question IDs to 1–1,024 questions about this state. |

Each question has a `type` and required, non-null `instructions`. Instruction
text and question IDs must not be blank. The question types are:

| Type | `criteria` | Answer |
| --- | --- | --- |
| `noul` | Optional descriptions under `true` and `false`. | `noul`: probability that the answer is true (0–1). |
| `choice` | 2–255 named options, each with a description or `null`. | `choice`: selected option key; `probabilities`: distribution by option key; `confidence`. |
| `score` | 2–10 ordered rubric levels. | `score`: expected zero-based rubric position; `legend`: position-to-level map; `probabilities`; `confidence`. |

`instructions` and criterion descriptions can be text or structured JSON
content where the schema permits it. The names under `answers` are exactly
the question IDs you sent. A successful response has only these top-level
fields:

| Response field | Meaning |
| --- | --- |
| `model` | The model that produced the answers. |
| `answers` | One typed answer for every requested question. |
| `usage` | `input_tokens` and `output_tokens` for this request. |

Choice and Score `confidence` are Decision-defined statistics derived from
their distributions, **not** calibrated correctness probabilities or a claim
of numeric equivalence with TypeSafe/Jev. Noul has no separate confidence
field. See the [call guide](../installation/decision-runtime/api.md) for an
example covering all three question types.

## Shared-question batch: `POST /v1/systemone/batches`

```json
{
  "model": "llm-semantic-router/Decision-1.0-Kai-0.6B",
  "states": [
    {"id": "case-a", "state": "Please refund the duplicate charge."},
    {"id": "case-b", "state": "Can you explain this invoice?"}
  ],
  "questions": {
    "refund_requested": {
      "type": "noul",
      "instructions": "Does the customer explicitly request a refund?"
    }
  }
}
```

The batch evaluates every state against every question. `model` is the same
explicit model ID used by `/v1/systemone`, and `questions` uses the same
question IDs, types, criteria, and instruction rules. Replace the single
`state` field with a nonempty `states` array: each element has a unique,
nonblank `id` (at most 128 characters) and a `state` with the same content
rules as a single request. The product of state count and question count may
not exceed **1,024 decisions**.

The response uses the same typed `answers` and per-request token `usage` as
the single-state API, grouped in an ordered result for each input state:

```json
{
  "model": "llm-semantic-router/Decision-1.0-Kai-0.6B",
  "results": [
    {"id": "case-a", "answers": {"refund_requested": {"type": "noul", "noul": 0.91}}, "usage": {"input_tokens": 18, "output_tokens": 0}},
    {"id": "case-b", "answers": {"refund_requested": {"type": "noul", "noul": 0.08}}, "usage": {"input_tokens": 17, "output_tokens": 0}}
  ],
  "usage": {"input_tokens": 35, "output_tokens": 0}
}
```

Each result's `id` matches the corresponding input state; `results` follows
input order. Top-level `usage` totals the per-state usage. The values above
illustrate the response shape, not model predictions.

This endpoint reduces repeated HTTP and question-formatting overhead when
several states share questions. It is a Decision Runtime extension to the
SystemOne single-state API, not an official TypeSafe SDK method; call it with
an HTTP client.

## Limits and errors

| Status | Meaning | Client action |
| --- | --- | --- |
| `413` | Request body, expanded input, or model token limit exceeded. | Reduce or split the input. |
| `422` | Invalid request, unknown field, or model not served here. | Check the schema and `GET /v1/models`. |
| `529` | Runtime admission is full. | Honor `Retry-After` before retrying. |
| `503` | Backend unavailable. | Check readiness before retrying. |

Single-state JSON bodies are capped at **256 KiB**, batch bodies at **2 MiB**,
and expanded rendered input at **16 MiB**. The model may impose a lower token
limit. Do not retry invalid or oversized requests unchanged.

## Readiness and diagnostics

| Endpoint | Returns |
| --- | --- |
| `GET /health` | Process liveness. |
| `GET /ready` | Whether the resident model is available (`503` until ready). |
| `GET /v1/models` | The model served by this instance. |
| `GET /openapi.json` | The exact schema of the running version. |
| `GET /api/status` | Artifact and scheduler diagnostics. |
| `GET /metrics` | Prometheus-format metrics. |

Successful inference responses may carry artifact identity in HTTP headers;
diagnostic data is not added to the SystemOne or batch JSON. Protect
`/api/status` and `/metrics` when publishing a public endpoint. For instance
setup and capacity controls, see [launch parameters](../installation/decision-runtime/parameters.md).

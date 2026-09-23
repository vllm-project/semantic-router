---
title: Decision Runtime API
sidebar_label: Decision Runtime
description: Serve Decision 1.0 models and call the strict single-state API or the separate shared-question batch extension.
---

# Decision Runtime API

`vllm-sr drun` starts one pinned Decision 1.0 model per process. Give each
instance a different port to serve multiple models on one machine. The runtime
does not select a default model: every request must name the model served by
that instance. See the [`drun` command reference](./cli#vllm-sr-drun-run) for
the available backend, image, capacity, and lifecycle options.

```bash
vllm-sr drun run llm-semantic-router/Decision-1.0-Kai-0.6B \
  --backend rocm --port 8001 --detach
```

For isolated validation with an already-loaded Docker image, pass its full
`sha256:` image ID with `--image` and set `--image-pull-policy never`. The ID
must exist locally and is never pulled; abbreviated IDs and Podman are not
accepted for this override. Published `repository@sha256:` image references
retain the normal pull-policy behavior.

The selected model and hardware pair must have a qualified release profile;
`drun` refuses unsupported combinations. In particular, the presence of a
backend option does not by itself mean every Decision model is qualified on
that hardware.

## One state: `POST /v1/systemone`

This is the strict SystemOne-compatible single-state endpoint. One `state` is
evaluated against one or more named `questions`. The three supported question
types are `noul` (yes/no), `choice` (2–255 named options), and `score` (2–10
ordered rubric levels). There is no implicit model and no batch form at this
path.

```bash
curl -sS http://127.0.0.1:8001/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "llm-semantic-router/Decision-1.0-Kai-0.6B",
    "state": "I was charged twice for order A1842. Please refund the duplicate payment.",
    "questions": {
      "refund_requested": {
        "type": "noul",
        "instructions": "Does the customer explicitly request a refund?"
      },
      "category": {
        "type": "choice",
        "instructions": "Classify the support request.",
        "criteria": {
          "billing": "Charges, invoices, or refunds",
          "product": "Product use or defects"
        }
      },
      "urgency": {
        "type": "score",
        "instructions": "Rate the urgency of this request.",
        "criteria": ["Routine", "Needs prompt attention", "Critical"]
      }
    }
  }'
```

Successful responses contain only `model`, `answers`, and `usage`. A Noul
answer contains `type` and `noul` (the probability of true). A Choice answer
contains `type`, `choice`, `confidence`, and `probabilities`. A Score answer
contains `type`, `score`, `confidence`, `legend`, and `probabilities`; `score` is
the expected zero-based rubric index. Request and response JSON are strict:
unknown fields are rejected, and answers must match the requested question
names and criteria.

Decision confidence is a versioned, type-aware statistic, not a claim of
numeric equivalence to another provider. For Choice it is the top-two
probability margin; for Score it is one minus normalized ordinal variance.
Noul exposes its true probability directly and has no separate confidence
field. The runtime computes these from unrounded probabilities before
serializing the response.

## Multiple states: `POST /v1/decision/batches`

This Decision-specific extension evaluates identified states against the same
model and question map. It is distinct from multiple questions on one state:
the single-state API shares a state across questions, while this endpoint
shares questions across states. It is not an alias of `/v1/systemone` and does
not introduce a `/v1/systemone/batch` route.

```bash
curl -sS http://127.0.0.1:8001/v1/decision/batches \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "llm-semantic-router/Decision-1.0-Kai-0.6B",
    "states": [
      {"id": "case-a", "state": "I was charged twice. Please refund the duplicate."},
      {"id": "case-b", "state": "Can you explain the new invoice?"}
    ],
    "questions": {
      "refund_requested": {
        "type": "noul",
        "instructions": "Does the customer explicitly request a refund?"
      }
    }
  }'
```

The response contains `model`, ordered `results`, and aggregate `usage`.
Each result repeats its caller-supplied `id` and contains `answers` and
per-state `usage`. State IDs must be unique. The runtime validates the whole
batch before evaluating it; the product of state count and question count may
not exceed 1,024 decisions. A batch can reduce repeated request overhead, but
it does not change the meaning of an individual answer.

## Service and diagnostics

`GET /ready` reports whether the pinned model is available. `GET /v1/models`
lists the model served by the instance. Runtime diagnostics are separate from
normal inference responses: `GET /api/status` reports model and scheduler
status, and `GET /metrics` exposes Prometheus-format metrics. Keep these
diagnostic routes on a protected control-plane listener or allowlist when
placing the runtime behind a public Gateway.

Malformed requests return `422`; oversized inputs return `413`; a saturated
runtime returns `529` with `Retry-After`; an unavailable backend returns
`503`. Clients should honor backpressure rather than retrying an overloaded
instance without delay. The live service publishes its exact OpenAPI schema
at `GET /openapi.json`.

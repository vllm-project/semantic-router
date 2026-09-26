---
title: Call the Decision API
description: Ask Noul, Choice, and Score questions with cURL or a SystemOne client.
---

Once [a model is running](./overview.md), send a **state** (the situation) and
one or more named **questions** to `POST /v1/systemone`. The model answers all
questions about that same state in one request.

This is the [TypeSafe SystemOne](https://docs.typesafe.ai/api) single-state
request and response format and path. Use the full **Decision model ID** served
by this instance. Noul, Choice, and Score all require `instructions`. Make
each question clear so the model can evaluate the intended decision.
Decision makes no claim that its confidence matches Jev's number. The separate
[multi-state batch endpoint](#many-states-one-question-set) is a Decision
extension, not part of SystemOne compatibility.

First, confirm the exact model ID accepted by the instance:

```bash
curl -fsS http://127.0.0.1:8001/v1/models
```

Copy a value from `models[].name` into the `model` field below. A Decision
instance serves one model; a different ID returns `422` rather than silently
selecting another model.

## One state, three question types

This example asks a yes/no question, selects a category, and rates urgency in
one call. You can copy it as-is after starting Kai on port 8001.

```bash
curl --fail-with-body -sS http://127.0.0.1:8001/v1/systemone \
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
        "instructions": "Classify this support request.",
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

For example, a response can look like this (**illustrative values**, not a
prediction for the message above):

```json
{
  "model": "llm-semantic-router/Decision-1.0-Kai-0.6B",
  "answers": {
    "refund_requested": {"type": "noul", "noul": 0.91},
    "category": {
      "type": "choice",
      "choice": "billing",
      "confidence": 0.8,
      "probabilities": {"billing": 0.9, "product": 0.1}
    },
    "urgency": {
      "type": "score",
      "score": 1.0,
      "confidence": 0.7,
      "legend": {"0": "Routine", "1": "Needs prompt attention", "2": "Critical"},
      "probabilities": {"0": 0.1, "1": 0.8, "2": 0.1}
    }
  },
  "usage": {"input_tokens": 54, "output_tokens": 0}
}
```

The response has three top-level fields: `model`, `answers`, and `usage`.
Each key under `answers` matches one of your question names:

| Question | Read this response field | Meaning |
| --- | --- | --- |
| Noul (`refund_requested`) | `answers.refund_requested.noul` | Probability of **yes** (0–1); your application chooses an action threshold. |
| Choice (`category`) | `answers.category.choice` | Selected option key, such as `billing`; `probabilities` contains the option distribution. |
| Score (`urgency`) | `answers.urgency.score` | Expected position on the **zero-based** ordered rubric; `legend` maps positions to your criteria. |

Choice and Score also return `confidence`. It is a Decision-defined statistic,
not a claim that the number matches another provider's confidence calculation.
Do not treat it as a calibrated probability of correctness. Noul has no
separate `confidence` field.

Every question needs a non-null `instructions` value: text or structured JSON
content. Choice accepts 2–255 named options, and Score accepts 2–10 ordered
levels. These limits apply to both the single-state and batch endpoints.

:::tip Make the URL match your deployment
Use `http://127.0.0.1:8001` for a local instance, or replace it with your
Gateway's API base URL. The base URL ends **before** `/v1/systemone`. The
runtime itself does not enforce an API key; if your Gateway requires one, add
its `Authorization: Bearer ...` header.
:::

## Use the official SystemOne SDKs

The official [Python SDK](https://github.com/typesafe-ai/typesafe-sdk-python)
and [JavaScript SDK](https://github.com/typesafe-ai/typesafe-sdk-js) both call
`/v1/systemone`. Configure their API base URL to point to your Decision
instance or Gateway and set the model explicitly: their default model is not
a Decision model.

Python (`pip install typesafe-sdk`):

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

with TypeSafeClient(
    base_url="http://127.0.0.1:8001",
    api_key="local-only",
    model="llm-semantic-router/Decision-1.0-Kai-0.6B",
) as client:
    result = client.system_one(
        state="I was charged twice. Please refund the duplicate payment.",
        questions={
            "refund_requested": Noul(
                instructions="Does the customer explicitly request a refund?"
            ),
            "category": Choice(
                instructions="Classify this support request.",
                criteria={
                    "billing": "Charges, invoices, or refunds",
                    "product": "Product use or defects",
                },
            ),
            "urgency": Score(
                instructions="Rate the urgency of this request.",
                criteria=["Routine", "Needs prompt attention", "Critical"],
            ),
        },
    )
    print(result.nouls["refund_requested"].noul)
    print(result.choices["category"].choice)
    print(result.scores["urgency"].score)
```

JavaScript/TypeScript (`npm install @typesafe-ai/sdk`):

```ts
import { choice, noul, score, TypeSafeClient } from "@typesafe-ai/sdk";

const client = new TypeSafeClient({
  baseURL: "http://127.0.0.1:8001",
  apiKey: "local-only",
  defaultModel: "llm-semantic-router/Decision-1.0-Kai-0.6B",
});
const result = await client.systemOne({
  state: "I was charged twice. Please refund the duplicate payment.",
  questions: {
    refund_requested: noul("Does the customer explicitly request a refund?"),
    category: choice("Classify this support request.", {
      billing: "Charges, invoices, or refunds",
      product: "Product use or defects",
    }),
    urgency: score("Rate the urgency of this request.", [
      "Routine",
      "Needs prompt attention",
      "Critical",
    ]),
  },
});
console.log(result.answers.refund_requested.noul);
console.log(result.answers.category.choice);
console.log(result.answers.urgency.score);
```

These SDKs require an API-key value and send a Bearer header. `local-only` is
only an SDK-required placeholder for **direct local testing**; the local
runtime has no authentication middleware. Use a real Gateway credential for
a protected deployment. The Python `result.nouls`, `result.choices`, and
`result.scores` convenience views are not fields in the wire JSON; the
response JSON uses `answers`.

## Many states, one question set

Use `POST /v1/systemone/batches` when the **same questions** apply to several
states. This shares the question set across states; multiple questions on one
state are already supported by `/v1/systemone`.

```bash
curl --fail-with-body -sS http://127.0.0.1:8001/v1/systemone/batches \
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

An illustrative response shows how each input state becomes one result:

```json
{
  "model": "llm-semantic-router/Decision-1.0-Kai-0.6B",
  "results": [
    {
      "id": "case-a",
      "answers": {"refund_requested": {"type": "noul", "noul": 0.91}},
      "usage": {"input_tokens": 18, "output_tokens": 0}
    },
    {
      "id": "case-b",
      "answers": {"refund_requested": {"type": "noul", "noul": 0.08}},
      "usage": {"input_tokens": 17, "output_tokens": 0}
    }
  ],
  "usage": {"input_tokens": 35, "output_tokens": 0}
}
```

The `questions` object uses the same types and IDs as a single-state request.
The same instruction and criteria rules apply.
Read `results` in the response: one entry per input state, in input order,
with its `id`, typed `answers`, and `usage`. The response also has aggregate
`usage`. State IDs must be unique; `number of states × number of questions`
cannot exceed 1,024. The `states` and `results` envelope is a Decision Runtime
extension, **not** an official TypeSafe SDK method; use HTTP directly for this
endpoint.

## Check and troubleshoot

- `GET /ready`: the model is loaded and ready to answer.
- `GET /v1/models`: the exact model ID this instance serves.
- `GET /openapi.json`: the live request and response schema.
- `422`: invalid payload or wrong model ID; fix the request.
- `413`: payload too large; split it into smaller calls.
- `529` with `Retry-After`: capacity is full; wait before retrying.
- `503`: backend unavailable; check instance health.

For complete field definitions and limits, see the [API reference](../../api/decision-runtime.md).
For request flow and deployment boundaries, see [How Decision Runtime works](./architecture.md).

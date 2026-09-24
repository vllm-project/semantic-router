---
title: Get started with Decision Runtime
description: Start a Decision 1.0 model and make your first decision request.
---

Decision Runtime turns a Decision 1.0 model into a small HTTP service. You give
it a **state** (the situation to evaluate) and one or more **questions**; it
returns a typed answer for each question. Use `vllm-sr decision serve` to run
one model per instance. To serve more models, start more instances on different ports.
You need the vLLM Semantic Router CLI and Docker or Podman; Decision Runtime is
standalone and does not require `vllm-sr serve`.

## 1. Start a model

:::note Availability
The first CLI release with Decision Runtime is being prepared. The command
below is the installed-release flow; contributors testing the current source
checkout can use the [local build guide](https://github.com/vllm-project/semantic-router/tree/main/src/vllm-sr/decision_runtime/image).
:::

This example starts Kai on port 8001 and keeps it running in the background:

```bash
vllm-sr decision serve llm-semantic-router/Decision-1.0-Kai-0.6B \
  --port 8001 --instance-name kai-demo --detach
```

`decision serve` detects the available backend when `--backend` is omitted and
uses the matching image recorded by that CLI release. Check the
[model and backend table](./models.md) for supported combinations.

Wait for the model to load, then check readiness:

```bash
curl --fail-with-body -sS http://127.0.0.1:8001/ready
```

## 2. Ask a question

```bash
curl --fail-with-body -sS http://127.0.0.1:8001/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "llm-semantic-router/Decision-1.0-Kai-0.6B",
    "state": "I was charged twice. Please refund the duplicate payment.",
    "questions": {
      "refund_requested": {
        "type": "noul",
        "instructions": "Does the customer explicitly request a refund?"
      }
    }
  }'
```

Look for `answers.refund_requested.noul` in the JSON response. It is the
model's probability for **yes**; your application chooses the threshold or
action. Every request must name the model running on that port.

For Choice and Score questions, response examples, Python/JavaScript clients,
and the difference between SystemOne and our batch extension, continue to
[Call the API](./api.md).

## 3. Manage the instance

```bash
vllm-sr decision status kai-demo
vllm-sr decision stop kai-demo
```

`status` tells you which model and revision are loaded. `stop` shuts down the
managed instance. To run two models at once, repeat step 1 with another model,
instance name, and port.

Next: [choose a model](./models.md), [set capacity and hardware options](./parameters.md),
or [see how a request flows through the runtime](./architecture.md).

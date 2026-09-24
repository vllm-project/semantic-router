---
title: Get started with Decision Runtime
description: Start a Decision 1.0 model and make your first decision request.
---

Decision Runtime turns a Decision 1.0 model into a small HTTP service. You give
it a **state** (the situation to evaluate) and one or more **questions**; it
returns a typed answer for each question. Use `vllm-sr drun` to run one model
per instance. To serve more models, start more instances on different ports.
You need the vLLM Semantic Router CLI and Docker or Podman; `drun` is
standalone and does not require `vllm-sr serve`.

## 1. Start a model

:::note Preview source builds
The command below assumes an installed release with a published Decision
Runtime image for your device. The current source checkout has no default
image yet; contributors can follow the
[image build guide](https://github.com/vllm-project/semantic-router/tree/main/src/vllm-sr/decision_runtime/image)
to validate an unreleased build.
:::

This example starts Kai on port 8001 and keeps it running in the background:

```bash
vllm-sr drun llm-semantic-router/Decision-1.0-Kai-0.6B \
  --port 8001 --instance-name kai-demo --detach
```

`drun` detects the available backend by default. Check the
[model and backend table](./models.md) for what this build can run.

Wait for the model to load, then check readiness:

```bash
curl -fsS http://127.0.0.1:8001/ready
```

## 2. Ask a question

```bash
curl -sS http://127.0.0.1:8001/v1/systemone \
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
vllm-sr drun status kai-demo
vllm-sr drun stop kai-demo
```

`status` tells you which model and revision are loaded. `stop` shuts down the
managed instance. To run two models at once, repeat step 1 with another model,
instance name, and port.

Next: [choose a model](./models.md), [set capacity and hardware options](./parameters.md),
or [see how a request flows through the runtime](./architecture.md).

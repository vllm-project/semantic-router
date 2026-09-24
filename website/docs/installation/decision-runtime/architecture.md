---
title: How Decision Runtime works
description: See how a request reaches one Decision model and returns an answer.
---

`vllm-sr drun` starts a standalone HTTP service for one Decision model. Your
application, SDK, or Gateway sends a request to that service's port. The
runtime checks the request, scores its questions with the loaded model, and
returns the answers under the same question IDs.

![Your app sends a SystemOne or batch request to the drun API. The API checks the request, a resident model scores the questions, and the answers return to your app.](/img/decision-runtime-flow.svg)

## Follow one request

1. **Send the state and questions.** Use `POST /v1/systemone` for one state or
   `POST /v1/decision/batches` for several states that share the same questions.
   Include the exact model ID of the running instance. See [call the API](./api.md)
   for request and response examples.
2. **Check and admit the request.** The API validates the model ID and request
   shape. When the instance is busy, it waits within its configured queue or
   returns `529` with `Retry-After` if it cannot admit more work.
3. **Score the questions.** The instance keeps one model loaded on its selected
   device. Each state-and-question pair is one decision. Compatible decisions
   from concurrent requests can share a model forward pass; the runtime maps
   every score back to its original state and question.
4. **Read the answer.** The response preserves your question IDs and includes
   model and usage information. Decision models score possible answers for
   each question; they do not generate an open-ended chat reply. Your application
   decides how to use those scores.

## Serve more than one model

Each `drun` instance has **one model, one revision, and one port**. To serve Kai
and Sol, start two instances on different ports, then send each request to the
port for its chosen model. The HTTP contract stays the same across models; the
model ID in each request must match the instance receiving it.

At startup, `drun` checks the selected model files and whether the model can
run on the chosen backend. `--backend auto` detects the host backend; an
explicit `--backend` requests one. If the model cannot run there, startup
fails. `/ready` succeeds only after the model is loaded. See
[models and backends](./models.md) for the current support matrix and
[launch parameters](./parameters.md) for the available controls.

## How batching helps

The batch API saves HTTP overhead when multiple states use the same questions.
Separately, the runtime can combine compatible decisions from different
requests into a bounded model batch. Batch results follow the caller's state
order, and answers retain their question IDs. `--max-batch` limits work in one model pass;
`--max-concurrency` and `--max-queue` control requests admitted or waiting.
See [performance tuning](./optimization.md) when those limits matter for your
workload.

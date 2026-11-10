---
title: Readiness and monitoring
sidebar_label: Readiness and monitoring
sidebar_position: 4
description: When the Router waits for its models, what a feature sees while a model is not ready, and the metrics and inventory that show what is running.
---

# Readiness and monitoring

The Router serves a configuration only once the models it runs for it have
loaded. After that, requests never wait for a model that cannot answer: an
unready model leaves its signals unknown, and each rule's policy decides what
that means.

```bash title="What is loading, and what is ready"
curl -s localhost:8080/startup-status
curl -s localhost:9190/metrics | grep '^vsr_model_runtime_ready'
```

## When a model is not ready

### At startup

At startup the Router waits for every deployment it manages: the task models
its routes use (domain, PII, guard, safety, fact-check, feedback and
hallucination models, and your own classifiers) and the decision models that
`systemone` signals and the `systemone` selection algorithm ask.

- While it waits, `/health` answers and `/ready` returns `503`.
- `/startup-status` lists each managed deployment with its state (`starting`,
  `loading`, `warming`, `ready`, ...), and `vllm-sr serve` prints the ones it
  still waits for.
- The wait, a first download included, is bounded by `VLLM_SRUN_READY_TIMEOUT`
  (10 minutes by default).
- If a model fails to load or is not ready in time, the Router does not start;
  its log and its last startup status (`phase: error`) name the deployment and
  the reason (see
  [Troubleshooting](./troubleshooting.md#the-runtime-reports-failed)).

:::note
Signals that an attached runtime answers as the decision model, or through a
binding that selects a head, also need model metadata during preparation, so
start those runtimes before the Router.
:::

Custom `systemone` questions can target an offline attached runtime without
blocking startup. They remain unknown until the current model card is ready and
supports the required task. Recovery does not bypass capability checks: a ready
incompatible card rejects the task. Bindings that select a head still require
their specific metadata; readiness does not silently switch a PII span detector
to a presence-only judgment.

A deployment declared but unused is never started and does not hold up
startup.

### On a reload

A configuration reload or `vllm-sr config apply` that adds a model waits the
same way while the previous configuration keeps serving:

- `/ready` stays `200`, and `GET /api/v1/config/hash` reports the activation as
  `pending`.
- The new configuration serves once its models are ready.
- If one fails to load, the reload is rejected and the previous configuration
  serves on.

### While serving

| Situation | What a feature sees |
| --- | --- |
| An attached runtime's decision model is still downloading or loading | Unknown |
| The answer arrives after the feature's timeout | Unknown |
| The runtime is overloaded | Unknown |
| The runtime process crashed | Unknown until the Router has restarted it (back-off from 1 s to 60 s) |
| Every model of a process failed to load | Unknown; the Router restarts that process with the same back-off until the model loads |
| The input is longer than the deployment allows and `overflow: reject` | An error for that input |

An unknown signal does not match. A decision's `rules.on_unknown` chooses what
an unknown signal means for that route (`no_match` by default, or
`fail_request`), and guard and PII modules have `on_error: allow | block`. The
`systemone` selection algorithm falls back to the first model in `modelRefs`.

## Check what is running

- The Router exports `vsr_model_runtime_ready{deployment="..."}` (1 when the
  deployment answers), `vsr_model_runtime_restarts_total`,
  `vsr_model_runtime_requests_total` by outcome and
  `vsr_model_runtime_unknown_answers_total` by reason on its metrics port.
- The Router's management API lists every deployment with its process, state,
  restarts and the model it serves (labels, device, engine arguments):
  `curl -s localhost:8080/api/v1/inventory/model-runtime`.
- An attached runtime answers `GET /v1/models` and `GET /health` directly.

The [metrics reference](../reference/metrics.md) lists every metric, and the
[configuration reference](../reference/configuration.md#router-managed-runtimes)
the environment variables that change the runtime command, the socket directory
and the model cache of managed runtimes.

---
title: Metrics and security
sidebar_label: Metrics and security
sidebar_position: 4
description: The metrics the Router frontend and the runtime export, how to read transport and server time, and the runtime's security properties.
---

# Metrics and security

The Router frontend exports metrics for every deployment it calls, and every
runtime exports its own on `GET /metrics`. This page lists both, then the
security properties of the runtime.

## Frontend metrics {#metrics}

Frontend (port 9190), available in both modes:

| Metric | Labels | Meaning |
| --- | --- | --- |
| `vsr_model_runtime_replica_ready` | `deployment`, `replica` | Physical worker readiness. Inventory additionally checks compatibility, dispatch backoff and ejection. |
| `vsr_model_runtime_replica_inflight` | `deployment`, `replica` | Locally admitted exchanges awaiting complete responses. |
| `vsr_model_runtime_replica_admitted_bytes` | `deployment`, `replica` | Bytes in those outstanding requests; not the worker's internal queue or token count. |
| `vsr_model_runtime_replica_requests_total` | `deployment`, `replica`, `outcome` | Completed dispatches: `ok`, `failed`, `rejected`, `canceled`, or `timeout`. A cancellation releases admission without marking the worker healthy or unhealthy; `timeout`s in a row eject it (`outlier_detection`). |
| `vsr_model_runtime_replica_ejections_total` | `deployment`, `replica` | Replicas taken out of dispatch after consecutive exchanges ran out of their deadline. |
| `vsr_model_runtime_replica_readmissions_total` | `deployment`, `replica`, `reason` | Ejected replicas returned to dispatch: `cooldown` when the cool-down ended, `answered` when an exchange succeeded. |
| `vsr_model_runtime_ready` | `deployment` | 1 while the deployment answers. |
| `vsr_model_runtime_requests_total` | `deployment`, `outcome` | Calls by outcome: `ok`, `timeout`, `cancelled`, `unknown_deployment`, `unavailable`, `overloaded`, `rejected`, `failed`. |
| `vsr_model_runtime_request_duration_seconds` | `deployment`, `surface` | Latency of calls that reached the runtime. |
| `vsr_model_runtime_transport_seconds` | `deployment`, `surface` | The part of each call's exchange outside the runtime: the router's time for the HTTP exchange minus the runtime's `Server-Timing` total. |
| `vsr_model_runtime_server_seconds` | `deployment`, `surface`, `phase` | The runtime's own time for each call's exchange, by `phase`: `parse`, `tokenize`, `queue`, `forward`, `post`, `serialize`, and `other` for the rest of its total. |
| `vsr_model_runtime_unknown_answers_total` | `deployment`, `reason` | Answers left unknown, by reason. |
| `vsr_model_runtime_restarts_total` | `deployment` | Restarts of managed processes. |
| `vsr_model_runtime_stage_decision_calls_total` | `deployment`, `round` | Decisions calls request stages sent: `first` is a stage's one call to the deployment, `later` counts any further call, whose questions did not travel with the others. |
| `vsr_model_runtime_bundle_question_wait_seconds` | | How long a stage's first question to a deployment waited for the stage's other questions to it. |
| `vsr_model_runtime_bundle_wait_seconds` | | How long a stage's first classify, embeddings or rerank call waited for the stage's others. |

A request stage sends every question it asks one deployment in one call: the
call goes once every signal that asks the deployment has asked, never on a
timer, so `later` stays at zero unless an admission limit lets fewer of a
stage's calls run at once than it asks. A stage's other calls go once none of
its signals can still add one, or 2 ms after the first of them.

The router times each HTTP exchange around its client, so its own encoding and
decoding count as transport. A bundled call records the exchange of its
bundle, so the transport and server metrics of a deployment's calls compare
with its `vsr_model_runtime_request_duration_seconds`. What a call's duration
holds beyond its exchange is the router's own share: waiting for the other
calls of its bundle and fusing them. The transport share of a deployment's
calls over five minutes:

```promql
sum by (deployment) (rate(vsr_model_runtime_transport_seconds_sum[5m]))
  / sum by (deployment) (rate(vsr_model_runtime_request_duration_seconds_sum[5m]))
```

For a runtime that sends no `Server-Timing` header (an older or a third-party
runtime), the router records no transport or server samples.

## Runtime metrics

Every runtime (`GET /metrics`) exports `vllm_srun_requests_total` by endpoint and
status, `vllm_srun_request_duration_seconds`,
`vllm_srun_queue_duration_seconds`,
`vllm_srun_forward_duration_seconds`, `vllm_srun_batch_rows`,
`vllm_srun_batch_tokens`, `vllm_srun_bundle_tasks`,
`vllm_srun_result_cache_total` by model and outcome,
`vllm_srun_queue_depth`, `vllm_srun_ready`, `vllm_srun_live` (0 while a
forward has run longer than `--forward-timeout-seconds`),
`vllm_srun_stalled_forwards_total` by model,
`vllm_srun_model_info` (by model, revision, family, engine, accelerator,
device, `dtype` and `numerics`) and `vllm_srun_model_memory_bytes` (the
bytes a model's loaded weights take, reduced-precision copies included).

## Security

- Code inside a model repository (`modeling_*.py` and similar) is never
  imported or run, and `trust_remote_code` is never used. Model code is part
  of the runtime or of plugins you install.
- Every file of a built-in model is checked against its recorded SHA-256
  before it is loaded. Links may point only inside the cache that holds the
  model.
- Tokens come from `HF_TOKEN` or the Hugging Face token file, never from
  command-line arguments, and are never logged.
- Request text is never logged, and metrics carry no request content.
- `vllm-srun serve` binds its worker API to `127.0.0.1` unless you pass
  `--host`. Workers have no authentication; expose them only on a trusted
  network. The frontend applies the API keys and model grants configured on
  each listener.

---
title: Reference
description: Command options, the models file, the HTTP API, environment variables, metrics and security details of the model runtime.
---

# Reference

This page lists the details the guides leave out. The design behind them is in
[`src/model-runtime/docs/design.md`](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/design.md).

## Commands

`vllm-sr serve MODEL ...` runs the runtime in your current Python environment.
Without a `MODEL` argument, `vllm-sr serve` starts the router instead.
`vllm-sr-runtime serve` is the same server with every option.

| Option | Default | Meaning |
| --- | --- | --- |
| `MODEL ...` | | Hub repositories, built-in model names or local package directories. Several models share one process. `MODEL@REVISION` pins a revision. |
| `--models FILE` | | A models file instead of `MODEL` arguments. |
| `--revision SHA` | | The 40-character commit to load, for one `MODEL`. |
| `--device` | `auto` | `auto`, `cpu`, `cuda[:N]`, `rocm[:N]`, `xpu[:N]` or `mps`. |
| `--host` | `127.0.0.1` | Address to listen on. |
| `--port` | `8100` | Port to listen on. |
| `--uds PATH` | | Listen on a Unix socket instead of TCP. |
| `--profile` | `exact` | `exact`, `shared_context`, `batching` or `max_speed`. |
| `--engine` | `native` | Engine plugin: `native` (PyTorch) or `onnxruntime`. |
| `--family` | detected | Force a model family plugin. |
| `--served-model-name` | the model name | The model ID the API reports, for one `MODEL`. |
| `--threads` | all cores | CPU threads for the process. |
| `--memory-budget GIB` | | Refuse to load a model estimated above this size. |
| `--max-queue` | `256` | Queued requests per model before the runtime answers 429. |
| `--max-queued-tokens` | `4194304` | Queued tokens per model before 429. |
| `--batch-window-ms` | `2.0` | How long the `batching` profile waits to combine requests. |
| `--max-batch-tokens` | `65536` | Tokens per forward pass. |
| `--max-request-bytes` | `8388608` | Largest request body. |
| `--max-bundle-tasks` | `64` | Most tasks in one `/v1/bundle` request. |
| `--result-cache-entries` | `16384` | Recent results each model keeps by content; `0` turns the cache off. |
| `--cache-dir` | `HF_HUB_CACHE` | Hugging Face cache directory. |
| `--offline` | off | Use only files already in the cache. |
| `--accept-licence ID` | | Accept a restricted licence (repeatable). |
| `--log-level` | `info` | `debug`, `info`, `warning` or `error`. |
| `--autotune-cache DIR` | | Keep GPU kernel tuning between runs. |

Other commands: `vllm-sr-runtime models` lists the built-in models with their
pinned revisions, and `vllm-sr-runtime plugins` lists the installed families,
engines, accelerators and profiles.

## Models file

A models file lists the models of one process, each with its own options:

```yaml title="models.yaml"
models:
  - model: vllm-sr/Vela-1.0-Encoder-307M-Domain
    name: vela-domain
    device: cpu
  - model: vllm-sr/Decision-2.0-Kai-0.6B
    name: decision-kai
    device: auto
    profile: exact
```

Each entry takes `model` (required), `revision`, `name`, `device`, `profile`,
`engine`, `family`, `memory_budget_gib` and `options` (family options). The
router writes this file for the processes it manages.

## HTTP API

Every request may name its `model`; it is required when a process serves
several models. The contract is the OpenAPI document served at
`GET /openapi.yaml`.

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/decisions` | Answer Choice, Noul, Score, Set and Span questions about a state. `POST /v1/systemone` is the same. |
| `POST /v1/classify` | Run a classification head over texts, text pairs or grounded answers: label probabilities, independent label scores or token spans. |
| `POST /v1/embeddings` | OpenAI-compatible embeddings of text, images and audio, with `dimensions` and `layer`. |
| `POST /v1/rerank` | Score documents against a query. |
| `POST /v1/bundle` | Several of the above, for one or more models, in one call. |
| `GET /v1/models` | What each model is, what it serves, its labels, limits, device and self-check status, and the process's `limits`: the most tasks one `/v1/bundle` may carry and the largest request body. With `/health` and `/health/live`, it reports the contract's `api_version`. |
| `GET /health` | 200 when every model is ready, 503 with each model's state otherwise. |
| `GET /health/live` | 200 while the process serves requests. |
| `GET /metrics` | Prometheus metrics. |

A response carries the answers and `usage`. Add
`"options": {"return_meta": true}` to a request to also get `meta`: the
revision, model digest, profile, engine, device and timings that answered it.

A request that cannot be served at all returns an HTTP error with
`{"error": {"code", "message"}}`: 400 `invalid_request`, 404
`model_not_found`, 413 `request_too_large`, 422 `unsupported_surface` (the
model does not serve that endpoint), 429 `overloaded`, 503 `not_ready`. A
failed item of a request (one input or one question) carries its own error
code and never fails the others.

### Classify

```json title="POST /v1/classify"
{
  "model": "vela-pii",
  "input": ["Hi, I'm Tom Baker (tom.baker@example.com)."],
  "options": {"overflow": "window", "window": {"tokens": 512, "overlap": 64}}
}
```

`input` is a string, a list of strings, a list of `{"text", "text_pair"}`
pairs, or a list of `{"context", "question", "answer"}` items for
hallucination checks. `options.overflow` is `reject`, `truncate` or `window`.
Span offsets count Unicode code points, end exclusive.

### Embeddings

```json title="POST /v1/embeddings"
{"model": "vela-embedding", "input": ["How do I reset my password?"], "dimensions": 256, "layer": 11}
```

`dimensions` and `layer` select a smaller vector or an earlier layer when the
model declares them. With `return_meta`, `meta.representation` identifies the
vector space, so you can keep vectors of different settings apart.

### Rerank

```json title="POST /v1/rerank"
{"model": "vela-reranker", "query": "How do I reset my password?",
 "documents": ["Open Settings, then Security.", "Our offices are closed on Sunday."], "top_n": 1}
```

### Bundle

```json title="POST /v1/bundle"
{
  "tasks": [
    {"id": "domain", "classify": {"model": "vela-domain", "input": ["What is 2 + 2?"]}},
    {"id": "kind", "decisions": {"model": "decision-kai", "state": "What is 2 + 2?",
      "questions": {"math": {"type": "noul", "instructions": "Is this a math question?"}}}}
  ],
  "options": {"deadline_ms": 80}
}
```

Results come back in task order, each with the status it would have had on
its own.

## Router configuration

| Field | Where | Meaning |
| --- | --- | --- |
| `provider: model_runtime` | deployment | The model runs in the model runtime. |
| `artifact`, `revision` | deployment | Hub repository and commit, or an absolute local path. |
| `device`, `profile` | deployment | See [Run it with the router](./deploy.md#describe-a-deployment). |
| `input.max_tokens`, `input.overflow` | deployment | Input limit of task models. |
| `process` | deployment | Managed deployments with the same name share a process. |
| `endpoint`, `served_name` | deployment | Attach to a runtime you run. |
| `deployment`, `contract`, `head` | binding | Which deployment a feature uses, the answer type it reads and an optional head name. |

The binding names (`domain_classifier`, `pii_classifier`, `prompt_guard`,
`fact_check_classifier`, `feedback_detector`, `modality_detector`,
`hallucination_detector`, `embedding`, `rag.reranker`, `safety.<rule>`,
`classifier.<rule>`) and their contracts are listed in the task guides.

## Router-managed runtimes

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `VLLM_SR_RUNTIME_COMMAND` | `vllm-sr-runtime` | Command the router runs for a managed process. |
| `VLLM_SR_RUNTIME_DIR` | a private temporary directory | Where the router puts the processes' Unix sockets and models files. |
| `VLLM_SR_RUNTIME_CACHE_DIR` | `/app/models/model-runtime` in router images | Hugging Face cache of managed runtimes. |
| `VLLM_SR_RUNTIME_CPU_PROCESSES` | one per CPU model, at most one per two cores | The most processes CPU models without a `process` are spread over. |
| `VLLM_SR_RUNTIME_PREPARED_DIR` | `/opt/router-model-artifacts` | Where the runtime finds prepared bundles (Vela Omni) before it looks on the Hub. |
| `HF_TOKEN` | | Token for gated or private repositories. |

The router talks to managed runtimes only over Unix sockets in a directory only
its own user can open (mode 0700). It restarts a process that exits, waiting 1
second at first and up to 60 seconds after repeated failures, and stops every
process (SIGTERM, then SIGKILL) when it exits. It also restarts a process in
which every model failed to load, the same way, so a model that failed for a
passing reason comes back without a configuration reload. A process that still
serves any model keeps running.

## Metrics

Router (port 9190):

| Metric | Labels | Meaning |
| --- | --- | --- |
| `vsr_model_runtime_ready` | `deployment` | 1 while the deployment answers. |
| `vsr_model_runtime_requests_total` | `deployment`, `outcome` | Calls by outcome: `ok`, `timeout`, `unavailable`, `overloaded`, `rejected`, `failed`. |
| `vsr_model_runtime_request_duration_seconds` | `deployment` | Latency of calls that reached the runtime. |
| `vsr_model_runtime_unknown_answers_total` | `deployment`, `reason` | Answers left unknown, by reason. |
| `vsr_model_runtime_restarts_total` | `deployment` | Restarts of managed processes. |

Runtime (`GET /metrics`): `vllm_sr_runtime_requests_total` by endpoint and
status, `vllm_sr_runtime_request_duration_seconds`,
`vllm_sr_runtime_queue_duration_seconds`,
`vllm_sr_runtime_forward_duration_seconds`, `vllm_sr_runtime_batch_rows`,
`vllm_sr_runtime_batch_tokens`, `vllm_sr_runtime_bundle_tasks`,
`vllm_sr_runtime_result_cache_total` by model and outcome,
`vllm_sr_runtime_queue_depth`, `vllm_sr_runtime_ready`,
`vllm_sr_runtime_model_info` and `vllm_sr_runtime_model_memory_bytes` (the
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
- `vllm-sr serve MODEL` listens on `127.0.0.1` unless you pass `--host`. The
  runtime has no authentication; expose it only on a trusted network.

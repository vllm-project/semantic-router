---
title: Reference
description: Command options, the models file, the HTTP API, environment variables, metrics and security details of the model runtime.
---

# Reference

This page lists the details the guides leave out. The design behind them is in
[`src/model-runtime/docs/design.md`](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/design.md).

## Commands

`vllm-sr serve MODEL ...` (engine mode) runs the runtime in the foreground, in
a Docker or Podman container from a router image. Without a `MODEL` argument,
`vllm-sr serve` starts the router instead.

| Option | Default | Meaning |
| --- | --- | --- |
| `MODEL ...` | | Hub repositories, built-in model names or local package directories, which the container reads through read-only mounts. Several models share one process. `MODEL@REVISION` pins a revision. |
| `--models FILE` | | A models file instead of `MODEL` arguments. Its local packages are mounted too. |
| `--revision SHA` | | The 40-character commit to load, for one `MODEL`. |
| `--platform` | `cpu` | The image and the GPU passthrough: `cpu` (`vllm-sr`), `amd` (`vllm-sr-rocm` with the ROCm devices) or `nvidia` (`vllm-sr-cuda` with the NVIDIA GPUs). macOS runs `cpu` only. |
| `--device` | `auto` | `auto`, or what the image runs: `cpu`, `rocm[:N]` with `--platform amd`, `cuda[:N]` with `--platform nvidia`, or a plugin's accelerator in an image that has the plugin. |
| `--host` | `127.0.0.1` | Host address the runtime's port is published on. |
| `--port` | `8100` | Host port the runtime is published on. |
| `--runtime-profile` | `exact` | `exact`, `shared_context`, `batching`, `max_speed`, or one a plugin adds. |
| `--image` | the platform's image | Another image, for example one with a plugin installed. |
| `--image-pull-policy` | `always` | `always`, `ifnotpresent` or `never`. |
| `--container-runtime` | detected | `docker` or `podman`. |
| `--log-level` | `info` | The runtime's log level. |

Engine mode keeps what the runtime downloads and compiles in
`~/.cache/vllm-sr/models` (`$XDG_CACHE_HOME/vllm-sr/models` when that is set);
`VLLM_SR_ENGINE_CACHE_DIR` moves it. `HF_TOKEN`, `HF_ENDPOINT` and
`HF_HUB_OFFLINE` reach the container by name, never on its command line.

`vllm-srun serve` is the same server with every option. It is the command the
images run; on your own machine it needs a source checkout
(`make model-runtime-install`).

| Option | Default | Meaning |
| --- | --- | --- |
| `MODEL ...` | | Hub repositories, built-in model names or local package directories. Several models share one process. `MODEL@REVISION` pins a revision. |
| `--models FILE` | | A models file instead of `MODEL` arguments. |
| `--revision SHA` | | The 40-character commit to load, for one `MODEL`. |
| `--device` | `auto` | `auto`, or an accelerator with an optional index: `cpu`, `cuda[:N]`, `rocm[:N]`, `xpu[:N]`, `mps`, or one a plugin adds. |
| `--host` | `127.0.0.1` | Address to listen on. |
| `--port` | `8100` | Port to listen on. |
| `--uds PATH` | | Listen on a Unix socket instead of TCP. |
| `--profile` | `exact` | `exact`, `shared_context`, `batching`, `max_speed`, or one a plugin adds. |
| `--engine` | `auto` | Engine plugin: `auto` (the first that runs the model, native first), `native` (PyTorch) or `onnxruntime` (the `onnx` extra). |
| `--family` | detected | Force a model family plugin. |
| `--served-model-name` | the model name | The model ID the API reports, for one `MODEL`. |
| `--threads` | all cores | CPU threads for the process. Each graph of an ONNX Runtime model runs its own pool of up to this many threads, so set it on a large host. |
| `--memory-budget GIB` | | Refuse to load a model estimated above this size. |
| `--max-queue` | `256` | Queued requests per model before the runtime answers 429. |
| `--max-queued-tokens` | `4194304` | Queued tokens per model before 429. |
| `--batch-window-ms` | `2.0` | How long the `batching` profile waits to combine requests. |
| `--max-batch-tokens` | `65536` | Tokens per forward pass. |
| `--max-request-bytes` | `8388608` | Largest request body. |
| `--max-bundle-tasks` | `64` | Most tasks in one `/v1/bundle` request. |
| `--load-attempts` | `5` | Loads of a model before it stays `failed`; a damaged package or a failed self-check is not retried. |
| `--load-retry-seconds` | `5` | Wait before the first reload of a model that failed to load, doubling up to 300 s. |
| `--result-cache-entries` | `16384` | Recent results each model keeps by content; `0` turns the cache off. |
| `--cache-dir` | `HF_HUB_CACHE` | Hugging Face cache directory. |
| `--offline` | off | Use only files already in the cache. |
| `--base-path DIR` | | A local copy of the pinned base model an adapter package needs, instead of a download. Its files are checked against the package's hashes. |
| `--accept-licence ID` | | Accept a restricted licence (repeatable). |
| `--log-level` | `info` | `debug`, `info`, `warning` or `error`. |
| `--autotune-cache DIR` | `$VLLM_SRUN_AUTOTUNE_CACHE` | Keep compiled GPU kernels, and the kernel tuning of models without recorded choices, between runs. Built-in models run their recorded choices on MI300X and MI325X instead of tuning. |

Other commands: `vllm-srun models` lists the built-in models with their
pinned revisions, `vllm-srun plugins` lists the installed families,
engines, accelerators and profiles, and `vllm-srun devices` prints, as
JSON, the devices this host offers and the one `--device auto` tries first.

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
| `GET /health` | Readiness: 200 when every model is ready, 503 otherwise. `status` is `starting`, `loading`, `warming`, `ready`, `degraded` or `failed`, and a process that serves several models lists each model's state. |
| `GET /health/live` | Liveness: 200 with `"status": "alive"` while the process serves requests, whether or not its models are ready. |
| `GET /metrics` | Prometheus metrics. |

A response carries the answers and `usage`. Add
`"options": {"return_meta": true}` to a request to also get `meta`: the
revision, model digest, profile, engine, device and timings that answered it.

The responses of `/v1/decisions`, `/v1/systemone`, `/v1/classify`,
`/v1/embeddings`, `/v1/rerank` and `/v1/bundle`, errors included, carry the
runtime's own time for the request in a `Server-Timing` header, in
milliseconds:

```text
Server-Timing: parse;dur=0.021, tokenize;dur=0.153, queue;dur=0.008, forward;dur=4.871, post;dur=0.034, serialize;dur=0.019, total;dur=5.141
```

| Phase | Time spent |
| --- | --- |
| `parse` | Reading and decoding the request body. |
| `tokenize` | Validating, rendering and tokenizing the request. |
| `queue` | Waiting for the model, before its first forward and between its forwards. |
| `forward` | Running its forwards, readout included. |
| `post` | Assembling the answers. |
| `serialize` | Encoding the response. |
| `total` | From the handler's start until the response is ready to send. What it holds beyond the phases is the server's own overhead. |

A bundle whose tasks go to several models reports the `queue`, `forward` and
`post` of the model that answered last. A client's own time for the call minus
`total` is the transport: its encoding and decoding, the connection and the
HTTP exchange. The router records it as a metric for every call.

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

### Decisions

A request names a `state` and its `questions`. A question's options are
`criteria`, or the ordered list the Router config uses for the same thing:

| Type | `criteria` | Or, as in the Router config |
| --- | --- | --- |
| `choice` | `{key: description}`, 2 to 255 options | `choices: [{key, description}]` |
| `noul` | `{false: description, true: description}`, both optional | `choices: [{key, description}]` with keys `false` and `true` |
| `score` | `[description, ...]`, 2 to 10 levels | `levels: [description, ...]` |
| `set`, `span` | `{label: description}`, 1 to 255 labels | `labels: [{key, description}]` |

A question takes one form or the other, and only its type's fields. Vela 2.0
answered these two requests on CPU; the numbers are rounded.

```json title="POST /v1/decisions"
{
  "model": "Vela-2.0-0.3B",
  "state": "Write a Python function that merges two sorted lists and explain its running time.",
  "questions": {
    "domain": {
      "type": "choice",
      "instructions": "Which domain does this request belong to?",
      "choices": [
        {"key": "code", "description": "Programming"},
        {"key": "math", "description": "Mathematics"},
        {"key": "other"}
      ]
    },
    "reasoning": {"type": "noul", "instructions": "Does answering this request need multi-step reasoning?"},
    "difficulty": {
      "type": "score",
      "instructions": "How difficult is this request?",
      "levels": ["Trivial", "Moderate", "Hard"]
    }
  }
}
```

```json title="Response"
{
  "model": "Vela-2.0-0.3B",
  "answers": {
    "domain": {
      "type": "choice",
      "choice": "code",
      "confidence": 0.99,
      "probabilities": {"code": 1.0, "math": 0.0, "other": 0.0},
      "abstain_probability": 0.01
    },
    "reasoning": {"type": "noul", "noul": 0.34},
    "difficulty": {
      "type": "score",
      "score": 1.3,
      "confidence": 0.28,
      "legend": {"0": "Trivial", "1": "Moderate", "2": "Hard"},
      "probabilities": {"0": 0.13, "1": 0.43, "2": 0.43}
    }
  }
}
```

A Set answers each label as a Noul under `<id>.<label>`, and lists the labels
at or above its threshold in `sets`. A Span returns the text it found in
`spans`, with Unicode code point offsets, end exclusive, and a Noul under its
ID. `thresholds` gives the threshold each question applied; set `threshold` on
the question to choose another.

```json title="POST /v1/decisions"
{
  "model": "Vela-2.0-0.3B",
  "state": "My card was charged twice and the parcel never arrived. Write to me at jane.doe@example.com.",
  "questions": {
    "problems": {
      "type": "set",
      "instructions": "Which problems does the customer report?",
      "labels": [
        {"key": "billing", "description": "A payment or charge problem"},
        {"key": "shipping", "description": "A delivery problem"},
        {"key": "account", "description": "A login or account problem"}
      ]
    },
    "contacts": {
      "type": "span",
      "instructions": "Find the personal contact details.",
      "labels": [{"key": "EMAIL_ADDRESS", "description": "An email address"}]
    }
  }
}
```

```json title="Response"
{
  "model": "Vela-2.0-0.3B",
  "answers": {
    "problems.billing": {"type": "noul", "noul": 0.95},
    "problems.shipping": {"type": "noul", "noul": 0.7},
    "problems.account": {"type": "noul", "noul": 0.03},
    "contacts": {"type": "noul", "noul": 1.0}
  },
  "spans": {
    "contacts": [
      {"label": "EMAIL_ADDRESS", "start": 71, "end": 91, "text": "jane.doe@example.com", "probability": 1.0}
    ]
  },
  "sets": {
    "problems": {
      "selected": ["billing", "shipping"],
      "probabilities": {"billing": 0.95, "shipping": 0.7, "account": 0.03}
    }
  },
  "thresholds": {"problems": 0.3, "contacts": 0.65}
}
```

An invalid question is answered `{"type", "error": "invalid_question",
"message"}`, where the message names the field, such as `set questions do not
take ['colour']`; the other questions are answered as usual. A request in which
no question is valid is answered 400 `invalid_request`, with each question's
reason in the message.

`src/model-runtime/tools/reference_examples.py --url <runtime>` sends this
section's requests to a runtime that serves Vela 2.0 0.3B and checks that the
answers match.

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
| `VLLM_SRUN_COMMAND` | `vllm-srun` | Command the router runs for a managed process. |
| `VLLM_SRUN_DIR` | a private temporary directory | Where the router puts the processes' Unix sockets and models files. |
| `VLLM_SRUN_CACHE_DIR` | `/app/models/model-runtime` in router images | Hugging Face cache of managed runtimes. |
| `VLLM_SRUN_CPU_PROCESSES` | one per CPU model, at most one per two cores | The most processes CPU models without a `process` are spread over, `device: auto` ones included on a host without a GPU. |
| `VLLM_SRUN_READY_TIMEOUT` | `10m` | How long the router waits for a deployment to become ready when it starts or reloads, as a duration such as `30m`. A first start may download and verify large models. |
| `VLLM_SRUN_RESULT_CACHE` | `4096` | Recent classify and decision results the router keeps per model, so a repeated request skips the runtime; `0` turns it off. |
| `VLLM_SRUN_AUTOTUNE_CACHE` | | The `--autotune-cache` directory of a runtime. |
| `HF_TOKEN` | | Token for gated or private repositories. |

The router talks to managed runtimes only over Unix sockets in a directory only
its own user can open (mode 0700). It restarts a process that exits, waiting 1
second at first and up to 60 seconds after repeated failures, and stops every
process (SIGTERM, then SIGKILL after 10 seconds, logged as
`runtime_process_killed`) when it exits. It also restarts a process in
which every model failed to load, the same way, so a model that failed for a
passing reason comes back without a configuration reload. A process that still
serves any model keeps running.

## Metrics

Router (port 9190):

| Metric | Labels | Meaning |
| --- | --- | --- |
| `vsr_model_runtime_ready` | `deployment` | 1 while the deployment answers. |
| `vsr_model_runtime_requests_total` | `deployment`, `outcome` | Calls by outcome: `ok`, `timeout`, `unavailable`, `overloaded`, `rejected`, `failed`. |
| `vsr_model_runtime_request_duration_seconds` | `deployment`, `surface` | Latency of calls that reached the runtime. |
| `vsr_model_runtime_transport_seconds` | `deployment`, `surface` | The part of each call's exchange outside the runtime: the router's time for the HTTP exchange minus the runtime's `Server-Timing` total. |
| `vsr_model_runtime_server_seconds` | `deployment`, `surface`, `phase` | The runtime's own time for each call's exchange, by `phase`: `parse`, `tokenize`, `queue`, `forward`, `post`, `serialize`, and `other` for the rest of its total. |
| `vsr_model_runtime_unknown_answers_total` | `deployment`, `reason` | Answers left unknown, by reason. |
| `vsr_model_runtime_restarts_total` | `deployment` | Restarts of managed processes. |

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

Runtime (`GET /metrics`): `vllm_srun_requests_total` by endpoint and
status, `vllm_srun_request_duration_seconds`,
`vllm_srun_queue_duration_seconds`,
`vllm_srun_forward_duration_seconds`, `vllm_srun_batch_rows`,
`vllm_srun_batch_tokens`, `vllm_srun_bundle_tasks`,
`vllm_srun_result_cache_total` by model and outcome,
`vllm_srun_queue_depth`, `vllm_srun_ready`,
`vllm_srun_model_info` and `vllm_srun_model_memory_bytes` (the
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
- `vllm-sr serve MODEL` publishes the runtime on `127.0.0.1` unless you pass
  `--host`. The runtime has no authentication; expose it only on a trusted
  network.

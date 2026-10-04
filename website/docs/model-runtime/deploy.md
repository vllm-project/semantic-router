---
title: Run it with the router
description: How the router runs models for you, how to put models on a GPU or in their own process, and how to attach to a runtime you run yourself.
---

# Run it with the router

The router can run its models in two ways:

- **Managed (the default).** The router starts the runtime as a child
  process, restarts it if it crashes, and stops it when the router stops.
- **Attached.** You run the runtime yourself, for example on a GPU machine or
  as a Kubernetes service, and the router connects to it with `endpoint`.

Both use the same configuration, the same models and the same answers.

## Built-in features need no configuration

When you turn on a built-in feature, such as a `domain` signal or the semantic
cache, the router already knows which Vela model it needs. It runs that model
in a managed runtime on the CPU. Set `use_cpu: false` on the feature's module
to let the runtime pick a GPU instead (`device: auto`).

Only models that an active feature uses are started. Declaring a deployment
that nothing uses does not load it.

## Describe a deployment

Write a deployment when you want to choose the model, the device or the
process yourself. A deployment is a named model under
`global.model_catalog.deployments`:

```yaml
global:
  model_catalog:
    deployments:
      vela-domain:
        provider: model_runtime
        artifact: vllm-sr/Vela-1.0-Encoder-307M-Domain
        device: cpu
        input:
          max_tokens: 512
          overflow: reject
      decision-kai:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Kai-0.6B
        device: auto
        process: decisions
```

| Field | Meaning |
| --- | --- |
| `provider` | Always `model_runtime` for models the runtime serves. |
| `artifact` | A Hugging Face repository, or an absolute path to a local copy. |
| `revision` | The exact 40-character commit to load. Built-in models are already pinned; other repositories need one. |
| `device` | `auto` (default), `cpu`, `cuda:N`, `rocm:N`, `xpu:N` or `mps`. |
| `profile` | `exact` (default) or an opt-in faster profile. See [Profiles](./profiles.md). |
| `input` | For task models: the longest input in tokens (`max_tokens`) and what to do with longer input (`overflow`: `reject`, `truncate` or `window`). |
| `process` | Runs deployments with the same name in one runtime process. |
| `endpoint` | Attach to a runtime you run instead of starting one. |
| `served_name` | The model's name on an attached runtime that serves several models (default: the deployment name). |

Then connect a feature to the deployment with a **binding**:

```yaml
global:
  model_catalog:
    bindings:
      domain_classifier:
        deployment: vela-domain
        contract: label_distribution.v1
```

Each feature reads one kind of answer, its `contract`: label probabilities
(`label_distribution.v1`), independent label scores (`label_scores.v1`), text
spans (`token_spans.v1`), vectors (`embedding.v1`) or relevance scores
(`relevance_scores.v1`). The task guides list the binding for each feature.
At startup the router checks every binding against the model's own
description (its heads, labels, dimensions and input limit) and refuses a
mismatch before it serves traffic.

A binding under `global.model_catalog.bindings` applies everywhere. A recipe
can override it under its own `routing.model_bindings`.

## Group models into processes

By default the models of one GPU share one runtime process, so all models on
`rocm:0` answer a request in one call and use memory efficiently. CPU models
are spread over several processes, one per model up to one per two cores the
router may use, so a request's models run in parallel; each process runs an
equal share of the cores as threads. `VLLM_SR_RUNTIME_CPU_PROCESSES` caps the
number of CPU processes, and `1` keeps every CPU model in one process.

Give a deployment its own `process` name when it should not share a fault
domain or memory with the others, for example a large decision model:

```yaml
global:
  model_catalog:
    deployments:
      decision-lux:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Lux-9B
        device: rocm:0
        process: large-decisions
```

If that process crashes or runs out of memory, only its deployments become
unavailable while the router restarts it; the other models keep answering.

## Attach to a runtime you run

Start a runtime anywhere the router can reach, then point a deployment at it:

```bash
vllm-sr serve vllm-sr/Decision-2.0-Lux-9B vllm-sr/Vela-1.0-Encoder-307M-PII --device rocm:0 --host 0.0.0.0 --port 8100
```

```yaml
global:
  model_catalog:
    deployments:
      gpu-lux:
        provider: model_runtime
        endpoint: http://gpu-runtime.internal:8100
        served_name: vllm-sr/Decision-2.0-Lux-9B
      gpu-pii:
        provider: model_runtime
        endpoint: http://gpu-runtime.internal:8100
        served_name: vllm-sr/Vela-1.0-Encoder-307M-PII
```

`endpoint` takes `http://host:port`, `https://host:port` or a Unix socket,
`unix:///path/to/runtime.sock`. The router does not start, restart or stop an
attached runtime; it checks its health and treats it as unavailable while it
is down.

A runtime started this way listens on `127.0.0.1` unless you pass `--host`.
Expose it only on a private network: it has no authentication of its own.

### On Kubernetes

The router image already contains the CPU runtime, so managed deployments work
in any cluster. The ROCm router image,
`ghcr.io/vllm-project/semantic-router/extproc-rocm`, contains the runtime with
PyTorch for ROCm: give the router pod an AMD GPU and set `device: rocm:0` on a
deployment, and the router runs that model on the GPU itself.

To share GPU models between several routers, run the runtime as its own
Deployment from the same image and attach every router to its Service. The
image's `vllm-sr-runtime` command starts it:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-runtime
spec:
  replicas: 1
  selector:
    matchLabels: {app: model-runtime}
  template:
    metadata:
      labels: {app: model-runtime}
    spec:
      containers:
        - name: runtime
          image: ghcr.io/vllm-project/semantic-router/extproc-rocm:latest
          command: ["vllm-sr-runtime"]
          args: ["serve", "vllm-sr/Decision-2.0-Lux-9B", "--device", "rocm:0", "--host", "0.0.0.0", "--port", "8100"]
          resources:
            limits: {amd.com/gpu: 1}
          ports:
            - {name: http, containerPort: 8100}
          readinessProbe:
            httpGet: {path: /health, port: http}
          livenessProbe:
            httpGet: {path: /health/live, port: http}
---
apiVersion: v1
kind: Service
metadata:
  name: model-runtime
spec:
  selector: {app: model-runtime}
  ports:
    - {name: http, port: 8100, targetPort: http}
```

Set the deployment's `endpoint` to `http://model-runtime.<namespace>.svc.cluster.local:8100`.
Use `/health` as the readiness probe: it succeeds only after every model has
loaded and passed its self-check.

## When a model is not ready

At startup the router waits until the task models its routes use have
loaded: domain, PII, guard, safety, fact-check, feedback and hallucination
models, and your own classifiers. If one of them fails to load, the router
does not start, and its log names the model and the reason (see
[Troubleshooting](model-runtime/troubleshooting.md#the-runtime-reports-failed)). This
holds for an attached runtime too, so start it before the router. Decision
models do not hold up the start: until one answers, its signals are unknown.

Once the router is serving, requests never wait for a model that cannot
answer:

| Situation | What a feature sees |
| --- | --- |
| The model is still downloading or loading | Unknown |
| The answer arrives after the feature's timeout | Unknown |
| The runtime is overloaded | Unknown |
| The runtime process crashed | Unknown until the router has restarted it (back-off from 1 s to 60 s) |
| The input is longer than the deployment allows and `overflow: reject` | An error for that input |

An unknown signal does not match. A decision's `rules.on_unknown` chooses what
an unknown signal means for that route (`no_match` by default, or
`fail_request`), and guard and PII modules have `on_error: allow | block`. The
`decision` selection algorithm falls back to the first model in `modelRefs`.

## Check what is running

- The router exports `vsr_model_runtime_ready{deployment="..."}` (1 when the
  deployment answers), `vsr_model_runtime_restarts_total`,
  `vsr_model_runtime_requests_total` by outcome and
  `vsr_model_runtime_unknown_answers_total` by reason on its metrics port.
- The router's management API lists every deployment with its process,
  state, restarts and the model it serves (labels, device, profile):
  `curl -s localhost:8080/api/v1/inventory/model-runtime`.
- An attached runtime answers `GET /v1/models` and `GET /health` directly.

The [reference](./reference.md#router-managed-runtimes) lists the environment
variables that change the runtime command, the socket directory and the model
cache of managed runtimes.

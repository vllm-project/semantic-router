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
| `device` | `auto` (default), `cpu`, `cuda:N`, `rocm:N`, `xpu:N`, `mps`, or an accelerator a plugin adds. |
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
`rocm:0` answer a request in one call and use memory efficiently. They take
turns on the GPU, one device call at a time; put a model on a GPU of its own
when it must not wait for the others. CPU models
are spread over several processes, one per model up to one per two cores the
router may use, so a request's models run in parallel; each process runs an
equal share of the cores as threads. `VLLM_SRUN_CPU_PROCESSES` caps the
number of CPU processes, and `1` keeps every CPU model in one process. Keep the
default where you can: in one process, a model on the optional ONNX Runtime
engine and a PyTorch model share the CPU's threads, and one of them answers
more slowly under load.

Deployments on `device: auto` (the default) are grouped by the device `auto`
picks on the router's host. On a host without a GPU that is the CPU, so each
of them gets a CPU process of its own, as with `device: cpu`. On a GPU host
they share the process of the first GPU, such as `rocm:0`, with the
deployments you put on that GPU; the runtime may still place a model on
another device when that GPU lacks the memory, and a model it places on the
CPU there may use every core the router has. `vllm-srun devices` shows
the device `auto` picks first. The router asks once, the first time it runs a
deployment on `auto`; if the runtime cannot answer, the `auto` deployments
share one process until the router restarts, and the router logs
`auto_device_unresolved`.

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
vllm-sr serve vllm-sr/Decision-2.0-Lux-9B vllm-sr/Vela-1.0-Encoder-307M-PII --platform amd --device rocm:0 --host 0.0.0.0 --port 8100
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

On a large host, give such a runtime `--threads`, or run it in a cpuset, when
it serves a model on the optional ONNX Runtime engine (an ONNX package, or an
Omni bundle): each graph of that model runs its own pool of up to `--threads`
CPU threads, and without the option up to every CPU the process may run on (an
Omni bundle has four graphs). The runtimes the router starts always get their
share of the cores as `--threads`.

### On Kubernetes

The router image already contains the CPU runtime, so managed deployments work
in any cluster. The ROCm router image,
`ghcr.io/vllm-project/semantic-router/vllm-sr-rocm`, contains the runtime with
PyTorch for ROCm, and `vllm-sr-cuda` the runtime with PyTorch for CUDA: give
the router pod a GPU (`amd.com/gpu` or `nvidia.com/gpu` in its resource
limits) and set `device: rocm:0` or `device: cuda:0` on a deployment, and the
router runs that model on the GPU itself. `vllm-sr serve --target kubernetes
--platform amd|nvidia` writes the image and the GPU limit into the chart's
values for you.

On first start the runtime downloads the models a router uses into its model
volume (`/app/models`, the chart's `persistence` claim, 10 GiB by default).
Size the claim for the models you route with: Vela Omni Mini alone takes
4.3 GB. In an air-gapped cluster, fetch them into it first
([Troubleshooting](model-runtime/troubleshooting.md#the-runtime-stays-in-loading-or-warming)).

Models you keep on the CPU in the ROCm image run on its ROCm build of PyTorch.
For most Vela task models that build fails the runtime's load-time check that
a model answers the same alone and in a batch, so under `exact` they answer
queued requests one at a time: the same answers, with less throughput under
load. A router whose models all run on the CPU should use the CPU image.
The ROCm image's PyTorch has no CPU LAPACK, so decoders with gated delta rule
layers (the Qwen3.5-based models, such as Decision 2.0 Lux-9B or Vela 2.0 4B)
can't load on its CPU: the runtime refuses `device: cpu` for them at startup,
before their weights load, and names the cause. Serve them on the GPU, or
from the CPU image.

To share GPU models between several routers, run the runtime as its own
Deployment from the same image and attach every router to its Service. The
image's `vllm-srun` command starts it:

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
          image: ghcr.io/vllm-project/semantic-router/vllm-sr-rocm:latest
          command: ["vllm-srun"]
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

The router serves a configuration only once the models it runs for it have
loaded. At startup it waits for every deployment it manages: the task models
its routes use (domain, PII, guard, safety, fact-check, feedback and
hallucination models, and your own classifiers) and the decision models that
`decision` signals and the `decision` selection algorithm ask. It also waits
for the task models of an attached runtime, so start that runtime before the
router. While it waits, `/health` answers and `/ready` returns `503`;
`/startup-status` lists each managed deployment with its state (`starting`,
`loading`, `warming`, `ready`, ...), and `vllm-sr serve` prints the ones it
still waits for. The wait, a first download included, is bounded by
`VLLM_SRUN_READY_TIMEOUT` (10 minutes by default). If a model fails to load or
is not ready in time, the router does not start; its log and its last startup
status (`phase: error`) name the deployment and the reason (see
[Troubleshooting](model-runtime/troubleshooting.md#the-runtime-reports-failed)).

The router does not wait for a decision model on an attached runtime, which has
a lifecycle of its own: until it answers, its signals are unknown. A deployment
that the configuration declares but that nothing uses is never started and does
not hold up the start.

A configuration reload or `vllm-sr config apply` that adds a model waits the
same way while the previous configuration keeps serving. `/ready` stays `200`,
`GET /api/v1/config/hash` reports the activation as `pending`, and the new
configuration serves once its models are ready. If one fails to load, the
reload is rejected and the previous configuration serves on.

Once the router is serving, requests never wait for a model that cannot
answer:

| Situation | What a feature sees |
| --- | --- |
| An attached runtime's decision model is still downloading or loading | Unknown |
| The answer arrives after the feature's timeout | Unknown |
| The runtime is overloaded | Unknown |
| The runtime process crashed | Unknown until the router has restarted it (back-off from 1 s to 60 s) |
| Every model of a process failed to load | Unknown; the router restarts that process with the same back-off until the model loads |
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

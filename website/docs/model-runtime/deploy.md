---
title: Frontend and runtime deployments
description: Serve decision models through a persistent frontend, place independent replicas, and attach externally managed runtimes.
---

# Frontend and runtime deployments

The frontend owns API listeners, authentication, configuration and model
resources. Router and Engine are two capabilities of the same frontend:

- **Engine** serves native System One questions without Chat backends or a
  user-authored routing configuration.
- **Router** additionally runs signals, decisions, algorithms and Chat routing.

```bash
vllm-sr serve vllm-sr/Vela-2.0-0.3B --engine
vllm-sr serve --config config.yaml
```

Startup sets `global.router.enabled`: every `serve` invocation without `-e`
starts Router mode; `-e` starts Engine mode and preserves saved routing without
initializing its consumers or Chat backends. Dashboard displays this startup
mode and manages model deployments independently. It does not switch modes.
The Dashboard and native API remain available in either mode. Listener keys
and native model grants still apply.

Model resources use either **managed** workers, which the frontend starts and
supervises, or **attached** workers, whose lifecycle belongs to another owner.
A logical deployment can have one or more compatible replicas. Task bindings
and public model names identify the logical deployment, independently of where
its workers run.

## Built-in features need no configuration

When you turn on a built-in feature, such as a `domain` signal or the semantic
cache, the router already knows which model it needs. Judgment tasks share the
default decision deployment; embeddings and other specialist tasks use their
configured models. To choose CPU or GPU placement, configure the deployment's
`device` or `replicas` as described below.

The default decision model and deployments used by enabled consumers or published
native model grants are loaded. Other declarations do not start workers.

## Describe a deployment

Write a deployment to choose the model, execution profile and worker placement. A deployment is a named model under
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
```

| Field | Meaning |
| --- | --- |
| `provider` | Always `model_runtime` for models the runtime serves. |
| `artifact` | A Hugging Face repository, or an absolute path to a local copy. |
| `revision` | The exact 40-character commit to load. Built-in models are already pinned; other repositories need one. |
| `device` | `auto` (default), `cpu`, `cuda:N`, `rocm:N`, `xpu:N`, `mps`, or an accelerator a plugin adds. |
| `profile` | `exact` (default) or an opt-in faster profile. See [Profiles](./profiles.md). |
| `input` | For task models: the longest input in tokens (`max_tokens`) and what to do with longer input (`overflow`: `reject`, `truncate` or `window`). |
| `replicas` | Explicit managed `device` or attached `endpoint`/`served_name` placements for this model. Omit for one worker. |
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

## Place and scale replicas

Use `-dp` to set the default deployment's independent worker count. Keep its
numerical profile separate from placement:

```bash
# Two workers on one explicitly selected host GPU.
vllm-sr serve vllm-sr/Vela-2.0-4B -e --platform rocm -dp 2 --device-ids 0
# Two workers across two host GPUs.
vllm-sr serve vllm-sr/Vela-2.0-4B -e --platform rocm -dp 2 --device-ids 0,1
```

Without device IDs, a new GPU placement uses the first N available GPUs and
refuses insufficient capacity; it does not silently place every worker on
one card. Existing authored placements resize in their declared order.
Specifying only MODEL preserves configured replicas and profile. These flags
write the same canonical resource shown below; there is no separate DP state.
For Kubernetes, use allocation ordinals in config instead of host IDs.

Each managed replica has its own process. All replicas of a deployment use the
same artifact, revision, profile and capabilities. A request's fused question
batch is dispatched as one unit after the logical model is selected. The pool
uses its own in-flight work observations to select a ready worker; those
observations are not the attached runtime's internal queue length.

```yaml
global:
  router:
    enabled: false
  model_catalog:
    deployments:
      primary:
        provider: model_runtime
        artifact: vllm-sr/Vela-2.0-4B
        profile: exact
        replicas:
          - device: rocm:0
          - device: rocm:1
    system:
      decision_model:
        deployment: primary
```

Use different GPUs to add compute capacity. Repeating a device explicitly starts
multiple workers on that GPU; each worker needs memory for its model. Measure
throughput and tail latency with your input lengths and concurrency before
choosing this layout. `batching` can improve cross-request utilization within
a worker; `max_speed` also changes numerics and requires a separate quality
assessment. Replicas do not shard weights and are not tensor or pipeline
parallelism.

Do not combine `replicas` with deployment-level `device`, `endpoint` or
`served_name`. Omitting `replicas` keeps the one-worker shorthand with the same dispatch and
status contract as an explicit single replica. Single-worker deployments retain
the frontend result cache; multiple replicas use independent worker caches. A replica's
placement is independent of bindings, so changes to routing consumers retain
unchanged worker identities. Reconfiguration prepares a compatible pool before
publishing it, and retired workers drain existing requests.

The pool admits a bounded number of outstanding requests per replica. It
returns overload when all ready replicas are full. Inventory reports desired
and ready replica counts, per-worker readiness, in-flight requests and
restarts. A degraded pool can keep serving through its healthy replicas.

CPU workers receive a stable thread budget based on host cores and
`VLLM_SRUN_CPU_PROCESSES`. Mode changes do not recalculate that budget from the
number of active consumers.

## Publish a native API

With the `primary` deployment above, add an explicit grant to a standalone
listener so clients can ask it questions directly:

```yaml
listeners:
  - name: http
    address: 0.0.0.0
    port: 8899
    systemone:
      models: [vllm-sr/Vela-2.0-4B]
```

Use `GET /v1/systemone/models` to discover published native models, then send
`POST /v1/systemone` or `POST /v1/decisions` with that `model` ID. A deployment
uses its Hub artifact ID unless you set `public_name`; local artifacts need an
explicit public name. Add listener `api_keys` when clients must authenticate.

The native grant is separate from the Chat `listeners[].models` allowlist and
works in either startup mode. An existing grant stays unchanged when a model
is replaced or scaled; update it when you intend to publish another model.
See the [quickstart](model-runtime/quickstart.md#3-send-a-request) for a complete native
request. Publication does not guarantee readiness; inspect it with
`vllm-sr instance --config config.yaml models`.

## Attach to a runtime you run

Start a runtime anywhere the router can reach, then point a deployment at it:

```bash
vllm-srun serve vllm-sr/Decision-2.0-Lux-9B --device rocm:0 --host 0.0.0.0 --port 8100
```

```yaml
global:
  model_catalog:
    deployments:
      gpu-lux:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Lux-9B
        endpoint: http://gpu-runtime.internal:8100
        served_name: vllm-sr/Decision-2.0-Lux-9B
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
--platform rocm|cuda` writes the image and the GPU limit into the chart's
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

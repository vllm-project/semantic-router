# Built-in Model Runtime

## Overview

The model runtime serves decision models for the Router. It is a separate
process with one HTTP contract: `POST /v1/decisions` (a strict superset of
System One, also available as `POST /v1/systemone`), `GET /v1/models`,
`GET /health` and `GET /metrics`. The Router talks to it over a Unix socket
when it manages the runtime, or over HTTP when it attaches to an engine
started elsewhere.

Phase 1 serves the six Decision 2.0 models out of the box:

| Model | Revision | Backbone |
| --- | --- | --- |
| `vllm-sr/Decision-2.0-Kai-0.6B` | `881bee41` | Qwen3 dense |
| `vllm-sr/Decision-2.0-Eos-0.8B` | `ad0aa724` | Qwen3.5 Gated DeltaNet hybrid |
| `vllm-sr/Decision-2.0-Sol-2B` | `4b75b521` | Qwen3.5 Gated DeltaNet hybrid |
| `vllm-sr/Decision-2.0-Nox-4B` | `ce1bdc9d` | Qwen3.5 Gated DeltaNet hybrid |
| `vllm-sr/Decision-2.0-Lux-9B` | `214ffa43` | Qwen3.5 Gated DeltaNet hybrid |
| `vllm-sr/Decision-2.0-Vega-27B` | `9b067a95` | Qwen3.8-27B with a LoRA adapter |

Every package is pinned by revision and verified file by file against its
manifest before load. Model code is built into the runtime; code shipped
inside a model package is never executed.

## What Problem Does It Solve?

Decision models are decoders with custom readouts, so they do not fit the
Router's in-process encoder bindings. The runtime gives them one supervised,
pluggable home: model families, execution engines and hardware accelerators
are plugins, and faster numerics are opt-in profiles measured against the
default.

## When to Use

Use it whenever a `decision` signal or the `decision` selection algorithm is
configured. Run it on its own with `vllm-sr serve <hf-model>` to serve a
decision model to other clients or to share one engine between Routers.

## Configuration

```yaml
global:
  model_catalog:
    deployments:
      decision-kai:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Kai-0.6B      # Hub repository or an absolute package path
        revision: 881bee413681d80ebeac86afcda8b4138dae516e
        device: auto                                  # auto, cpu, cuda[:N] or rocm[:N]
        profile: exact                                # exact, shared_context, batching or max_speed
      decision-shared:
        provider: model_runtime
        endpoint: http://decision-runtime:8100        # attach; unix:///path also works
```

Without `endpoint`, the Router starts `vllm-sr-runtime serve` for each
deployment that a signal or selector uses, on a private Unix socket, restarts
it with back-off when it exits, and stops it on shutdown. Until the runtime
reports ready, decision signals are unknown and decision selectors fall back.
`VLLM_SR_RUNTIME_COMMAND` overrides the command, `VLLM_SR_RUNTIME_CACHE_DIR`
sets the model cache and `VLLM_SR_RUNTIME_DIR` the socket directory.

The `vllm-sr` image ships the runtime with CPU PyTorch, so managed CPU
deployments work out of the box, and downloaded models persist under
`/app/models/model-runtime`. For a GPU, start the runtime with a GPU build of
PyTorch (`vllm-sr serve <model> --device rocm:0`, or the runtime image built
from `src/model-runtime/Dockerfile` with a GPU wheel) and attach to it with
`endpoint`.

Run an engine on its own:

```bash
pip install -e src/model-runtime
vllm-sr serve vllm-sr/Decision-2.0-Kai-0.6B --device cpu --port 8100
vllm-sr serve vllm-sr/Decision-2.0-Lux-9B --device rocm:0 --profile shared_context
```

| Profile | Numerics | Purpose |
| --- | --- | --- |
| `exact` | identical to the released packages | the default |
| `shared_context` | approximate | run the shared state of a multi-question request once |
| `batching` | approximate | share forwards between concurrent requests |
| `max_speed` | approximate | faster kernels that may change rounding |

CPU and ROCm (MI300X, MI325X) are validated. CUDA is implemented and unit
tested but not yet validated on NVIDIA hardware. The runtime exposes
Prometheus metrics at `/metrics`, and the Router exports
`vsr_model_runtime_*` metrics per deployment.

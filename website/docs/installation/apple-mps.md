---
title: Apple GPU on macOS
description: Run supported Router model processes natively with MPS while keeping the Router stack in Docker.
---

# Apple GPU on macOS

The Apple host bridge is experimental. MPS is implemented by the model runtime,
but model answers and performance still need qualification on Apple silicon.
Do not treat a running process or a successful request as a recorded accuracy
result. Models without MPS golden answers report `golden.status: unverified`.

## Start and stop

Use an Apple silicon Mac, an arm64 installation of the CLI, and a running local
Docker engine. Linux, Intel Macs, Rosetta Python, Kubernetes, Podman, and remote
Docker contexts are rejected. The Docker engine must allow its containers to
reach host loopback services through `host.docker.internal`.

```bash
vllm-sr serve --config config.yaml --platform apple
vllm-sr status model-runtime
vllm-sr logs model-runtime -f
vllm-sr stop
```

The Router remains in its container in both instance modes; the existing
Dashboard/minimal and gateway controls are preserved. The CLI manages a native
host supervisor. The Router sends it the
model groups needed by active recipes, and reaches the existing model HTTP/JSON
API through an authenticated TCP proxy. Reloaded generations share unchanged
groups; their last consumer releases the host model process. Each group lifetime
has its own lease identity, so a retiring group's release cannot stop a process
owned by a concurrently acquired group. Abandoned leases expire independently.
Declaring an unused model does not load it. Explicit external runtime endpoints
remain external. Only managed `device: auto` placements move to MPS. Authored
`device: cpu` placements continue in the container on private Unix sockets;
mixed CPU/auto replica pools retain those placements. Apple does not support
`-dp` or `--device-ids`; author placement in the configuration instead.

MODEL overrides the judgment deployment; it does not select a separate runtime
server. `--engine` keeps the Router frontend and disables recipe routing, exactly
as on the other platforms:

```bash
vllm-sr serve vllm-sr/Vela-2.0-0.3B --platform apple --engine --dtype float32
vllm-sr stop
```

The deployment's authored device still applies: choose `device: auto` to place
it on MPS. There is no bare runtime listener on port 8000, no `/engine` control
route, and no Apple-specific `--models`, `--device`, `--host`, `--port` or `--uds`
serve flags. Host workers listen on Unix sockets inside private, short-path
directories; only the authenticated supervisor proxy is exposed over TCP.

Local packages must live in the stack's mounted models directory. An artifact
such as `/app/models/my-model` is mapped to that mount's host source. Other
image paths and paths escaping the mount (including through symlinks) are
rejected. This mapping is identical in Router and Engine modes.
Use separate `VLLM_SR_STACK_NAME` values for independent Router or engine stacks.

## Installation and cache

The first run copies the pure Python runtime and plugin metadata from the
selected Router image. Its immutable image ID identifies the environment cache.
The CLI provisions a native Python interpreter and installs macOS dependency
wheels matching the runtime image's direct dependency versions. It does not
copy Linux torch libraries or install into the user's Python. No separate
runtime installation is required, but first use needs network access for the
managed interpreter, dependencies, and uncached models.
The image records its enabled runtime extras, including `multimodal` in the
default build. Markers are evaluated for the managed macOS arm64 interpreter,
not Linux. Rebuild older images lacking this extras manifest; do not reuse a
partial native environment. A missing macOS wheel fails installation explicitly.

State, native environments, models, and bounded logs live under
`~/Library/Caches/vllm-sr/apple`. Stop retains these downloads and logs. Changing
the image selects a separate environment. A warm environment can be reused,
but image pull policy and uncached model downloads may still require networking.
Host downloads follow host proxy/certificate settings, which can differ from
Docker's settings.

For development, build the Router image containing this bridge and select it
explicitly using the maintained local-image workflow:

```bash
vllm-sr serve --config config.yaml --platform apple \
  --router-image YOUR_LOCAL_ROUTER_IMAGE --image-pull-policy never
```

The Router image must contain both this host-attachment implementation and the
matching runtime. Older released images do not acquire this feature merely
because a newer CLI selects `--platform apple`.

## Limits and troubleshooting

MPS currently advertises FP32 reference kernels, without bf16 autocast, Triton,
or GPU graphs. CPU fallback is disabled for managed model workers. Unsupported
operators or model families fail rather than silently becoming CPU inference.
Prepared image-only assets and arbitrary accelerator plugins are not qualified
for this bridge. Accelerating Router models does not relocate configured
answer-generating backend LLMs onto the Mac.

The runtime currently advertises `bf16=False` even on Macs where PyTorch BF16
operations work. BF16 detection and qualification are separate runtime work;
this bridge does not change that capability or silently change model precision.

- Vela 1.0 encoder-family FP32 models have an MPS smoke result reported by a
  reviewer, not complete per-model golden qualification. FP32 encoder members
  can pass placement without BF16; operator support and goldens still apply.
- BF16-published decoder models, including Decision 2.0 Kai, cannot use MPS at
  their default precision today. The shipped config binds Kai, so it needs an
  explicit configuration change before it can start with Apple placement.
- Set `dtype: float32` on each affected deployment (or use `--dtype float32`
  for the judgment deployment), or keep it on `device: cpu`. FP32 removes the
  BF16 placement requirement, not other model/operator restrictions. Its
  answers must be qualified separately. Unbound declarations are not loaded.

For example, in an existing canonical configuration, the Kai deployment can
explicitly select host FP32:

```yaml
global:
  model_catalog:
    deployments:
      decision-kai:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Kai-0.6B
        device: auto
        dtype: float32
```

The host verifies the bound package and its actual runtime placement before
starting its worker. This may download and verify uncached files, but does not
load an engine or allocate model weights on the GPU. A rejected placement names
the deployment and the remedy rather than entering worker load retries.

The host GPU shares system memory with macOS, applications, and the Docker VM.
FP32 weights alone are roughly 16 GB for a 4B model and 36 GB for a 9B model,
before activations and temporary allocations. Moving out of Docker removes the
VM model-memory limit; it does not remove the Mac's total memory limit.
Explicit `device: cpu` deployments stay inside Docker and remain subject to
the VM's memory allocation, even when other deployments use the host GPU.

`status all` also reports the host supervisor when one is recorded. A supervisor
running does not mean every model is ready; model health remains in the runtime
and Router inventory. `stop` attempts host cleanup even when Docker is down,
and removes only a process whose recorded identity still matches. If a Router
crashes, its unreleased host model groups expire after 45 seconds. The supervisor
itself remains available until stop. Logs remain available afterward.

If the bridge cannot connect, check the Docker context, host firewall/VPN,
`host.docker.internal`, and runtime logs. If native dependency installation fails,
check wheel availability for the selected image's versions and host networking.
For model load failures, check MPS operator support, model access, and memory.

## Qualification

MPS answers must be recorded for each pinned model revision on real Apple
silicon, with its dependency/hardware record and a measured error bound.
`src/model-runtime/tools/golden_answers.py --device mps --record FILE` requires
an explicit `--tolerance`; readiness uses that per-model tolerance rather than
the generic GPU tolerance. Missing MPS records remain unverified. Do not invent
records, mark the accelerator validated, or change engine arguments to hide
differences. Lifecycle/HTTP fixtures run in ordinary CI; GPU acceptance requires
an Apple-silicon runner.

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

The Router and Dashboard remain in containers; the existing gateway stack is
preserved. The CLI manages a native host supervisor. The Router sends it the
model groups needed by active recipes, and reaches the existing model HTTP/JSON
API through an authenticated TCP proxy. Reloaded generations share unchanged
groups; their last consumer releases the host model process. Declaring an unused
model does not load it. Explicit external runtime endpoints remain external.

Engine mode serves models directly on host loopback, without starting a Router:

```bash
vllm-sr serve vllm-sr/Decision-1.0-Kai-0.6B --platform apple --port 8000
curl http://127.0.0.1:8000/health
vllm-sr stop
```

Loading continues in the background in engine mode. Check `/health` before
calling the model. Engine mode supports MODEL names, pinned `MODEL@REVISION`,
and `--models` files; it uses MPS and accepts loopback TCP rather than `--uds`.
Local host model directories are supported in engine mode. Router-mode models
must name Hub artifacts; paths inside images are not host model paths.
Use separate `VLLM_SR_STACK_NAME` values for independent Router or engine stacks.

## Installation and cache

The first run copies the pure Python runtime and plugin metadata from the
selected Router image. Its immutable image ID identifies the environment cache.
The CLI provisions a native Python interpreter and installs macOS dependency
wheels matching the runtime image's direct dependency versions. It does not
copy Linux torch libraries or install into the user's Python. No separate
runtime installation is required, but first use needs network access for the
managed interpreter, dependencies, and uncached models.

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

The host GPU shares system memory with macOS, applications, and the Docker VM.
FP32 weights alone are roughly 16 GB for a 4B model and 36 GB for a 9B model,
before activations and temporary allocations. Moving out of Docker removes the
VM model-memory limit; it does not remove the Mac's total memory limit.

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
records, mark the accelerator validated, or change the `exact` profile to hide
differences. Lifecycle/HTTP fixtures run in ordinary CI; GPU acceptance requires
an Apple-silicon runner.

# Router images: one stage graph with the model runtime against the binding images

Phase 4 deletes the in-process bindings (candle, ONNX Runtime, OpenVINO and the
ML and NLP crates) and builds every router image from one Dockerfile. This
record compares the five router images before and after: image size, build
time, router startup and time-to-ready. Both generations were built with the
same builder and started on the same cores.

- **Date:** 2026-10-04.
- **Machine:** a 160-vCPU AMD EPYC 9575F virtual machine (2 × 64 cores), used
  CPU only. Every build and container ran on the same 32 pinned vCPUs. Other
  workstreams' pinned jobs ran on other cores, and the 1-minute load stayed
  between 4 and 51.
- **Commits:**
  - Before: `1c6d372ec`, the last tree with the binding images.
    `tools/docker/Dockerfile.extproc` and `tools/docker/Dockerfile.extproc-rocm`
    built `extproc` and `extproc-rocm`. `src/vllm-sr/Dockerfile`, `.rocm` and
    `.cuda` built `vllm-sr`, `vllm-sr-rocm` and `vllm-sr-cuda`.
  - After: `30e6c4ca3` (IP2b). One file, `tools/docker/Dockerfile.extproc`,
    builds all five. `--build-arg ACCELERATOR=cpu|rocm|cuda` selects the
    PyTorch build, and `--target extproc|vllm-sr` selects the image. Both
    targets share the Go router build, the Vela Omni bundle, the image-routing
    assets and the runtime stage.
- **Builds:** a `docker-container` buildx builder, with its BuildKit daemon
  pinned to the 32 vCPUs and on the host network
  (`--driver-opt cpuset-cpus=<32 vCPUs>,network=host`). Each image was built
  once, one after another, with `--no-cache --load`. Build time is the wall
  time of `docker buildx build`, including base-image pulls, wheel downloads
  and the load into the Docker daemon.
- **Sizes:** from `docker save`. "Unpacked" is the sum of the uncompressed
  layers. "Pushed" is the sum of the gzip layer blobs that a registry stores
  and a node pulls. Layers that two images share are counted in each image.
- **Startup:** each CPU image was started with `docker run` (host network) on
  the configuration [`removal-footprint.yaml`](removal-footprint.yaml). It has
  the domain, jailbreak and PII classifiers, an embedding signal, and decisions
  that read all of them.
  - The client polls the Router API every 0.1 s. It records the first 200 from
    `/health` (API up) and from `/ready` (every model-backed signal serves, i.e.
    time-to-ready), and the container's memory at ready from `docker stats`.
  - Two resource shapes: the Helm chart's limits of 2 CPUs and 7 GiB
    (`--cpus 2 --memory 7g`), and all 32 vCPUs.
  - Run 0 of the 2-CPU shape starts with an empty model volume, so it downloads
    the models. Runs 1–5, and every run of the 32-vCPU shape, start with the
    models in place.
  - The GPU images were not started: they share the router and runtime layers
    with the CPU images and differ only in the PyTorch wheel.
- **Raw data:** [`removal-footprint.json`](removal-footprint.json).

## Images

| Image | Build before → after (s) | Unpacked before → after (GB) | Pushed gzip before → after (GB) | Layers |
| --- | --- | --- | --- | --- |
| `extproc` | 177 → 90 (−49 %) | 2.29 → 2.28 (−1 %) | 1.51 → 1.27 (−16 %) | 12 → 14 |
| `vllm-sr` | 106 → 87 (−18 %) | 2.29 → 2.28 (0 %) | 1.27 → 1.27 (0 %) | 20 → 17 |
| `extproc-rocm` | 274 → 341 (+25 %) | 17.40 → 15.57 (−11 %) | 5.57 → 7.24 (+30 %) | 17 → 16 |
| `vllm-sr-rocm` | 311 → 349 (+12 %) | 17.25 → 15.58 (−10 %) | 5.44 → 7.24 (+33 %) | 24 → 19 |
| `vllm-sr-cuda` | 130 → 207 (+60 %) | 5.12 → 8.73 (+70 %) | 3.43 → 5.23 (+53 %) | 28 → 18 |

- **CPU images:** `extproc` now carries the model runtime with CPU PyTorch
  (0.9 GB unpacked), and still builds in half the time and pushes 16 % less.
  The binding image compiled the Rust crates and carried a router build per
  binding (0.7 GB) plus their native libraries. `vllm-sr` already shipped the
  model runtime with CPU PyTorch next to the bindings, so it keeps its size and
  builds 18 % faster.
- **GPU images:** in each GPU image, one layer is most of the image: the
  official PyTorch wheel with the GPU libraries it bundles.
  - ROCm: `torch` 2.12.0+rocm7.1 with `fla-core`, 14.2 GB unpacked and 6.2 GB
    pushed.
  - CUDA: `torch` 2.10.0+cu128, 7.3 GB unpacked and 4.2 GB pushed.
  - The binding images ran candle and ONNX Runtime on the GPU, with no GPU
    PyTorch. The CUDA image used the CUDA 12.4 and cuDNN libraries of its base
    image (3.1 GB unpacked). The ROCm images used ONNX Runtime ROCm with
    MIGraphX and a Composable Kernel attention build. Those compress better and,
    for CUDA, are smaller.
  - The GPU images are therefore 30–53 % larger to pull and up to 60 % slower
    to build (the wheel download and install dominate). The ROCm images are
    10–11 % smaller unpacked.
  - Trimming the wheels' bundled libraries to the GPU architectures the images
    support is the open lever. It needs a GPU parity run per architecture, so
    it is not part of this phase.

## Startup and time-to-ready

| Image | Resources | Cold ready before → after (s) | Warm ready before → after (s) | Memory at ready before → after (GiB) |
| --- | --- | --- | --- | --- |
| `extproc` | 2 CPUs, 7 GiB | 37.3 → 22.8 | 16.9 (15.8–17.4) → 9.2 (9.2–9.8) (−45 %) | 6.49 → 3.01 |
| `extproc` | 32 vCPUs | — | 11.3 (11.2–11.5) → 4.8 (4.8–4.9) (−57 %) | 6.48 → 3.88 |
| `vllm-sr` | 2 CPUs, 7 GiB | 37.6 → 22.9 | 17.1 (15.5–18.1) → 9.6 (9.4–10.4) (−44 %) | 6.51 → 3.02 |
| `vllm-sr` | 32 vCPUs | — | 11.1 (10.9–11.4) → 5.0 (5.0–5.1) (−55 %) | 6.51 → 3.88 |

- **The model runtime makes the router ready sooner and with half the
  memory.** On the chart's 2 CPUs, a warm start is ready 44–45 % sooner, and
  the first start with an empty model volume 39 % sooner. On 32 vCPUs, a warm
  start is ready 55–57 % sooner. Memory at ready falls from 6.5 GiB to 3.0 GiB
  on 2 CPUs and to 3.9 GiB on 32 vCPUs.
- **The 32-vCPU shape uses more memory than the 2-CPU one** because of the CPU
  process plan. On 2 CPUs the router runs the four models in one runtime
  process with two threads. On 32 vCPUs it runs four processes of eight threads,
  and each process holds its own interpreter and PyTorch.
- **The Router API is up 0.22–0.29 s after `docker run`** in every run of both
  generations. The API starts before the models load, and `/ready` reports when
  they serve.
- **Cold starts depend on the network.** Run 0 downloads the pinned models from
  Hugging Face. Before, the router downloaded its artifacts itself; now the
  runtime downloads each model it serves.
- **The two images start alike.** `vllm-sr` and `extproc` run the same router
  and runtime; `vllm-sr` adds the CLI's assets and its start script.

## Without ONNX Runtime and the Omni bundle (#4619)

Vela Omni runs on the native engine from its published files, so the images
no longer install ONNX Runtime (now the optional `onnx` extra) or carry a
prepared Omni bundle.

- **Date:** 2026-10-06.
- **Commits:** before, staging `91d369ff2`; after, `1ff74ff22`.
- **Builds:** node D's Docker daemon (BuildKit, its layer cache),
  `tools/docker/Dockerfile.extproc` with `--build-arg ACCELERATOR=cpu|rocm`
  and `--target extproc|vllm-sr`. The staging ROCm images are the integration
  lead's verification builds from the same daemon and file.
- **Sizes:** the sum of each image's layers, uncompressed (`docker history`),
  the "Unpacked" column above.

| Image | Before (GB) | After (GB) | Change |
| --- | --- | --- | --- |
| `extproc` | 2.341 | 1.607 | −0.734 (−31 %) |
| `vllm-sr` | 2.352 | 1.617 | −0.735 (−31 %) |
| `extproc-rocm` | 14.386 | 13.656 | −0.730 (−5 %) |
| `vllm-sr-rocm` | 14.397 | 13.667 | −0.730 (−5 %) |

- **What left:** the Omni Nano bundle's layer (`/opt/router-model-artifacts`,
  662 MB) and ONNX Runtime 1.30 with its dependencies (the runtime's
  third-party layer, 232 → 159 MB on the CPU). Every other layer is unchanged.
- **The E2E candidate image** (`extproc` in CI) was built with both Omni
  variants, so it also loses Mini's 5.1 GB bundle.
- **Startup:** a router that routes images now downloads Omni Nano (0.67 GB)
  into its model volume on its first start, as it downloads every other model;
  later starts load it from there.

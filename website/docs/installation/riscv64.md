---
title: RISC-V64
description: Build and run the experimental standalone Router on native Linux RISC-V64.
---

# RISC-V64 standalone Router

Linux RISC-V64 is an **experimental** platform for the standalone Router.
The Router runs natively and forwards requests directly to model endpoints;
Envoy is not part of this deployment. Models used for classification,
embeddings or reranking, when configured, run in an external model runtime.
LLM generation uses a separate backend endpoint.

## Coverage

Native validation uses a SpacemiT X60 machine with 8 CPU cores, 16 GB RAM,
openKylin 2.0 SP1 and Linux 6.6.63-cloud. The build uses Debian-compatible
containers and public Go/Rust toolchains; no QEMU is required. Other machines
need to qualify their CPU instruction support, kernel, container runtime and
configuration independently.

| Component or capability | Coverage |
| --- | --- |
| Go Router process | Native Linux/RISC-V64 build and standalone container execution |
| GLIDE native library | Compiled from the pinned source and Cargo lockfile; not a stub |
| Rule routing and HTTP forwarding | Keyword/default routing, buffered responses, SSE and restart smoke |
| Router-side learned models | External endpoints; local model inference is not qualified here |
| Envoy / ExtProc deployment | Outside this RISC-V qualification |
| Dashboard, CLI-managed stack and Operator | Outside this RISC-V qualification |
| Optional storage and plugin combinations | Qualify the selected configuration separately |

The smoke test uses fixed responses to check model selection and transport.
Valkey-backed features and persistence configurations need separate integration tests.

## Build and run

Use the same checkout for SDK inputs, Dockerfiles, configuration and tests.
You need Docker with Buildx, access to its socket, Python 3.11+ for the smoke
fixture, and network access to public dependency sources. Native compilation
can take over an hour on the tested machine; keep enough disk space for the
SDK, build cache and runtime image.

From the repository root on the native machine:

```bash
python3 tools/docker/riscv64/build.py --tag public
```

This builds the SDK and `semantic-router-router:public`. Docker layer caching
is enabled by default; `--no-cache` explicitly requests a cold build. The SDK
uses checksummed toolchain archives and a digest-pinned base image. Package
repositories and dependency downloads still require network access.

The native build first exports the Router binary from
`tools/docker/riscv64/Dockerfile`. The independent
`tools/docker/Dockerfile.attached` packages it in the `vllm-sr-attached` target
without bundled PyTorch or `vllm-srun`; the bundled image Dockerfile is unchanged.
Configure reachable external endpoints before starting it.
Follow the repository's
[native build and deployment guide](https://github.com/vllm-project/semantic-router/blob/main/tools/docker/riscv64/README.md)
for the restricted-user `docker run` command, backend credentials and smoke
test. That guide also covers reusing an existing SDK and build concurrency.

Keep the request port bound to host loopback until authentication and network
access controls are in place. A ready Router does not prove that the backend
is reachable or that its credentials work; send a buffered request and an SSE
request through the full path.

## Image identity and distribution

The artifact identifier `vllm-sr-riscv64` maps to the attached-runtime target
and Linux/RISC-V64 only. It is separate from bundled CPU/CUDA/ROCm images;
do not add RISC-V to those manifests with different model-runtime contents.

Artifact registration does not itself publish an image. Until a registry
artifact and its qualification receipt are available, use the native build
above; there is no published RISC-V tag promised by this guide. Native
community CI and recurring publication require a native build worker and
the normal project qualification process.

See [Deployment Support](support-matrix) for the distinction between manual
qualification and recurring project coverage.

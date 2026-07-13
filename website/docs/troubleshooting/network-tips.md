---
title: Network Tips
sidebar_label: Network Tips
---

This guide shows how to build and run in restricted or slow network environments with the repository-native `make` and `vllm-sr` workflow. The canonical Dockerfiles accept mirror settings as build arguments, so no forked Dockerfile is required.

What you’ll solve:

- Hugging Face model downloads blocked/slow
- Go modules fetching blocked during Docker build
- Container image pulls blocked or slow

## TL;DR: Choose your path

- Fastest and most reliable: keep local models in `config/models` beside `config/config.yaml` and skip HF network entirely.
- Otherwise: set the Hugging Face mirror variables before `vllm-sr serve`.
- For source builds: pass `GOPROXY` and `GOSUMDB` to `make vllm-sr-dev`.
- Keep the Go checksum database enabled; a module proxy is not a substitute for integrity verification.

You can mix these based on your situation.

## 1. Hugging Face models

The router will download embedding models on first run unless you provide them locally. Prefer Option A if possible.

### Option A — Use local models (no external network)

1) Download the required model(s) with any reachable method (VPN/offline) into `config/models`, beside the repository's maintained `config/config.yaml`. Example layout:

   - `config/models/all-MiniLM-L12-v2/`
   - `config/models/category_classifier_modernbert-base_model`

2) In `config/config.yaml`, point to the local path. Example:

   ```yaml
   global:
     model_catalog:
       embeddings:
         semantic:
           # point to a local folder under /app/models (mounted by the local runtime)
           bert_model_path: /app/models/all-MiniLM-L12-v2
   ```

3) No extra environment variables are required. The local `vllm-sr` runtime mounts the config directory's `models` folder at `/app/models`.

### Option B — Use an HF endpoint mirror

Set a regional endpoint before starting the canonical local runtime (the example below uses a China mirror). `vllm-sr serve` passes `HF_ENDPOINT` into the router container:

```bash
export HF_ENDPOINT=https://hf-mirror.com
make vllm-sr-dev
vllm-sr serve --config config/config.yaml --image-pull-policy never
```

## 2. Build with Go mirrors

The repository's router and ExtProc Dockerfiles accept `GOPROXY` and `GOSUMDB` build arguments. Pass them through the canonical Make targets:

```bash
# CPU router image plus the local CLI and supporting images
GOPROXY=https://goproxy.cn,direct \
GOSUMDB=sum.golang.google.cn \
make vllm-sr-dev

# AMD/ROCm router image
GOPROXY=https://goproxy.cn,direct \
GOSUMDB=sum.golang.google.cn \
make vllm-sr-dev VLLM_SR_PLATFORM=amd

# NVIDIA/CUDA router image
GOPROXY=https://goproxy.cn,direct \
GOSUMDB=sum.golang.google.cn \
make vllm-sr-dev VLLM_SR_PLATFORM=nvidia

# Standalone ExtProc images, when those artifacts are needed
GOPROXY=https://goproxy.cn,direct \
GOSUMDB=sum.golang.google.cn \
make docker-build-extproc
```

Use a trusted HTTPS proxy and checksum database. Do not set `GOSUMDB=off`; that disables Go's public module-integrity check.

## 3. Build and run

Build from the repository, then start the exact local image through the supported CLI path:

```bash
# CPU
GOPROXY=https://goproxy.cn,direct \
GOSUMDB=sum.golang.google.cn \
make vllm-sr-dev
vllm-sr serve --config config/config.yaml --image-pull-policy never

# AMD/ROCm
GOPROXY=https://goproxy.cn,direct \
GOSUMDB=sum.golang.google.cn \
make vllm-sr-dev VLLM_SR_PLATFORM=amd
vllm-sr serve --config config/config.yaml --image-pull-policy never --platform amd

# NVIDIA/CUDA
GOPROXY=https://goproxy.cn,direct \
GOSUMDB=sum.golang.google.cn \
make vllm-sr-dev VLLM_SR_PLATFORM=nvidia
vllm-sr serve --config config/config.yaml --image-pull-policy never --platform nvidia
```

Keep the GPU build and serve platform flags paired as shown so the CLI never falls back to the CPU image. For NVIDIA prerequisites and execution-provider verification, see the repository's `docs/agent/nvidia-local.md` playbook.

## 4. Kubernetes clusters with limited egress

Container runtimes on Kubernetes nodes do not automatically reuse the host Docker daemon settings. When registries are slow or blocked, pods can sit in `ImagePullBackOff`. Pick one or combine several of these mitigations:

### 4.1 Configure containerd or CRI mirrors

- For clusters backed by containerd (Kind, k3s, kubeadm), edit `/etc/containerd/config.toml` or use Kind’s `containerdConfigPatches` to add regional mirror endpoints for registries such as `docker.io`, `ghcr.io`, or `quay.io`.
- Restart containerd and kubelet after changes so the new mirrors take effect.
- Avoid pointing mirrors to loopback proxies unless every node can reach that proxy address.

### 4.2 Preload or sideload images

- Build required images locally, then push them into the cluster runtime. For Kind, run `kind load docker-image --name <cluster> <image:tag>`; for other clusters, use `crictl pull` or `ctr -n k8s.io images import` on each node.
- Patch deployments to set `imagePullPolicy: IfNotPresent` when you know the image already exists on the node.

### 4.3 Publish to an accessible registry

- Tag and push images to a registry that is reachable from the cluster (cloud provider registry, privately hosted Harbor, etc.).
- Update your `kustomization.yaml` or Helm values with the new image name, and configure `imagePullSecrets` if the registry requires authentication.

### 4.4 Run a local pull-through cache

- Start a registry proxy (`registry:2` or vendor-specific cache) inside the same network, configure it as a mirror in containerd, and regularly warm it with the images you need.

### 4.5 Verify after adjustments

- Use `kubectl describe pod <name>` or `kubectl get events` to confirm pull errors disappear.
- Check that services such as `semantic-router-metrics` now expose endpoints and respond via port-forward (`kubectl port-forward svc/<service> <local-port>:<service-port>`).

## 5. Troubleshooting

- Go modules still time out:
  - Verify `GOPROXY` and `GOSUMDB` are present in the go-builder stage logs.
  - Run `make -n vllm-sr-build GOPROXY=<proxy> GOSUMDB=<sumdb>` to confirm the build arguments before retrying.
  - If a stale layer is suspected, copy that dry-run command, add `--no-cache` after the selected container runtime's `build`, and run it without pruning shared builder caches.

- HF models still download slowly:
  - Prefer Option A (local models).
  - Export `HF_ENDPOINT` in the same shell that runs `vllm-sr serve`.

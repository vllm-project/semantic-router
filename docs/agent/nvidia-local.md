# vLLM Semantic Router on NVIDIA CUDA

This playbook documents how to run the **router itself** (BERT
classifiers + embedding-driven KB lookup) on an NVIDIA GPU. Scope is
router-side ML inference — BERT-based intent classification, PII
detection, jailbreak guard, mmBERT embedding for Knowledge-Base
signals, hallucination mitigation, feedback detection. The backend
LLM is reached over the network and is **not** served from this host.

**How this relates to `amd-local.md`:** this file is the NVIDIA
equivalent of [`amd-local.md`](amd-local.md) — how to build the router
image on CUDA and serve it with `vllm-sr serve --platform nvidia`.
Both live under `docs/agent/` and have matching entries in
[`environments.md`](environments.md).

[`deploy/amd/README.md`](../../deploy/amd/README.md) is a different
concern: a *routing profile* guide (the `balance.yaml` recipe running
against a single ROCm vLLM backend). The naming looks parallel but
the topic isn't. `deploy/nvidia/` is intentionally left empty until a
matching NVIDIA routing profile is authored.

> **When this playbook is NOT for you:** upstream `onnx-binding/README.md`
> documents CPU≈GPU latency parity for BERT-size **embeddings** at small
> batches — the embedding path uses 2D-Matryoshka layer early-exit and
> gains little from a GPU. The **classifiers** are a different story (see
> [Performance (CPU vs GPU)](#performance-cpu-vs-gpu): ~65–240× latency,
> ~400× throughput). Measure first: if your routing is embedding-dominated
> at low QPS you may not need a GPU; reach for this playbook when you run
> classifier-heavy signal extraction, high classifier QPS, or want GPU
> co-location with other tenants.

---

## Overview

- Runtime image: `ghcr.io/vllm-project/semantic-router/vllm-sr-cuda:latest`
  (published to `ghcr.io` via `docker-publish.yml` on `main` / nightly and
  `docker-release.yml` on tag releases; local build still supported —
  see Step 1)
- Dockerfile: [`src/vllm-sr/Dockerfile.cuda`](../../src/vllm-sr/Dockerfile.cuda)
- Inference backend: **ONNX Runtime + CUDA Execution Provider**
  (mirrors how ROCm path uses ORT + ROCm EP — `candle-binding` import is
  rewritten to `onnx-binding` at module level via `go.onnx.mod`)
- Config overrides: added under `global.model_catalog` in your routing
  recipe (see Step 2)
- Verified hardware: RTX 4090 (compute_cap 8.9), CUDA driver 580
- Verified BERT-family modules on GPU:
  - `embeddings.semantic` (mmBERT 32k 2D-Matryoshka)
  - `prompt_guard` (jailbreak detector)
  - `classifier.domain` (intent classifier)
  - `classifier.pii` (PII detector)
  - `hallucination_mitigation.{fact_check, detector, explainer}`
  - `feedback_detector`

### Required runtime fix (must be in the image)

Building the CUDA image and setting `use_cpu: false` is **not sufficient**
on its own. The classifier execution-provider selection in onnx-binding
had a defect where the `Auto` provider only attempted ROCm, never CUDA,
so on a CUDA build every classifier (intent, PII, jailbreak, factcheck)
silently fell back to CPU. The embedding path also created unbounded CUDA
arenas that OOM'd later sessions. Both are fixed in:

- `onnx-binding/src/model_architectures/classification/mmbert_classifier.rs`
  — `Auto` arm now tries CUDA when the `cuda` feature is built.
- `onnx-binding/src/model_architectures/embedding/mmbert_embedding.rs`
  — CUDA sessions now use `with_memory_limit` + `SameAsRequested` arena
  strategy, matching the classifier path.

If you build the image from a tree that predates these fixes, the image
will run, attach the GPU, and still execute classifiers on CPU. Verify
with Step 4b below (you must see multiple `Using CUDA execution provider`
lines, not just one for embedding).

---

## Prerequisites

On the host:

1. NVIDIA driver supporting CUDA 12.x runtime (verified: driver 580 with
   CUDA 13.0 host runtime; backward-compatible with the CUDA 12.4
   container runtime used in the image).
2. `nvidia-container-toolkit` installed and configured for Docker so
   `--gpus all` and `--runtime nvidia` resolve.
3. Verify the host can see a GPU before continuing:

   ```bash
   nvidia-smi
   docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
   ```

   Both must list the same GPU. If the second fails, fix
   nvidia-container-toolkit before going further.

4. **A running `vllm-sr` deployment** (started via `vllm-sr serve` or
   your own orchestration). This playbook only swaps the router
   container; everything else (envoy, backend, dependencies) is left
   untouched. See the top-level project README for the standard
   `vllm-sr serve` bootstrap if you do not already have one running.

---

## Step 1: Pull or build the CUDA image

### 1a. Pull the published image (recommended)

`vllm-sr-cuda` is published to `ghcr.io` on every merge to `main`,
nightly, and each tag release (via
[`docker-publish.yml`](../../.github/workflows/docker-publish.yml) and
[`docker-release.yml`](../../.github/workflows/docker-release.yml)):

```bash
docker pull ghcr.io/vllm-project/semantic-router/vllm-sr-cuda:latest
```

Substitute the immutable tag (e.g. `v0.3.0`) for `latest` in production.

Available tags on `ghcr.io/vllm-project/semantic-router/vllm-sr-cuda`:

- `latest` — most recent `main`
- `nightly-<YYYYMMDD>` — nightly build from a specific date
- `v<X>.<Y>.<Z>` — immutable release tag

### 1b. Build locally (advanced / dev)

Local build is only needed to iterate on `Dockerfile.cuda` itself or to
build a tree that predates the published image. The image is
multi-stage: it cross-builds `candle-binding`, `ml-binding`,
`nlp-binding` (CPU-only by design), then builds `onnx-binding` with
`--features "cuda,ort/load-dynamic"`, then assembles a runtime stage on
top of `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` with
`onnxruntime-gpu==1.22.0`.

Recommended path — the Make target (wires `Dockerfile.cuda` +
`linux/amd64` + `vllm-sr-cuda` image name automatically):

```bash
cd /path/to/semantic-router
VLLM_SR_PLATFORM=nvidia make vllm-sr-build
```

The `VLLM_SR_PLATFORM=nvidia` block in
[`tools/make/docker.mk`](../../tools/make/docker.mk) resolves the router
image to `vllm-sr-cuda`, dockerfile to `src/vllm-sr/Dockerfile.cuda`,
and arch to `amd64`.

Equivalent raw `docker build` (if you cannot use `make`):

```bash
cd /path/to/semantic-router

DOCKER_BUILDKIT=1 docker build \
    --platform linux/amd64 \
    --build-arg TARGETARCH=amd64 \
    --build-arg BUILDPLATFORM=linux/amd64 \
    -f src/vllm-sr/Dockerfile.cuda \
    -t vllm-sr-cuda:local \
    .
```

Expected build time: **~20 minutes cold**, ~3 minutes warm. Expected
image size: **~4.2 GB** (vs ~440 MB for the default CPU image).

A per-Dockerfile ignore file
([`Dockerfile.cuda.dockerignore`](../../src/vllm-sr/Dockerfile.cuda.dockerignore))
excludes runtime-state directories (`.vllm-sr/`, `milvus-data/`,
`etcd/`, `postgres-data/`) so the build context loader does not hit
EACCES on root-owned subdirectories.

### Verify the image before swapping containers

The remaining steps in this playbook use the shell variable
`$VLLM_SR_CUDA_IMAGE` to refer to whichever image you picked in 1a or
1b — set it once here:

```bash
# Pulled path (1a):
export VLLM_SR_CUDA_IMAGE=ghcr.io/vllm-project/semantic-router/vllm-sr-cuda:latest

# Locally built path (1b):
# export VLLM_SR_CUDA_IMAGE=vllm-sr-cuda:local
```

```bash
# ORT inside the image must list CUDAExecutionProvider:
docker run --rm --gpus all --entrypoint /opt/vllm-sr-venv/bin/python \
    "$VLLM_SR_CUDA_IMAGE" -c '
import onnxruntime as ort
print("ORT version:", ort.__version__)
print("Providers:", ort.get_available_providers())
assert "CUDAExecutionProvider" in ort.get_available_providers()
print("OK")
'
```

Expected:

```
ORT version: 1.22.0
Providers: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
OK
```

If `CUDAExecutionProvider` is absent, do **not** continue. Re-check
that `--gpus all` works on the host and the build did not silently
fall back to the CPU wheel.

---

## Step 2: Apply the config override

The router defaults every BERT/embedding module to `use_cpu: true`
(see [`canonical_defaults.go`](../../src/semantic-router/pkg/config/canonical_defaults.go)).
Without overrides, even when the container has GPU access, the router
runs entirely on CPU.

Add the following override block under `global.model_catalog` in your
routing recipe (e.g. [`deploy/recipes/balance.yaml`](../../deploy/recipes/balance.yaml)
or whichever recipe your deployment loads):

```yaml
global:
  router:
    config_source: file
    strategy: priority
  model_catalog:
    embeddings:
      semantic:
        use_cpu: false
    modules:
      prompt_guard:
        use_cpu: false
      classifier:
        domain:
          use_cpu: false
        pii:
          use_cpu: false
      hallucination_mitigation:
        fact_check:
          use_cpu: false
        detector:
          use_cpu: false
        explainer:
          use_cpu: false
      feedback_detector:
        use_cpu: false
```

This block is a **no-op on CPU images** — ORT only exposes
`CPUExecutionProvider` there, so `use_cpu: false` simply has no GPU EP
to fall back to and the module stays on CPU. Safe to leave committed.

Note that `vllm-sr serve` reads the **transformed** runtime config
(`runtime-config.yaml`), not the source recipe directly. If your
tooling snapshots the recipe into a runtime config, make sure the
override flows all the way through to the runtime config the router
actually loads.

---

## Step 3: Start the router on the CUDA image

### Recommended: `vllm-sr serve --platform nvidia`

`vllm-sr serve --platform nvidia` now does everything this step used to
require by hand:

1. selects the published CUDA image
   (`ghcr.io/vllm-project/semantic-router/vllm-sr-cuda:latest`) — the
   same default the AMD path uses for ROCm,
2. attaches the GPU (`--gpus all` on Docker, CDI
   `nvidia.com/gpu=all` on Podman), and
3. flips `use_cpu` to `false` for the router internal models under
   `global.model_catalog`, so the Step 2 override block is applied
   automatically even if your recipe still defaults to CPU.

```bash
vllm-sr serve --platform nvidia --config <your-recipe>.yaml
```

Overrides, if you need them:

- `--image <ref>` or `VLLM_SR_IMAGE=<ref>` — pin a specific CUDA image
  (bypasses the default). `VLLM_SR_IMAGE_NVIDIA` sets the nvidia-only
  default without affecting other platforms.
- `VLLM_SR_NVIDIA_PRESERVE_CPU=1` — keep the recipe's `use_cpu` values
  instead of flipping them to `false` (use when the router shares a GPU
  that has no spare headroom).
- `VLLM_SR_NVIDIA_GPU_PASSTHROUGH=0` — skip `--gpus all` injection.

If steps 4a–4e below pass, you are done — skip the manual recipe.

### Fallback: manual `docker run` (older `vllm-sr`, or custom orchestration)

Use this only on a `vllm-sr` build that predates `--platform nvidia`
image selection, or when you drive the container yourself. `vllm-sr
serve` calls `docker run` internally with a fixed set of mounts, env
vars, ports, and network; to run the CUDA image by hand, recreate the
container with the same spec **plus** `--gpus all`.

### 3a. Capture the running container spec

```bash
# Save current container's env vars to a 600-mode file (may contain API keys)
docker inspect vllm-sr-router-container --format '{{range .Config.Env}}{{.}}
{{end}}' | \
    grep -E '^(ANTHROPIC_API_KEY|OPENAI_API_KEY|SR_LOG_LEVEL|OTEL_EXPORTER_OTLP_ENDPOINT|VLLM_SR_DASHBOARD_CONTAINER_NAME|VLLM_SR_ENVOY_CONTAINER_NAME|VLLM_SR_ROUTER_CONTAINER_NAME|VLLM_SR_RUNTIME_CONFIG_PATH|VLLM_SR_SOURCE_CONFIG_PATH)=' \
    > /tmp/router-env.list
chmod 600 /tmp/router-env.list
```

> Adjust the env-var allowlist to match whatever your deployment
> actually sets (backend API keys, telemetry endpoints, container
> names, config paths).
>
> Do **not** copy `PATH`, `LD_LIBRARY_PATH`, or `VIRTUAL_ENV` from the
> old container — the new image sets its own values (different paths
> for CUDA runtime libs). Pulling them across will break ORT's dynamic
> loader.

### 3b. Stop and remove the old container

```bash
docker stop vllm-sr-router-container
docker rm vllm-sr-router-container
```

### 3c. Start the new container with `--gpus all`

```bash
docker run -d \
    --name vllm-sr-router-container \
    --network vllm-sr-network \
    --gpus all \
    --add-host=host.docker.internal:host-gateway \
    -p 0.0.0.0:50051:50051 \
    -p 0.0.0.0:8080:8080 \
    -p 0.0.0.0:9190:9190 \
    --env-file /tmp/router-env.list \
    -v "$(pwd)/models:/app/models" \
    -v "$(pwd)/active-config.yaml:/app/config.yaml" \
    -v "$(pwd)/.vllm-sr:/app/.vllm-sr" \
    -w /app \
    "$VLLM_SR_CUDA_IMAGE" \
    /app/.vllm-sr/runtime-config.yaml /app/.vllm-sr
```

Adjust the host paths, ports, and network name to match your
deployment's layout.

---

## Step 4: Verify the router is on GPU

Wait ~30 seconds for model load, then run the checks below in order.

### 4a. Container is alive and not crash-looping

```bash
docker ps --filter "name=vllm-sr-router-container" --format \
    "table {{.Names}}\t{{.Status}}\t{{.Image}}"
```

Expected: `Up <N> seconds`, image matches `$VLLM_SR_CUDA_IMAGE` from
Step 1 (either
`ghcr.io/vllm-project/semantic-router/vllm-sr-cuda:latest` or
`vllm-sr-cuda:local`).

### 4b. Router actually selected the CUDA EP

```bash
docker logs vllm-sr-router-container 2>&1 | \
    grep "Using CUDA execution provider"
```

Expected (one line per loaded ONNX model):

```
INFO: Using CUDA execution provider (NVIDIA GPU) — verified
INFO: Using CUDA execution provider (NVIDIA GPU) — verified
...
```

If you see `Using CPU execution provider` everywhere, work through these
causes in order:

1. **Config override not applied** — re-check that `runtime-config.yaml`
   (not just the source recipe) contains the `model_catalog` block;
   the router reads `runtime-config.yaml`.
2. **Image predates the EP fix** — see "Required runtime fix" in the
   Overview. Without it, embedding loads on CUDA but every classifier
   stays on CPU regardless of config.
3. **Out of VRAM** — see "Some models load on CUDA, others fall back to
   CPU" in Troubleshooting; check `nvidia-smi` free memory.

### 4c. GPU memory actually allocated to BERT weights

```bash
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
```

Compared with a CPU-only baseline you should see **~3 GB additional
memory used** (BERT mmBERT + classifier models resident on device).
Concrete observed delta on RTX 4090: `17986 MiB → 21139 MiB`
(+3153 MiB).

### 4d. `embedding_models_init_started` shows `use_cpu: false`

```bash
docker logs vllm-sr-router-container 2>&1 | \
    grep embedding_models_init_started | tail -1 | python3 -m json.tool
```

Expected: `"use_cpu": false`.

### 4e. KB classifier is operational

```bash
docker logs vllm-sr-router-container 2>&1 | \
    grep knowledge_base_classifier_initialized
```

Expected: one `knowledge_base_classifier_initialized` line per KB in
the active config.

---

## Step 5: Sanity-check a real query

Send any prompt through the router and confirm it routes successfully:

```bash
curl -sS -X POST http://localhost:8899/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "auto",
      "messages": [{"role": "user", "content": "What is the LEED standard?"}]
    }' | head
```

The router applies its decision tree (n-gram → KB embedding similarity
→ decision rule), routes to the appropriate backend, and the response
flows back through envoy. If steps 4a-4e all pass and a real query
returns 200, you are done.

---

## Performance (CPU vs GPU)

Measured with the same harness and metric as the ROCm
[`bench/cpu-vs-gpu/`](../../bench/cpu-vs-gpu/) benchmark — Prometheus
`llm_signal_extraction_latency_seconds` — ported to CUDA (`--gpus all`).
Both CPU and GPU runs use the router's real signal-extraction path, so the
comparison is apples-to-apples.

- **Hardware**: NVIDIA GeForce RTX 4090 (24 GB), consumer GPU — a baseline,
  not a datacenter number.
- **Model**: mmBERT-32K domain/intent classifier.
- 12 requests per size (+4 warmup), batch 1.

### Signal extraction latency — all three classifiers (avg ms)

All three signal classifiers (domain/intent, jailbreak, PII) extracted per
request. GPU average latency is stable; CPU is reported alongside (see the
variance caveat below).

| Prompt tokens | domain CPU | domain GPU | jailbreak CPU | jailbreak GPU | PII CPU | PII GPU |
|--------------:|-----------:|-----------:|--------------:|--------------:|--------:|--------:|
| ~500          | 1,104      | 6.2        | 1,634         | 10.6          | 1,144   | 5.6     |
| ~2,000        | 2,453      | 11.6       | 2,239         | 10.7          | 1,920   | 7.9     |
| ~8,000        | 2,157      | 19.8       | 2,474         | 16.5          | 1,868   | 13.8    |
| ~16,000       | 2,082      | 32.1       | 2,700         | 28.3          | 1,977   | 19.7    |

GPU signal extraction stays in the single-digit-to-low-tens of ms across the
whole 500–16K range for every classifier; CPU sits in the ~1–2.7 s band. Net:
**GPU is roughly two orders of magnitude faster** (~65–240× on average across
signals and sizes). The same ~100× gap reproduced across three independent
measurement paths (the Prometheus metric above, the router's `[Perf] classifier
inference` log line, and a direct ONNX Runtime CPU-vs-CUDA EP micro-benchmark),
so it is not a harness artifact.

> **Why this does not contradict the "CPU ≈ GPU" note below or in
> [`onnx-binding/README.md`](../../onnx-binding/README.md).** That parity result
> is for the **embedding** model with 2D-Matryoshka *layer early-exit* — much
> less compute per call. The table above is the full-depth **classifier**. Light
> early-exit embedding ≈ CPU/GPU parity; full classifier = large GPU win.

### Throughput under concurrency

The ext_proc classifier path handles one request at a time — there is no batch
knob like an LLM server, so the real-world analog of "batch size / throughput"
is concurrency: N clients hitting the router at once. Sustained over 15 s at a
fixed ~1K-token prompt:

| Concurrency | CPU QPS | CPU P50 | GPU QPS | GPU P50 |
|------------:|--------:|--------:|--------:|--------:|
| 1           | 0.2     | 5.2 s   | 65      | 15 ms   |
| 8           | 0.2     | 29 s    | 78      | 101 ms  |
| 16          | 0.2     | 61 s    | 79      | 200 ms  |
| 32          | 0.2     | 114 s   | 80      | 399 ms  |

CPU throughput is capped at ~0.2 req/s — adding clients does **not** raise it,
it just makes requests queue and latency blow up (5 s → 114 s). GPU sustains
~80 req/s and degrades gracefully (P50 stays sub-second at 32-way concurrency).
That is roughly a **400× throughput** difference, and it is the clearest reason
to put the router's classifiers on a GPU: not single-request speed, but keeping
classification off the critical path under load.

**Caveats.**

- **Bounded classifier input (512 tokens).** The classifier caps its input at
  512 tokens by design — see `MAX_CLASSIFICATION_SEQ_LEN` in
  `onnx-binding/src/model_architectures/classification/mmbert_classifier.rs`
  (added in `547adc6e`, "handle long prompt without oom"). So the model sees at
  most 512 tokens regardless of prompt length, and classifier latency is
  effectively **bounded / decoupled from context length** — that is intended
  behavior, not a benchmark artifact. The sweep across 500–16K therefore
  measures bounded classification plus the per-request tokenization/preprocess
  of the full prompt, not deeper model work at longer contexts.
- **What actually grows with prompt length** is tokenization of the full input
  (done on the host CPU before truncation). It shows up cleanly on GPU (~6 → 32
  ms across 500 → 16K). The larger, non-monotonic swings in the CPU column are
  shared-host noise (heavy tail, P95 ≫ avg), not a real context-length trend —
  so treat CPU as order-of-magnitude and the speedup as a conservative band.
- **Where the GPU win matters**: high classifier QPS / throughput (CPU
  seconds-per-request becomes a bottleneck under load) and GPU co-tenancy — not
  single-request end-to-end latency, which is dominated by LLM token generation,
  not classification.

---

## Rollback

If the CUDA image misbehaves, rollback is one container restart on the
default CPU image:

```bash
docker stop vllm-sr-router-container
docker rm vllm-sr-router-container
# bring the router back up on the default CPU image via your normal
# `vllm-sr serve` flow
```

The `model_catalog` overrides remain in the recipe but are silently
ignored on the CPU image, so no further config change is needed.

---

## Troubleshooting

### `huggingface-cli not found` at router startup

Symptom: container exits with code 1, log ends with:

```
huggingface-cli check failed: huggingface-cli not found
```

Cause: `huggingface_hub==1.5.0` ships the `hf` CLI but does not pull
`click` as a hard dependency under Python 3.10. The runtime image is
Ubuntu 22.04 + Python 3.10.

Fix: this is already addressed in
[`Dockerfile.cuda`](../../src/vllm-sr/Dockerfile.cuda) (the pip
install line includes `'click>=8.1.7'`). If you see this on a stale local
build, rebuild the image.

### `Error: mmBERT model not initialized` in the log

Some initial errors during the **first second** of startup are normal —
the KB classifier tries to embed exemplars before the model finishes
loading and retries them once it does. They should stop within the
first 1-2 seconds. If errors continue indefinitely, the model never
loaded; check:

```bash
docker logs vllm-sr-router-container 2>&1 | \
    grep -iE "Selected mmBERT ONNX|Loading layer|Using.*execution provider"
```

You should see one `Selected mmBERT ONNX file: ...` and several
`Loading layer-N from ...` lines. If they are absent, the model files
are not visible inside the container — re-check the
`-v $(pwd)/models:/app/models` bind mount.

### `--gpus all` rejected with "could not select device driver"

Cause: `nvidia-container-toolkit` is not installed or not registered as
a Docker runtime. Reinstall and reload:

```bash
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify with:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### GPU memory does not increase after restart

Most likely cause: you edited the source recipe (the snapshot) but did
**not** regenerate the runtime config (`runtime-config.yaml`). The
router reads `runtime-config.yaml`, not the source recipe.

Fix: either re-run your config-transform step (full regenerate) or
patch `runtime-config.yaml` directly for a quick test.

### Performance still feels identical to CPU

This applies to the **embedding** path, not the classifier. For small-batch
mmBERT-32K *embedding* with 2D-Matryoshka layer early-exit, upstream
[`onnx-binding/README.md`](../../onnx-binding/README.md) reports CPU ≈ GPU
latency at batch size 1 for sequence lengths up to 512. The full-depth
*classifier* is a different story — GPU is ~65–240× faster there (see
[Performance (CPU vs GPU)](#performance-cpu-vs-gpu) above). If your workload is
embedding-dominated, the real win from this image is:

- **High-QPS scenarios** (many concurrent queries hitting the embedding
  model at once)
- **Co-tenancy** (you already need GPU on the same host for vLLM and
  want a single hardware accelerator footprint)
- **Future model upgrades** (when you move to larger embedding /
  classifier models that GPUs accelerate proportionally more)

For today's small mmBERT-32k workload at low QPS, the CPU image is
genuinely competitive. Benchmark before and after under your **actual**
QPS pattern before declaring victory.

### Some models load on CUDA, others fall back to CPU (mixed state)

Symptom: the startup log shows a mix of `Using CUDA execution provider`
and `Using CPU execution provider`, and may include:

```
WARN: CUDA EP failed: ... bfc_arena.cc ... Failed to allocate memory ...
WARN: CUDA EP failed: ... CUBLAS_STATUS_ALLOC_FAILED ...
```

Cause: not enough free VRAM. The router opens several CUDA sessions
(one primary embedding session, one per 2D-Matryoshka early-exit layer,
plus one per classifier). Each session needs a cuBLAS handle and an arena
block. On a GPU shared with other tenants (e.g. a co-located vLLM serving
a large model), the later sessions OOM and silently fall back to CPU.

Diagnose:

```bash
# How much VRAM is actually free, and who is using it
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

If `memory.free` is only a few hundred MB while another process holds
most of the card, that is the cause — not a code bug.

Fixes, in order of preference:

1. **Cap per-session arena size.** Set `ORT_GPU_MEM_LIMIT` to a small
   value (e.g. `512MB`) so every session fits into the limited free VRAM:

   ```bash
   # via the container env
   docker run ... -e ORT_GPU_MEM_LIMIT=512MB "$VLLM_SR_CUDA_IMAGE" ...
   ```

   The embedding and classifier CUDA branches both honor
   `get_gpu_mem_limit()`, which reads this env var first (supports
   `512MB`, `4GB`, or raw bytes). Without it they auto-probe free VRAM,
   and the auto-probe falls back to 4 GB per session when it cannot read
   the GPU — too large on a contended card.

   > Note: `ORT_GPU_MEM_LIMIT` may not be in the `vllm-sr` CLI
   > passthrough allowlist (`runtime_support.py: PASSTHROUGH_ENV_RULES`).
   > To use it via `vllm-sr serve` you must either add it to that
   > allowlist or use the manual `docker run` path (Step 3c).

2. **Free up VRAM** by stopping or shrinking the co-tenant process
   (other vLLM instance, etc.).

3. **Accept the mixed state.** A CPU-resident classifier still works
   correctly; for the small mmBERT-32k models the latency difference is
   minor (see the section above). The bounded-arena fix degrades to CPU
   gracefully rather than crashing.

To verify which model fell back, align the `Using CPU` lines with the
nearest preceding `*_detector_backend_loading` / `Loading layer-N` log
line.

### Confirm the GPU is actually computing (not just holding weights)

Memory being resident on the GPU proves the model loaded there, but not
that inference runs on it. To confirm live GPU compute, monitor the
router process's SM utilization while sending classification requests:

```bash
# Find the router's GPU PID
ROUTER_PID=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader \
  | grep router | cut -d, -f1 | tr -d ' ')

# Sample per-process SM% for 10s; send requests in the middle
nvidia-smi pmon -i 0 -c 10 -o T &
sleep 3
for i in $(seq 1 15); do
  curl -s -o /dev/null -X POST http://localhost:8080/api/v1/classify/intent \
    -H "Content-Type: application/json" \
    -d '{"text":"Explain TCP vs UDP and when to use each"}'
done
```

The router PID's `sm` column should rise from `-`/`0` (idle) to a small
positive value during the request burst, then return to idle. That
transition is the proof GPU compute happened. (The value stays low for
small BERT models on a fast GPU — that is expected, not a problem.)

---

## Router image differences: NVIDIA CUDA vs AMD ROCm

Apples-to-apples comparison of the two GPU-enabled router images,
sourced from `src/vllm-sr/Dockerfile.cuda` and
`src/vllm-sr/Dockerfile.rocm`:

| Aspect | AMD ROCm router (`vllm-sr-rocm`) | NVIDIA CUDA router (`vllm-sr-cuda`) |
|---|---|---|
| Published tag | `ghcr.io/vllm-project/semantic-router/vllm-sr-rocm:latest` | `ghcr.io/vllm-project/semantic-router/vllm-sr-cuda:latest` |
| Base image | `rocm/dev-ubuntu-22.04:7.0` | `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` |
| ORT wheel | `onnxruntime_rocm-1.22.1` (radeon repo) | `onnxruntime-gpu==1.22.0` (PyPI) |
| `onnx-binding` feature | `--features rocm-dynamic` | `--features "cuda,ort/load-dynamic"` |
| GPU passthrough | `--device=/dev/kfd --device=/dev/dri` | `--gpus all` (Docker) or CDI `nvidia.com/gpu=all` (Podman) |
| CLI entrypoint | `vllm-sr serve --platform amd` | `vllm-sr serve --platform nvidia` (selects the CUDA image + flips `use_cpu` to false, at parity with the AMD path) |

The two images are independent; do not mix `--gpus all` with
`--device=/dev/kfd` on the same router container.

---

## Files involved

- [`src/vllm-sr/Dockerfile.cuda`](../../src/vllm-sr/Dockerfile.cuda) —
  the runtime image, mirrored from `Dockerfile.rocm`
- [`src/vllm-sr/Dockerfile.cuda.dockerignore`](../../src/vllm-sr/Dockerfile.cuda.dockerignore) —
  per-Dockerfile ignore, excludes runtime-state dirs
- [`onnx-binding/Cargo.toml`](../../onnx-binding/Cargo.toml) — feature
  graph (`cuda`, `rocm`, `rocm-dynamic`, etc.); read this if you need
  to understand or change which EP is compiled in
- [`src/semantic-router/go.onnx.mod`](../../src/semantic-router/go.onnx.mod) —
  the `replace` directive that rewires `candle-binding` Go imports to
  `onnx-binding` at module level. This is the mechanism that lets a
  config field named `embeddings.semantic.use_cpu` (originally a candle
  concept) actually steer the ONNX Runtime EP selection
- [`src/semantic-router/pkg/config/canonical_defaults.go`](../../src/semantic-router/pkg/config/canonical_defaults.go) —
  source of truth for `use_cpu: true` defaults
- [`tools/make/docker.mk`](../../tools/make/docker.mk) — the
  `VLLM_SR_PLATFORM=nvidia` block wiring `Dockerfile.cuda` +
  `vllm-sr-cuda` image name + `linux/amd64` into the Make path
- [`.github/workflows/docker-publish.yml`](../../.github/workflows/docker-publish.yml) —
  publishes `vllm-sr-cuda` to `ghcr.io` on merge to `main` and nightly
  (amd64-only, matching ROCm)
- [`.github/workflows/docker-release.yml`](../../.github/workflows/docker-release.yml) —
  publishes tagged `vllm-sr-cuda` releases on `v*` tags

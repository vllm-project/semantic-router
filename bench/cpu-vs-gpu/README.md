# CPU vs GPU / SDPA vs FA / BUFFERED vs STREAMED Benchmarks

Measures signal extraction latency (jailbreak, PII, domain) for ONNX Runtime on AMD ROCm GPUs via Envoy ext_proc, using Prometheus histograms.

## Prerequisites

- AMD GPU with ROCm 7.0+ (`/dev/kfd`, `/dev/dri`)
- Docker
- `envoyproxy/envoy:v1.33-latest` image

## Setup

Build the router image (includes CK Flash Attention custom op):

```bash
docker build -f tools/docker/Dockerfile.extproc-rocm -t semantic-router:rocm .
```

Download models into `bench/cpu-vs-gpu/models/`:

```bash
pip install huggingface_hub
python3 -c "
from huggingface_hub import snapshot_download
for repo in [
    'mmbert32k-intent-classifier-merged',
    'mmbert32k-jailbreak-detector-merged',
    'mmbert32k-pii-detector-merged',
]:
    snapshot_download(
        f'llm-semantic-router/{repo}',
        local_dir=f'bench/cpu-vs-gpu/models/{repo}-onnx',
        allow_patterns=['onnx/*', '*.json'],
        ignore_patterns=['*.safetensors', '*.bin', '*.pt'],
    )
"
```

Each model dir needs `model_sdpa_fp16.onnx` (for SDPA/CPU) and `model_fa_fp16.onnx` (for FA). Generate FA models with `onnx-binding/ort-ck-flash-attn/scripts/rewrite_graph.py` if not already present.

## Benchmarks

### CPU vs GPU

Compares ONNX CPU vs ROCm GPU across 500/2K/8K/16K token prompts:

```bash
BENCH_IMAGE=semantic-router:rocm REQUESTS_PER_SIZE=10 ./bench-long-context.sh
```

### SDPA vs Flash Attention

Compares standard attention vs CK Flash Attention on GPU:

```bash
BENCH_IMAGE=semantic-router:rocm NUM_REQUESTS=20 ./bench-sdpa-vs-fa.sh
```

### ONNX GPU vs ONNX CPU vs Candle CPU

Compares the same `jailbreak`, `pii`, and `domain` signals across ONNX GPU, ONNX CPU, and Candle CPU execution:

```bash
BENCH_IMAGE=semantic-router:rocm \
CANDLE_IMAGE=semantic-router:candle-bench \
REQUESTS_PER_SIZE=10 ./bench-3way.sh
```

### BUFFERED vs STREAMED (E2E body mode comparison)

Compares the original Envoy `BUFFERED` body mode (full `json.Unmarshal`/`Marshal`) against the new `STREAMED` mode with gjson/sjson fast-path JSON processing, semi-streaming chunked body delivery, and prompt compression.

The STREAMED variant builds the patched binary from source directly inside the base container (volume-mounting the repo), so no separate Docker image is needed.

```bash
# GPU + Flash Attention (recommended — makes JSON/streaming overhead visible)
BASE_IMAGE=semantic-router:rocm-fa USE_GPU=true \
    REQUESTS_PER_SIZE=10 WARMUP_REQUESTS=3 ./bench-buffered-vs-streamed.sh

# CPU-only (signal extraction dominates, streaming gains are proportionally smaller)
BASE_IMAGE=semantic-router:rocm USE_GPU=false \
    REQUESTS_PER_SIZE=10 ./bench-buffered-vs-streamed.sh
```

The script:

1. Runs the **BUFFERED** variant using the stock base image with `request_body_mode: BUFFERED` and `global.router.streamed_body.enabled` set to `false`
2. Runs the **STREAMED** variant by building the patched binary inside the container, using `request_body_mode: STREAMED` and `global.router.streamed_body.enabled` set to `true`
3. Collects E2E latency (curl timing) and signal extraction latency (Prometheus histograms) at 500/2K/8K/16K token sizes
4. Generates a markdown comparison report in `results/`

#### Sample results (rocm-fa, MI300X GPU)

| Tokens | BUFFERED (ms) | STREAMED (ms) | Reduction |
|--------|--------------|---------------|-----------|
| ~500   | 17           | 17            | 0%        |
| ~2000  | 25           | 21            | 16%       |
| ~8000  | 63           | 45            | 29%       |
| ~16000 | 143          | 103           | 28%       |

Jailbreak signal extraction at 16K tokens drops from 127ms to 10ms (prompt compression: 16K → 512 tokens).

Reports are written to `results/`.

## Scripts

| File | Description |
|------|-------------|
| `bench-3way.sh` | ONNX GPU vs ONNX CPU vs Candle CPU latency comparison |
| `bench-long-context.sh` | CPU vs GPU, multi token-size, Prometheus metrics |
| `bench-sdpa-vs-fa.sh` | SDPA vs FA on GPU, Prometheus metrics |
| `bench-buffered-vs-streamed.sh` | BUFFERED vs STREAMED body mode, builds patched binary inside container |
| `config-bench.yaml` | Canonical v0.3 router config template for ONNX benchmarks |
| `config-bench-candle.yaml` | Canonical v0.3 router config template for Candle CPU benchmarks |
| `envoy-bench.yaml` | Envoy ext_proc proxy config (STREAMED mode) |
| `envoy-bench-fa.yaml` | Envoy ext_proc proxy config for FA benchmarks |

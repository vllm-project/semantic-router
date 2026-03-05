# CPU vs GPU / SDPA vs FA Benchmarks

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

**CPU vs GPU** — compares ONNX CPU vs ROCm GPU across 500/2K/8K/16K token prompts:

```bash
BENCH_IMAGE=semantic-router:rocm REQUESTS_PER_SIZE=10 ./bench-long-context.sh
```

**SDPA vs Flash Attention** — compares standard attention vs CK Flash Attention on GPU:

```bash
BENCH_IMAGE=semantic-router:rocm NUM_REQUESTS=20 ./bench-sdpa-vs-fa.sh
```

Reports are written to `results/`.

## Scripts

| File | Description |
|------|-------------|
| `bench-long-context.sh` | CPU vs GPU, multi token-size, Prometheus metrics |
| `bench-sdpa-vs-fa.sh` | SDPA vs FA on GPU, Prometheus metrics |
| `config-bench.yaml` | Router config template (`USE_CPU_PLACEHOLDER` sed-replaced) |
| `envoy-bench.yaml` | Envoy ext_proc proxy config |

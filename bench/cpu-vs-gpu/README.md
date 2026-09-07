# Signal extraction backend benchmarks

These scripts compare the latency of the `jailbreak`, `pii`, and `domain`
signals through Envoy ExtProc. They use ONNX Runtime on CPU or AMD ROCm and
read signal latency from the Router's Prometheus histograms.

Use the results to compare backends on one controlled host. Numbers from
different machines, model revisions, prompt sets, or warmup settings are not
directly comparable.

## Prerequisites

- an AMD GPU with ROCm 7.0 or later for GPU runs
- Docker with access to `/dev/kfd` and `/dev/dri`
- Python 3 and `huggingface_hub` for model download
- the `envoyproxy/envoy:v1.33-latest` image, or an override through
  `ENVOY_IMAGE`

Run all commands from the repository root.

## Prepare the image and models

```bash
docker build \
  -f tools/docker/Dockerfile.extproc-rocm \
  -t semantic-router:rocm .

python3 -m pip install huggingface_hub
python3 - <<'PY'
from huggingface_hub import snapshot_download

for name in (
    "mmbert32k-intent-classifier-merged",
    "mmbert32k-jailbreak-detector-merged",
    "mmbert32k-pii-detector-merged",
):
    snapshot_download(
        repo_id=f"llm-semantic-router/{name}",
        local_dir=f"bench/cpu-vs-gpu/models/{name}-onnx",
        allow_patterns=["onnx/*", "*.json"],
        ignore_patterns=["*.safetensors", "*.bin", "*.pt"],
    )
PY
```

Each downloaded model must contain `onnx/model_sdpa_fp16.onnx`. The Flash
Attention comparison also needs `onnx/model_fa_fp16.onnx`; generate it with
[`rewrite_graph.py`](../../onnx-binding/ort-ck-flash-attn/scripts/rewrite_graph.py)
when the model repository does not provide one.

## Run a benchmark

CPU versus GPU across approximately 500, 2K, 8K, and 16K tokens:

```bash
BENCH_IMAGE=semantic-router:rocm \
REQUESTS_PER_SIZE=10 \
./bench/cpu-vs-gpu/bench-long-context.sh
```

Standard attention versus CK Flash Attention on the GPU:

```bash
BENCH_IMAGE=semantic-router:rocm \
NUM_REQUESTS=20 \
./bench/cpu-vs-gpu/bench-sdpa-vs-fa.sh
```

An additional developer harness, `bench-3way.sh`, compares ONNX GPU, ONNX
CPU, and Candle CPU. It uses `config-bench-candle.yaml` and expects the three
Candle model directories under `bench/cpu-vs-gpu/models/candle/`; model
preparation is intentionally separate from the ONNX quick path above. Both
benchmark templates enable `global.router.streamed_body.enabled`, so these
runs compare inference backends rather than buffered and streamed request
handling.

Both scripts warm up the runtime before collecting samples. Override
`WARMUP_REQUESTS` when needed, but keep it constant across compared runs.
ROCm may compile kernels during the first inference, so a cold run can take
several minutes.

The scripts use the container names `sr-bench` and `envoy-bench` and remove
existing containers with those exact names. Do not run them alongside another
job that owns those names.

## Results

Timestamped reports and raw metrics are written under
`bench/cpu-vs-gpu/results/`. Keep the generated report with the hardware,
image digest, model revisions, and environment overrides used for the run;
the repository does not treat a sample from one host as a performance
guarantee.

## Maintained files

| File | Purpose |
| --- | --- |
| `bench-long-context.sh` | Compare ONNX CPU and ROCm GPU by prompt length. |
| `bench-sdpa-vs-fa.sh` | Compare SDPA and CK Flash Attention on ROCm. |
| `bench-3way.sh` | Add Candle CPU to the ONNX CPU/GPU comparison. |
| `config-bench.yaml` | Canonical v0.3 Router config template. |
| `config-bench-candle.yaml` | Candle variant used only by the 3-way harness. |
| `envoy-bench.yaml` | Envoy ExtProc and metrics listener config. |

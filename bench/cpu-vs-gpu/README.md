# Signal extraction backend benchmarks

These scripts compare the latency of the `jailbreak`, `pii`, and `domain`
signals through Envoy ExtProc. They use ONNX Runtime on CPU or AMD ROCm and
read signal latency from the Router's Prometheus histograms.

Use the results to compare backends on one controlled host. Numbers from
different machines, model revisions, prompt sets, or warmup settings are not
directly comparable.

## Prerequisites

- an AMD GPU with ROCm 7.0 or later for the ROCm scripts, or an NVIDIA GPU
  with `nvidia-container-toolkit` for the `bench-cuda-*` scripts
- Docker with access to `/dev/kfd` and `/dev/dri`, or a working
  `docker run --gpus all`
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

## Run the NVIDIA CUDA benchmarks

`bench-cuda-long-context.sh` is the CUDA counterpart of
`bench-long-context.sh`: same metric, same prompt sizes, `--gpus all` instead
of the ROCm device flags, and all three classifiers (domain, jailbreak, PII)
exercised per request through `config-bench-cuda.yaml`. It needs the router
image built from [`src/vllm-sr/Dockerfile.cuda`](../../src/vllm-sr/Dockerfile.cuda)
and the model directories without the `-onnx` suffix the ROCm path uses:

```bash
python3 - <<'PY'
from huggingface_hub import snapshot_download

for name in (
    "mmbert32k-intent-classifier-merged",
    "mmbert32k-jailbreak-detector-merged",
    "mmbert32k-pii-detector-merged",
    "mmbert-embed-32k-2d-matryoshka",
):
    snapshot_download(
        repo_id=f"llm-semantic-router/{name}",
        local_dir=f"bench/cpu-vs-gpu/models/{name}",
        allow_patterns=["onnx/*", "*.json"],
        ignore_patterns=["*.safetensors", "*.bin", "*.pt"],
    )
PY

BENCH_IMAGE=vllm-sr-cuda:local \
REQUESTS_PER_SIZE=10 \
./bench/cpu-vs-gpu/bench-cuda-long-context.sh
```

`bench-cuda-throughput.sh` measures throughput instead of single-request
latency. ExtProc classifies one request at a time, so the analog of batch size
is concurrency: N clients over a fixed duration.

```bash
BENCH_IMAGE=vllm-sr-cuda:local \
CONCURRENCIES="1 8 16 32" \
./bench/cpu-vs-gpu/bench-cuda-throughput.sh
```

Both scripts refuse to publish numbers from a run that did not classify.
Every benchmark request must return HTTP 200, and each of the three signals
must record at least one new `llm_signal_extraction_latency_seconds` sample
per successful request; either check failing aborts the run with a non-zero
exit instead of writing a report. `signal_samples.py` owns that comparison.
The GPU phase additionally requires every prepared classifier to report a
non-CPU device, so a phase that silently fell back to the CPU cannot be
published as a speedup.

These scripts use the container names `sr-bench-cuda`, `sr-bench-cuda-tp`,
`envoy-bench-cuda`, and `envoy-bench-cuda-tp`. Every port is overridable
(`EXTPROC_PORT`, `API_PORT`, `METRICS_PORT`, `ENVOY_PORT`, `STUB_PORT`) for
hosts where the defaults are taken; a built-in stub upstream answers on
`STUB_PORT` so the routed request completes.

The classifier caps its input at 512 tokens, so classifier latency is
expected to stay flat across the 500–16K sweep. The longer prompts exercise
tokenization and the ExtProc path rather than deeper inference.

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
| `bench-cuda-long-context.sh` | Compare ONNX CPU and CUDA GPU by prompt length. |
| `bench-cuda-throughput.sh` | Compare ONNX CPU and CUDA GPU under concurrency. |
| `load_test.py` | Concurrent request driver for the CUDA throughput script. |
| `signal_samples.py` | Verify signal extraction ran before numbers are published. |
| `config-bench-cuda.yaml` | Canonical v0.3 config for the CUDA benchmarks. |

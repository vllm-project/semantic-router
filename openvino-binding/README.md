# OpenVINO Binding for Semantic Router

OpenVINO Binding provides native C++ and Go integrations for running semantic routing models with Intel OpenVINO.
It supports sequence classification, token classification, and embedding inference, and can be compared directly against Candle backend performance.

## What Is Included

- OpenVINO native runtime integration through CGO
- ModernBERT sequence classification APIs
- ModernBERT token classification APIs
- ModernBERT embedding APIs
- Benchmark programs for classifier and embedding comparison (OpenVINO vs Candle)

## Repository Layout

- `cpp/`: C++ implementation
- `semantic-router.go`: Go binding surface
- `bench/mmbert_classifier_bench.go`: sequence classifier benchmark (OpenVINO vs Candle)
- `bench/mmbert_embedding_bench.go`: embedding benchmark (OpenVINO vs Candle)
- `scripts/build_and_run_mmbert_classifier_bench.sh`: classifier benchmark automation script
- `scripts/build_and_run_mmbert_embedding_bench.sh`: embedding benchmark automation script

## Prerequisites

- OpenVINO runtime installed
- Go toolchain
- CMake and C++ compiler
- Python (for tokenizer/model conversion)
- Optional but recommended for conversion:
  - `optimum[openvino]`
  - `openvino-tokenizers`
  - `transformers`

## Build

From repository root:

```bash
cd openvino-binding
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

Build Go binding checks:

```bash
cd openvino-binding
go build -v ./...
```

## Benchmarks (Primary Entry: Script/Make)

Use script or Make targets as the default operational path.
Manual `go run` is for debugging only (see Appendix).

### 1) Sequence Classifier Benchmark

Run via script:

```bash
bash openvino-binding/scripts/build_and_run_mmbert_classifier_bench.sh
```

Common modes:

```bash
bash openvino-binding/scripts/build_and_run_mmbert_classifier_bench.sh --build-only
bash openvino-binding/scripts/build_and_run_mmbert_classifier_bench.sh --run-only
```

Run via Make:

```bash
make benchmark-openvino-classifier
make benchmark-openvino-classifier ARGS='--build-only'
make benchmark-openvino-classifier ARGS='--run-only'
```

Useful overrides (environment variables):

- `CLASSIFIER_MODEL_DIR`
- `CLASSIFIER_OPENVINO_MODEL_PATH`
- `CLASSIFIER_CANDLE_MODEL_PATH`
- `CLASSIFIER_MIN_MATCH_RATE`
- `CLASSIFIER_MAX_CONFIDENCE_DELTA`
- `OPENVINO_TOKENIZERS_LIB`
- `OPENVINO_LIB_DIR`

### 2) Embedding Benchmark

Run via script:

```bash
bash openvino-binding/scripts/build_and_run_mmbert_embedding_bench.sh
```

Common modes:

```bash
bash openvino-binding/scripts/build_and_run_mmbert_embedding_bench.sh --build-only
bash openvino-binding/scripts/build_and_run_mmbert_embedding_bench.sh --run-only
```

Run via Make:

```bash
make benchmark-openvino-embedding
make benchmark-openvino-embedding ARGS='--build-only'
make benchmark-openvino-embedding ARGS='--run-only --length-profile fixed-128'
```

Useful script arguments:

- `--stage-timing`
- `--stage-timing-samples N`
- `--length-profile mixed|fixed-32|fixed-128|fixed-512|fixed-1024|fixed-2048`

When tokenized text length exceeds `OV_MAX_LENGTH`, benchmark logs a `note:` line indicating truncation before embedding.

Useful overrides (environment variables):

- `MMBERT_MODEL_PATH`
- `OPENVINO_MODEL_PATH`
- `CANDLE_MODEL_PATH`
- `OV_MAX_LENGTH` (default: `512`)
- `CANDLE_TARGET_DIM`
- `EMBEDDING_STAGE_TIMING`
- `EMBEDDING_STAGE_TIMING_SAMPLES`
- `EMBEDDING_LENGTH_PROFILE`
- `OPENVINO_TOKENIZERS_LIB`
- `OPENVINO_LIB_DIR`

## Tokenizer File Names

OpenVINO exports commonly produce tokenizer files named:

- `openvino_tokenizer.xml`
- `openvino_tokenizer.bin`

Current runtime loader behavior:

1. Prefer `openvino_tokenizer.xml`
2. Fallback to `tokenizer.xml` for backward compatibility

This means both naming styles are supported without extra manual file renaming.

## Model Conversion to OpenVINO IR

OpenVINO requires model IR files:

- `openvino_model.xml`
- `openvino_model.bin`

### Convert from a local model directory

Use a local model path as input:

```bash
optimum-cli export openvino \
  --model models/<your-model> \
  --task text-classification \
  models/<your-model>/openvino \
  --weight-format fp32
```

For token classification models, set task accordingly:

```bash
optimum-cli export openvino \
  --model models/<your-token-model> \
  --task token-classification \
  models/<your-token-model>/openvino \
  --weight-format fp32
```

Note: local conversion requires actual model weights in the local directory (for example `model.safetensors` or `pytorch_model.bin`).

## Precision and Performance

Different numeric types may improve speed, depending on hardware and workload.

- `fp32`: baseline, highest numerical stability
- `fp16`: often reduces memory bandwidth pressure and can improve latency
- `int8`: often offers larger CPU speedups but requires accuracy validation

Recommended tuning workflow:

1. Measure baseline with `fp32`
2. Test `fp16` and compare latency/throughput
3. Evaluate `int8` with representative datasets and verify accuracy before production

## Appendix: Debug (Manual go run)

Use manual mode only for low-level debugging. Script/Make is the supported operational path.

Classifier debug run:

```bash
cd openvino-binding/bench
OPENVINO_MODEL_PATH=/path/to/openvino_model.xml \
CANDLE_MODEL_PATH=/path/to/model_dir \
OPENVINO_TOKENIZERS_LIB=/path/to/libopenvino_tokenizers.so \
LD_LIBRARY_PATH="/path/to/openvino/libs:/path/to/openvino-binding/build:/path/to/candle-binding/target/release:$LD_LIBRARY_PATH" \
go run mmbert_classifier_bench.go
```

Embedding debug run:

```bash
cd openvino-binding/bench
OPENVINO_MODEL_PATH=/path/to/openvino_model.xml \
CANDLE_MODEL_PATH=/path/to/model_dir \
OPENVINO_TOKENIZERS_LIB=/path/to/libopenvino_tokenizers.so \
OV_MAX_LENGTH=512 \
CANDLE_TARGET_DIM=768 \
EMBEDDING_LENGTH_PROFILE=mixed \
LD_LIBRARY_PATH="/path/to/openvino/libs:/path/to/openvino-binding/build:/path/to/candle-binding/target/release:$LD_LIBRARY_PATH" \
go run mmbert_embedding_bench.go
```

## Notes for Customer Deployments

- Keep model and tokenizer files in a consistent per-model directory layout.
- Validate both correctness and performance when changing precision.
- Use repeated runs and fixed CPU/NUMA settings for stable benchmark comparisons.

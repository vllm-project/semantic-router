# OpenVINO binding

This module exposes Intel OpenVINO inference to the Semantic Router through a
C++ library and Go binding. It supports sequence classification, token
classification, and text embeddings. The benchmark helpers compare the same
model with the OpenVINO and Candle backends; they are not published hardware
performance claims.

## Prerequisites

- Go, CMake, and a C++17 compiler
- the OpenVINO runtime and `openvino-tokenizers`
- Python with `transformers` and `optimum[openvino]` when converting models
- `numactl` for the benchmark scripts
- a built Candle library for OpenVINO-versus-Candle comparisons

CMake can discover OpenVINO from its CMake package or from the active Python
environment. If discovery fails, activate the environment that contains
`openvino` and `openvino-tokenizers` before building.

## Build and test

Run the maintained targets from the repository root:

```bash
make build-openvino-binding
make test-openvino-binding
```

The test target converts its fixture models when they are missing, so it needs
network access on the first run. To build the C++ library directly:

```bash
cmake -S openvino-binding -B openvino-binding/build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build openvino-binding/build --parallel
```

The shared library is written to
`openvino-binding/build/libopenvino_semantic_router.so`.

## Model files

An OpenVINO model directory contains:

- `openvino_model.xml` and `openvino_model.bin`
- either `openvino_tokenizer.xml` and `openvino_tokenizer.bin`, or the legacy
  `tokenizer.xml` and `tokenizer.bin` names

The runtime prefers the `openvino_tokenizer.*` names and accepts the legacy
pair for compatibility.

Convert a local sequence-classification model with:

```bash
optimum-cli export openvino \
  --model models/<model-name> \
  --task text-classification \
  --weight-format fp32 \
  models/<model-name>/openvino
```

Use `--task token-classification` for token classifiers and
`--task feature-extraction` for embedding models. The input directory must
contain the original model weights and tokenizer files.

## Benchmarks

The script-backed Make targets build the binding and Candle dependency before
running:

```bash
make benchmark-openvino-classifier
make benchmark-openvino-embedding
```

Use `ARGS` to separate build and run phases or select an embedding length
profile:

```bash
make benchmark-openvino-classifier ARGS='--build-only'
make benchmark-openvino-classifier ARGS='--run-only'
make benchmark-openvino-embedding \
  ARGS='--run-only --length-profile fixed-128'
```

The default model locations are:

| Benchmark | Model root | OpenVINO IR |
| --- | --- | --- |
| Classifier | `models/mmbert-intent-classifier-merged` | `<root>/openvino/openvino_model.xml` |
| Embedding | `models/mmbert-embed-32k-2d-matryoshka` | `<root>/openvino/openvino_model.xml` |

Override them with `CLASSIFIER_MODEL_DIR` for the classifier or
`MMBERT_MODEL_PATH` for embeddings. The lower-level path overrides are
`CLASSIFIER_OPENVINO_MODEL_PATH`, `CLASSIFIER_CANDLE_MODEL_PATH`,
`OPENVINO_MODEL_PATH`, and `CANDLE_MODEL_PATH`.

Embedding runs accept
`--length-profile mixed|fixed-32|fixed-128|fixed-512|fixed-1024|fixed-2048`,
`--stage-timing`, and `--stage-timing-samples N`. `OV_MAX_LENGTH` defaults to
512; inputs beyond that limit are truncated and reported by the benchmark.

## Interpreting results

Keep model revision, precision, CPU topology, thread settings, NUMA binding,
and sample profile fixed when comparing runs. Treat `fp32` as the numerical
baseline, then evaluate `fp16` or `int8` with representative accuracy data
before using a faster precision in production.

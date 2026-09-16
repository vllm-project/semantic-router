# ML Model-Selection Training

This directory builds query-level model selectors from benchmark records. The
pipeline embeds each query, appends an optional domain-category feature, and
exports KNN, KMeans, SVM, or MLP models for the native selection bindings.

Use this pipeline when you have measured the same queries against several
candidate models and want a learned selector. It does not create trustworthy
labels from model names alone.

## Pipeline

1. Benchmark candidate models with `benchmark.py`, or provide existing JSONL.
2. Train one or more selectors with `train.py`.
3. Inspect held-out quality and latency against simple baselines.
4. Validate the exported artifact through the Go/native path before using it in
   a router config.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some validation paths also require the repository's compiled Candle and ML
bindings.

## Collect Benchmark Records

Input is JSONL with at least `query`. `ground_truth` and `category` are
preserved when present.

```json
{"query":"What is the capital of France?","ground_truth":"Paris","category":"other"}
```

Define endpoints in a copy of `models.example.yaml`, then run:

```bash
python benchmark.py \
  --queries queries.jsonl \
  --model-config models.yaml \
  --output benchmark_output.jsonl
```

The benchmarker sends every unique query to every configured model and writes
the response, measured latency, and its available quality score. Check the
scoring method in `benchmark.py` against your task before treating
`performance` as a training label.

## Train Selectors

```bash
python train.py \
  --data-file benchmark_output.jsonl \
  --output-dir models \
  --algorithm all \
  --device cpu
```

`train.py --help` lists algorithm-specific options. The default embedding model
is Qwen3. The exported files are:

| File | Selector |
|---|---|
| `knn_model.json` | quality-weighted nearest neighbours |
| `kmeans_model.json` | cluster-based selection |
| `svm_model.json` | support-vector classification |
| `mlp_model.json` | multilayer perceptron, when PyTorch is available |

The feature vector combines the query embedding with a one-hot category from
`data_loader.py`. Missing or unknown categories use the loader's fallback; use
the exact category strings defined there when preparing data.

## Download or Publish Artifacts

```bash
python download_model.py --output-dir models
python upload_model.py --model-dir models --repo-id ORGANIZATION/REPOSITORY
```

Both commands use `HF_TOKEN` when authentication is required. Review model and
dataset licenses before publishing.

## Optional Training Service

`server.py` exposes the same training pipeline through a local FastAPI service:

```bash
python server.py --host 127.0.0.1 --port 8686
```

Do not expose it to an untrusted network; a training request can consume
substantial compute and write artifacts.

## Validate Before Deployment

`validate.go` exercises exported selectors through the native bindings. Its
downloaded default data is a convenience sample, not a release gate.

```bash
go run validate.go --help
go run validate.go \
  --no-download \
  --data-file benchmark_output.jsonl \
  --models-dir models \
  --algorithm all
```

For a deterministic training/export/native parity check, install `pytest` and
run from the repository root:

```bash
python -m pytest src/training/model_selection/ml_model_selection/tests/test_native_parity.py -q
```

This builds the current Rust binding and compares real sklearn predictions
against the exported artifact through its C ABI. It covers linear/RBF SVM,
binary/multiclass voting, Python reload, KNN latency weighting and neighbor ties.
Only NumPy, scikit-learn, pytest, and Cargo are needed; Torch is optional for
these selectors.

SVM and KNN exports now declare `format_version: 2`. Deploy a runtime that
supports this version alongside the training tools. SVM exports preserve the
original SVC decision functions and input scale. KNN uses normalized feature
distance and the same fixed 10-second latency scale in Python and Rust. Existing
Python SVM files can recover their exact top-level parameters; existing KNN files
adopt the corrected selection rules. See the [native artifact compatibility
reference](../../../../ml-binding/README.md#artifact-compatibility-and-prediction-rules)
for migration details and the treatment of native-only legacy SVM classifiers.

The small checked-in Rust fixtures can be regenerated after an intentional
training-contract change with:

```bash
python src/training/model_selection/ml_model_selection/tests/generate_native_fixtures.py
cargo test --manifest-path ml-binding/Cargo.toml
```

Use a held-out split and report the dataset, candidate models, scoring method,
embedding model, selector parameters, random seed, and quality/latency tradeoff.
Do not copy one local run's output into this README as a general performance
claim.

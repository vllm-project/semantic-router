# ML Model-Selection Training

This directory builds query-level model selectors from benchmark records. The
pipeline embeds each query, appends an optional domain-category feature, and
exports KNN, KMeans, SVM, or MLP models for the router's model selection.

Use this pipeline when you have measured the same queries against several
candidate models and want a learned selector. It does not create trustworthy
labels from model names alone.

## Pipeline

1. Benchmark candidate models with `benchmark.py`, or provide existing JSONL.
2. Train one or more selectors with `train.py`.
3. Inspect held-out quality and latency against simple baselines.
4. Validate the exported artifact through the router's selectors before using it
   in a router config.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Validation builds the router's selectors with Go and embeds queries through
the model runtime (`vllm-sr serve`).

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

If the input has no domain categories, enrich the benchmark records with a
running router before training:

```bash
python add_category_to_training_data.py \
  --vsr-url http://localhost:8080 \
  --input benchmark_output.jsonl \
  --output benchmark_with_category.jsonl
```

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

The service binds to `127.0.0.1` by default. The shipped Kubernetes and OpenShift
sidecars keep this address and use in-container health probes, so only the
co-located Dashboard can reach the API through their shared network namespace.
For Docker, join the Dashboard container network namespace with
`--network container:dashboard`; publishing port 8686 does not make a loopback
listener reachable.

`--host` or `ML_SERVICE_HOST` can override the address for an operator-managed
remote deployment. Such a deployment requires authenticated workload transport
and a restrictive network policy. The service itself has no workload
authentication; do not publish it directly. A training request can consume
substantial compute and write artifacts.

This loopback boundary does not replace server-owned job/artifact handles,
private shared storage, or bounded job lifecycle controls. Those remain separate
requirements before enabling this pipeline for untrusted users.

## Validate Before Deployment

`validate.go` runs the router's own selectors (`pkg/modelselection`) on the
exported artifacts. Query embeddings come from a model runtime serving the
embedding model the selectors were trained with. Its downloaded default data is
a convenience sample, not a release gate.

```bash
vllm-sr serve Qwen/Qwen3-Embedding-0.6B --device cpu --port 8100
make run-ml-selection-validate GO_TOOL_ARGS="--help"
make run-ml-selection-validate GO_TOOL_ARGS="--runtime http://127.0.0.1:8100 \
  --no-download --data-file benchmark_output.jsonl --models-dir models --algorithm all"
```

For a deterministic training/export/router parity check, run from the
repository root with NumPy, scikit-learn, pytest and Go:

```bash
make test-model-selection-parity
```

It trains real sklearn selectors, exports them and replays the same queries
through the router's selectors (the `selectorparity` helper), with the category
one-hot appended as in production. It covers linear/RBF SVM, binary and
multiclass voting, Python reload, unversioned exports, KNN latency weighting and
neighbor ties. Torch, model downloads and GPUs are not needed.

### Artifact compatibility and prediction rules

New Python exports use `format_version: 2`; deploy a router that understands it
together with the training tools.

- SVM stores the exact fitted SVC support vectors, signed dual coefficients,
  intercepts and per-class support counts in `svc`. Linear and RBF inference use
  the training feature scale (`input_normalization: "none"`) and libsvm
  one-vs-one voting, including binary sign conventions and first-class vote
  ties; the input is not normalized again.
- Unversioned Python SVM exports carry those exact parameters at the top level.
  The loader prefers them to the old approximate per-model classifiers, so these
  files need no retraining. A file with only the old classifiers cannot recover
  the fitted SVC; re-export it to adopt version 2.
- KNN uses Euclidean distance on L2-normalized feature vectors (cosine ordering
  for nonzero vectors). Neighbors sort by distance and then sample index; equal
  model totals select the lexicographically first model name. Voting is
  `0.9 * quality + 0.1 / (1 + latency_ns / 10_000_000_000)`.

The loaders reject malformed feature shapes, nonfinite values and inconsistent
sample or support counts; a request whose feature dimension does not match the
artifact fails that selection.

Use a held-out split and report the dataset, candidate models, scoring method,
embedding model, selector parameters, random seed, and quality/latency tradeoff.
Do not copy one local run's output into this README as a general performance
claim.

---
title: ML-Based Model Selection
sidebar_label: ML Model Selection
---

# ML-Based Model Selection

ML-based selection learns which model in a decision's candidate pool is most
appropriate for a request. It is useful when historical evaluation data carries
more information than a fixed priority order or a small set of hand-written
rules.

The selector runs after a routing decision has matched. It can choose only from
the Models assigned to that decision name by the Entrypoint; it does not
discover, deploy, or authenticate backends.

## Available selectors

| Selector | How it chooses | Consider it when |
|----------|----------------|------------------|
| [KNN](/docs/tutorials/algorithm/selection/knn) | Combines recorded quality and speed across similar queries | Similar requests tend to favor the same model and traceability matters |
| [KMeans](/docs/tutorials/algorithm/selection/kmeans) | Maps a request to a learned cluster | The workload forms stable clusters and lookup cost matters |
| [SVM](/docs/tutorials/algorithm/selection/svm) | Uses a learned decision boundary | Candidate models separate cleanly in feature space |
| [MLP](/docs/tutorials/algorithm/selection/mlp) | Scores candidates with a neural network | You have enough data for a non-linear selector and can operate its runtime dependency |

There is no universally best selector. Compare each candidate against simple
baselines such as a fixed default, random choice, and the best single model on
the same held-out dataset.

## Before you train

You need:

- two or more OpenAI-compatible model endpoints
- representative queries and ground-truth answers or another defensible scoring
  method
- enough repeated coverage to evaluate each candidate model across important
  workload slices
- an embedding model that is identical during training and online inference
- a plan for secrets, rate limits, cost, and retention of model responses

Benchmarking sends every selected query to multiple provider endpoints and
stores their responses, quality scores, and latency. Treat the output as
sensitive when prompts or responses contain user data.

## Dashboard workflow

Open `/ml-setup` in the Dashboard to run the guided workflow:

1. Upload a model-endpoint YAML file and a query JSONL file.
2. Benchmark the candidate models.
3. Train one or more selectors.
4. Define decisions and download a configuration fragment.

The Benchmark and Train steps produce data and model artifacts under the
Dashboard's ML data directory.

:::caution Current configuration export

The generated `ml-model-selection-values.yaml` is a routing fragment, not a
complete v0.3 Router manifest. Place selector settings in a Recipe's `routing`
value, create the Provider Models and routing Model cards separately, and
assign their readable names through an Entrypoint. Then run
`vllm-sr validate --config ...` before deployment.

:::

## Command-line workflow

### 1. Install the training dependencies

```bash
cd src/training/model_selection/ml_model_selection
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Prepare evaluation queries

Use one JSON object per line. `ground_truth` is required to score model output;
`category` is optional but useful for sliced evaluation.

```jsonl
{"query":"What is the derivative of x^2?","ground_truth":"2x","category":"math","metric":"MATH"}
{"query":"Which city is the capital of France?","ground_truth":"B","category":"other","metric":"em_mc","choices":"A) London B) Paris C) Berlin D) Rome"}
```

Choose a metric that matches the task. The benchmark supports exact/containment
matching, multiple-choice extraction, GSM8K and MATH answer extraction, text F1,
and code evaluation. Inspect the benchmark output rather than assuming one
metric is suitable for every domain.

### 3. Describe the candidate endpoints

Keep credentials in environment variables. Do not write literal API keys into
the YAML file. This benchmark input is not the Router's `providers.models`
contract; it only names endpoints the training script should evaluate.

```yaml
models:
  - name: local-small
    endpoint: http://localhost:8000/v1
  - name: hosted-model
    endpoint: https://provider.example/v1
    api_key: ${PROVIDER_API_KEY}
```

### 4. Benchmark every candidate

```bash
python benchmark.py \
  --queries queries.jsonl \
  --model-config models.yaml \
  --output benchmark-output.jsonl \
  --concurrency 4
```

Start with low concurrency. Increase it only after confirming that every
endpoint can sustain the request rate and that provider rate limits are not
distorting latency measurements.

### 5. Train selectors

```bash
python train.py \
  --data-file benchmark-output.jsonl \
  --output-dir models
```

By default the script trains KNN, KMeans, SVM, and MLP artifacts. Use
`--algorithm knn|kmeans|svm|mlp` to train one selector, or `--skip-mlp` when the
PyTorch dependency is unavailable. Training accepts `cpu`, `cuda`, or `mps`.
The current Router decision factory runs the loaded MLP artifact on CPU; its
`device` field is accepted for compatibility but is not wired to selection.

The output directory contains JSON artifacts such as `knn_model.json`,
`kmeans_model.json`, `svm_model.json`, and `mlp_model.json`. Their contents are
specific to the benchmarked model names, embedding model, and feature layout.

### 6. Configure the Router

Merge the selector settings into a complete canonical config. This example is a
Recipe fragment; the full manifest still needs Models, an Entrypoint, and any
signals used by the decision. Each ML Decision owns its `algorithm.ml` block.

```yaml
recipes:
  - name: ml-selection
    routing:
      decisions:
        - name: math
          description: Route math requests with the trained KNN selector.
          priority: 100
          rules:
            operator: AND
            conditions:
              - type: domain
                name: math
          algorithm:
            type: knn
            ml:
              models_path: /models/selection
              embedding_dim: 1024
              knn:
                k: 5
                pretrained_path: /models/selection/knn_model.json

entrypoints:
  - model_names: [ml-router]
    recipe: ml-selection
    assignments:
      math:
        models:
          - model: local/small
          - model: hosted/frontier
```

The configured `embedding_dim` and online embedding model must match the
training artifacts. Assigned Model names must match the candidate labels
recorded in the benchmark data.

```bash
vllm-sr validate --config config.yaml
```

## Evaluate before rollout

Use a held-out dataset that was not used to fit or tune the selector. Report:

- answer-quality metric by workload slice
- selected-model distribution
- end-to-end latency and provider cost
- regret relative to an oracle that picks the best evaluated response
- comparison with fixed-default, best-single-model, and random baselines
- failures, timeouts, and excluded samples

Publish the dataset revision, source commit, model revisions, embedding model,
hardware, command, and raw report with any headline result. Percentages or QPS
without that provenance do not describe expected production performance.

## Common problems

### Artifact not found

Check `models_path` and each `pretrained_path` from inside the Router runtime,
not only on the host. Mount or package the files at the same paths used by the
configuration.

### Embedding dimension mismatch

Use the same embedding model and feature layout for training and inference, and
set `embedding_dim` to the exported artifact's dimension.

### Poor held-out quality

Confirm that model names and labels match, inspect class and domain coverage,
look for train/test leakage, and compare against simple baselines. More complex
selectors do not compensate for unrepresentative benchmark data.

## References

- [Training Router Models](./training-overview)
- [Model Performance Evaluation](./model-performance-eval)
- [Training source and complete CLI options](https://github.com/vllm-project/semantic-router/tree/main/src/training/model_selection/ml_model_selection)
- [FusionFactory (arXiv:2507.10540)](https://arxiv.org/abs/2507.10540)
- [Avengers-Pro (arXiv:2508.12631)](https://arxiv.org/abs/2508.12631)

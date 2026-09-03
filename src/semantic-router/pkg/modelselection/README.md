# ML model selectors

This package implements the runtime side of four learned model-selection
algorithms: K-nearest neighbors (`knn`), K-means (`kmeans`), support vector
machines (`svm`), and a multilayer perceptron (`mlp`). A selector runs only
after a routing decision has matched and chooses one of that decision's
`modelRefs`.

These selectors are experimental. Use them when you have representative
query-to-model outcome data and can evaluate the resulting policy against your
own workload. For a policy that does not require a trained artifact, start with
`static`, `router_dc`, `automix`, `multi_factor`, or `latency_aware` instead.

## How selection works

1. The router evaluates signals and matches a decision.
2. It embeds the request and appends the detected domain as a one-hot feature.
3. The configured selector loads its trained artifact and scores the
   decision's candidate models.
4. The selected model name must match a candidate `modelRef`; otherwise the
   request fails instead of silently choosing a different model.

| Algorithm | What it learns | Useful when |
| --- | --- | --- |
| `knn` | Nearby labeled query outcomes | Similar requests tend to prefer the same model. |
| `kmeans` | Query clusters and a model choice per cluster | Workloads contain recurring, separable request groups. |
| `svm` | Boundaries between model assignments | Labeled assignments are available and separate cleanly. |
| `mlp` | A nonlinear mapping from request features to models | The dataset is large enough to justify a learned nonlinear policy. |

## Configure the router

Declare artifact locations in the shared ML settings, then select the
algorithm on a decision. This minimal topology includes the provider aliases
referenced by the decision. Replace the endpoints and artifact path before
using it:

```yaml
version: v0.3

listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899

providers:
  models:
    - name: small-model
      provider_model_id: small-model
      backend_refs:
        - name: small-model-server
          endpoint: small-model:8000
          protocol: http
          weight: 1
    - name: large-model
      provider_model_id: large-model
      backend_refs:
        - name: large-model-server
          endpoint: large-model:8000
          protocol: http
          weight: 1
  defaults:
    default_model: small-model

routing:
  decisions:
    - name: learned-route
      description: Select a candidate with the trained KNN policy.
      priority: 100
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: small-model
        - model: large-model
      algorithm:
        type: knn
  modelCards:
    - name: small-model
    - name: large-model

global:
  router:
    model_selection:
      enabled: true
      ml:
        embedding_dim: 384
        knn:
          k: 5
          pretrained_path: /models/selection/knn_model.json
```

The artifact must be readable inside the router container. Its embedding
dimension, feature layout, and model names must match the runtime config. A
missing or untrained artifact is an error for a multi-model decision.

The exhaustive field reference is
[`config/config.yaml`](../../../../config/config.yaml). The public pages for
[`knn`](../../../../website/docs/tutorials/algorithm/selection/knn.md),
[`kmeans`](../../../../website/docs/tutorials/algorithm/selection/kmeans.md),
[`svm`](../../../../website/docs/tutorials/algorithm/selection/svm.md), and
[`mlp`](../../../../website/docs/tutorials/algorithm/selection/mlp.md) explain
each policy and its limits.

## Train an artifact

Training is kept outside the serving process. The maintained Python pipeline
benchmarks candidate models, converts their quality and latency observations
into labeled examples, and writes artifacts consumed by this package:

```bash
cd src/training/model_selection/ml_model_selection
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

python benchmark.py --help
python train.py --help
```

See the [training README](../../../training/model_selection/ml_model_selection/README.md)
for input formats and end-to-end commands. Do not reuse published benchmark
numbers as acceptance criteria: candidate endpoints, labels, embedding models,
and traffic distributions determine the result.

## Feature compatibility

Runtime and training both use the query embedding followed by a fixed domain
one-hot vector. The domain order is defined by `VSRCategories` in
[`trainer.go`](trainer.go) and must remain identical to the Python data loader.
Changing the embedding model or its dimension requires retraining.

## Test changes

From the router module:

```bash
cd src/semantic-router
go test ./pkg/modelselection ./pkg/selection
```

Unit tests cover artifact loading, feature construction, candidate matching,
and selector behavior. Evaluate routing quality separately on a held-out
workload before deploying a trained policy.

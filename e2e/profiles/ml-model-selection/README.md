# ML model-selection E2E profile

This profile verifies that the Router can load serialized KNN, K-Means, SVM,
and MLP selector artifacts and use them after a domain decision matches. It
deploys mock inference backends and an Envoy AI Gateway path, then runs:

- `model-selection`;
- `domain-classify`;
- `chat-completions-request`.

It is an integration fixture, not a production deployment or a model-quality
benchmark. The tests check runtime wiring and selected-model contracts; they do
not establish that a selector is optimal for another workload.

## External artifacts

Setup expects these files under `.cache/ml-models/`:

```text
knn_model.json
kmeans_model.json
svm_model.json
mlp_model.json
```

If they are absent, the profile installs `huggingface-hub` with the available
Python package manager and downloads
`abdallah1008/semantic-router-ml-models`. The run therefore needs network
access, package-install permission, disk space, and trust in that artifact
source. Pre-populate the cache in a controlled environment when those runtime
downloads are not acceptable.

The artifacts are copied into the Kind environment and mounted at
`/tmp/ml-models`. Paths in [`values.yaml`](values.yaml) must match that mount.

## Run

From the repository root:

```bash
make e2e-test E2E_PROFILE=ml-model-selection
```

Keep the cluster for inspection when model loading fails:

```bash
make e2e-test-debug E2E_PROFILE=ml-model-selection
kubectl logs deployment/semantic-router \
  --namespace vllm-semantic-router-system
```

## Configuration contract

[`values.yaml`](values.yaml) defines domain decisions and per-decision
algorithms. Each ML Decision owns its artifact configuration under
`algorithm.ml`, while the canonical Entrypoint assigns the candidate Models to
every Decision by name. Shared ML settings must agree within one Recipe;
different Recipes may use different artifacts and embedding dimensions.

Keep these pieces aligned:

- decision domain labels;
- feature-vector shape and ordering used during training;
- algorithm type and artifact path;
- model names stored in the artifact;
- canonical Router Model names, Entrypoint assignments, and mock backend names.

An artifact can load successfully and still make meaningless choices if its
feature schema or labels differ from the runtime config.

## Diagnose failures

- **Download fails:** install the Python dependency in advance or populate
  `.cache/ml-models`; verify artifact-source access.
- **Model file not found in the pod:** compare the host cache, Kind mount, and
  `pretrained_path` values.
- **Selector fails to load:** validate JSON shape against the matching training
  exporter and native binding.
- **No decision matched:** compare domain labels exactly; spaces, punctuation,
  and case are part of the configured value.
- **Wrong backend:** compare the artifact's output model name with
  `models[].name`, the decision's Entrypoint assignment, and gateway routes.

Training and evaluation belong in
[`src/training/model_selection/ml_model_selection`](../../../src/training/model_selection/ml_model_selection/README.md),
not in this E2E profile.

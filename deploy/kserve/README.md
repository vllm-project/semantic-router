# OpenShift KServe integration

These assets connect Semantic Router to one or more KServe
`LLMInferenceService` backends on OpenShift. The deployment script discovers
the predictor Service, renders Router and Envoy configuration, and exposes
OpenShift Routes for inference and the Router API.

Use the simulator mode for a CPU-only integration check. Use a real
`LLMInferenceService` only after its model-serving requirements are satisfied.

## Requirements

- an OpenShift project and authenticated `oc` CLI;
- KServe and the `LLMInferenceService` CRD installed;
- storage and image access required by the selected manifests;
- GPU resources for a real GPU model or `--classifier-gpu`.

The helper scripts can install development dependencies, but cluster operators
should review those scripts and use their platform's managed operators where
appropriate.

## Preview the generated resources

Run from the repository root:

```bash
deploy/kserve/deploy.sh --help
deploy/kserve/deploy.sh \
  --namespace semantic-router-demo \
  --simulator \
  --dry-run
```

`--dry-run` is the safest way to inspect the selected ConfigMaps, Deployment,
Services, Route, storage, and permissions before the script applies them.

## CPU simulator

The simulator creates two KServe backends with distinct model names and then
deploys the Router path:

```bash
deploy/kserve/deploy.sh \
  --namespace semantic-router-demo \
  --simulator
```

Use `--classifier-gpu` only when the cluster exposes a schedulable GPU for the
Router classifier. It is independent of the simulator backend mode.

## Existing model service

Confirm the `LLMInferenceService` is ready, then pass its resource name and the
model name returned by its OpenAI-compatible API:

```bash
oc get llminferenceservices --namespace my-project

deploy/kserve/deploy.sh \
  --namespace my-project \
  --inferenceservice granite32-8b \
  --model granite32-8b
```

Optional flags select the storage class, PVC sizes, embedding model, and
classifier device. Keep the `--model` value aligned with Router model cards and
the backend's served-model name.

Example KServe resources are under [`inference-examples/`](inference-examples/).
They are hardware- and platform-sensitive; inspect image, storage, runtime,
resource, and model-access fields before applying them.

## Verify

```bash
oc get pods,services,routes --namespace my-project
oc logs --namespace my-project \
  -l app=semantic-router --all-containers
```

Obtain the route instead of copying a hostname:

```bash
ENVOY_HOST="$(oc get route semantic-router-kserve \
  --namespace my-project -o jsonpath='{.spec.host}')"

curl --fail-with-body "https://$ENVOY_HOST/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "What is 2 + 2?"}],
    "max_tokens": 32
  }'
```

[`test-semantic-routing.sh`](test-semantic-routing.sh) provides the
scenario-specific smoke checks used by this example.

## Diagnose failures

- Backend not discovered: compare the `LLMInferenceService`, predictor Service,
  and namespace selected by `deploy.sh`.
- Router init container fails: inspect model-download logs, credentials, PVC,
  and egress policy.
- No Route response: inspect Route admission, Service endpoints, and Envoy logs.
- Wrong selected model: compare the served-model name, rendered Router config,
  and request model.

```bash
oc describe llminferenceservice <name> --namespace my-project
oc get endpoints --namespace my-project
oc logs --namespace my-project \
  -l app=semantic-router -c model-downloader
```

## Remove the example

The script does not define a universal cleanup policy for pre-existing KServe
models. Delete Router resources with the repository Kustomize package, and
delete only the example model resources you created. Preserve shared
`LLMInferenceService`, operator, storage, and namespace resources unless their
owners approve removal.

For production KServe topology and configuration ownership, see the
[inference platforms guide](../../website/docs/installation/k8s/inference-platforms.md).

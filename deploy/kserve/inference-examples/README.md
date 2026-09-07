# KServe inference examples

These manifests provide small deployment targets for Semantic Router platform
testing on OpenShift AI. They assume the corresponding KServe or
`LLMInferenceService` CRDs and accelerator operators are already installed.

| Files | Use case | Hardware |
| --- | --- | --- |
| `servingruntime-granite32-8b.yaml` + `inferenceservice-granite32-8b.yaml` | Granite 3.2 8B with a dedicated KServe runtime | NVIDIA GPU |
| `inferenceservice-llm-d-sim-model-a.yaml` + `inferenceservice-llm-d-sim-model-b.yaml` | Two lightweight llm-d routing targets backed by `facebook/opt-125m` | CPU-friendly |
| `inferenceservice-qwen-0.6b-gpu.yaml` | Qwen3 0.6B through the alpha `LLMInferenceService` API | NVIDIA GPU |

Apply only the pair or standalone example you intend to use. From this
directory:

```bash
# Granite KServe runtime and model
oc apply -f servingruntime-granite32-8b.yaml
oc apply -f inferenceservice-granite32-8b.yaml

# Or two simulator targets
oc apply -f inferenceservice-llm-d-sim-model-a.yaml
oc apply -f inferenceservice-llm-d-sim-model-b.yaml

# Or the small GPU-backed LLMInferenceService
oc apply -f inferenceservice-qwen-0.6b-gpu.yaml
```

Inspect the Granite predictor URL with:

```bash
oc get inferenceservice granite32-8b \
  -o jsonpath='{.status.components.predictor.address.url}'
```

Use the resulting service address as a provider backend only after the
resource reports ready. Review namespaces, model URIs, images, storage,
accelerator selectors, and resource requests before using these example
manifests outside a test cluster.

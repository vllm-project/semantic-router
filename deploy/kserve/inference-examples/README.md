# KServe Examples

This directory contains example KServe resource configurations for deploying vLLM models on OpenShift AI.

## Files

- `servingruntime-granite32-8b.yaml` - ServingRuntime configuration for vLLM with Granite 3.2 8B
- `inferenceservice-granite32-8b.yaml` - InferenceService to deploy the Granite 3.2 8B model
- `inferenceservice-llm-d-sim-model-a.yaml` - LLMInferenceService simulator (Model-A, CPU-friendly)
- `inferenceservice-llm-d-sim-model-b.yaml` - LLMInferenceService simulator (Model-B, CPU-friendly)
- `inferenceservice-qwen-0.6b-gpu.yaml` - LLMInferenceService for Qwen 0.6B on GPU

## Usage

```bash
# Deploy the ServingRuntime
oc apply -f servingruntime-granite32-8b.yaml

# Deploy the InferenceService
oc apply -f inferenceservice-granite32-8b.yaml

# Or deploy the llm-d simulators (Model-A and Model-B)
oc apply -f inferenceservice-llm-d-sim-model-a.yaml
oc apply -f inferenceservice-llm-d-sim-model-b.yaml

# Get the internal service URL for use in semantic router config
oc get inferenceservice granite32-8b -o jsonpath='{.status.components.predictor.address.url}'
```

These examples can be customized for your specific models and resource requirements. The simulator LLMInferenceServices use a small HF model (opt-125m) to satisfy the LLMISVC `model.uri`.

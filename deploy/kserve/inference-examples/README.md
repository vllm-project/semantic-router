# KServe InferenceService Examples

This directory contains example KServe resource configurations for deploying vLLM models on OpenShift AI.

## Files

- `servingruntime-granite32-8b.yaml` - ServingRuntime configuration for vLLM with Granite 3.2 8B
- `inferenceservice-granite32-8b.yaml` - InferenceService to deploy the Granite 3.2 8B model

## Usage

```bash
# Deploy the ServingRuntime
oc apply -f servingruntime-granite32-8b.yaml

# Deploy the InferenceService
oc apply -f inferenceservice-granite32-8b.yaml

# Get the internal service URL for use in semantic router config
oc get inferenceservice granite32-8b -o jsonpath='{.status.components.predictor.address.url}'
```

These examples can be customized for your specific models and resource requirements.

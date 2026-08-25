---
title: Deploy with Istio Gateway
description: Run Semantic Router as an ExtProc service behind an Istio Gateway and two model backends.
---

# Deploy with Istio Gateway

This guide shows a reference topology for running Semantic Router as an
ExtProc service behind an Istio Gateway. Istio owns ingress and `HTTPRoute`
processing; Semantic Router owns prompt-aware model selection. The supplied
`EnvoyFilter` and `DestinationRule` are specific to this example, so do not
reuse manifests from a different gateway integration without comparing its
ExtProc mode.

## Responsibility split

The deployment consists of:

- **Semantic Router** evaluates routing policy and selects a model.
- **Istio Gateway** accepts client traffic and calls Semantic Router through
  ExtProc.
- **Gateway API resources** map the selected model to a Kubernetes backend.
- **The two vLLM deployments** are example inference backends and require
  suitable compute and model access.

## Prerequisites

You need:

- Kubernetes `1.31`–`1.35`, the supported range for the pinned Istio `1.29`
  release, with at least two schedulable NVIDIA GPUs for the supplied model
  manifests or equivalent capacity for replacement backends;
- [kubectl](https://kubernetes.io/docs/tasks/tools/);
- [Helm](https://helm.sh/docs/intro/install/); and
- [istioctl](https://istio.io/latest/docs/ops/diagnostic-tools/istioctl/).

The supplied manifests request one `nvidia.com/gpu` device for each of two
vLLM Deployments. They pin the vLLM image used by this example. You can replace
them with other OpenAI-compatible backends, but the Router providers, Service
names, and `HTTPRoute` matches must change together.

## Step 1: Verify the cluster

```bash
kubectl wait --for=condition=Ready nodes --all --timeout=300s
```

## Step 2: Deploy LLM models

The example deploys `meta-llama/Llama-3.1-8B-Instruct` and
`microsoft/Phi-4-mini-instruct` with separate vLLM servers. Export a Hugging
Face token before creating the Kubernetes Secret. To use different
OpenAI-compatible backends, update the model names and endpoint references in
both the Router values and the route manifests.

```bash
kubectl create secret generic hf-token-secret --from-literal=token=$HF_TOKEN
```

```bash
# Create vLLM service running llama3-8b
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/istio/vLlama3.yaml
```

The first start downloads model weights and can take several minutes. Deploy
the second backend, then wait for both Deployments.

```bash
# Create vLLM service running phi4-mini
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/istio/vPhi4.yaml
```

```bash
kubectl wait --for=condition=Available deployment/llama-8b --timeout=900s
kubectl wait --for=condition=Available deployment/phi4-mini --timeout=900s
kubectl get pods,services
```

## Step 3: Install Gateway API and Istio

This direct-Service topology needs Kubernetes Gateway API and Istio; it does
not need Gateway API Inference Extension CRDs. Install a compatible pair and
pin the versions in your deployment automation:

```bash
export GATEWAY_API_VERSION=v1.5.1
export ISTIO_VERSION=1.29.6

kubectl apply --server-side \
  -f "https://github.com/kubernetes-sigs/gateway-api/releases/download/${GATEWAY_API_VERSION}/standard-install.yaml"

curl -L https://istio.io/downloadIstio | ISTIO_VERSION="${ISTIO_VERSION}" sh -
export PATH="$PWD/istio-${ISTIO_VERSION}/bin:$PATH"
istioctl install -y --set profile=minimal

kubectl wait --for=condition=Available deployment/istiod \
  -n istio-system --timeout=300s
```

## Step 4: Update vsr config (Optional)

The semantic router configuration is provided via a Helm values file. If you need to customize the configuration (e.g., to match different model names or endpoints), download the values file and modify it:

```bash
# Download the values file for customization
curl -O https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/istio/semantic-router-values/values.yaml
```

Ensure that the models in the config file match the models you are using. It is usually good to start with basic features of vsr such as prompt classification and model routing before experimenting with other features such as PromptGuard or ToolCalling.

## Step 5: Deploy vLLM Semantic Router

Deploy Semantic Router with the integration values:

```bash
# Install semantic router using Helm from GHCR OCI registry
helm install semantic-router oci://ghcr.io/vllm-project/charts/semantic-router \
  --version 0.0.0-latest \
  --namespace vllm-semantic-router-system \
  --create-namespace \
  -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/istio/semantic-router-values/values.yaml

# Wait for deployment to be ready (this may take several minutes for model downloads)
kubectl wait --for=condition=Available deployment/semantic-router -n vllm-semantic-router-system --timeout=600s

# Verify deployment status
kubectl get pods -n vllm-semantic-router-system
```

**Note**: The values file contains provider bindings, signals, decisions, and
routing rules. Download and review
[values.yaml](https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/istio/semantic-router-values/values.yaml)
before adapting it to a real provider pool. Keep
`global.router.clear_route_cache: true`: after ExtProc writes
`x-selected-model`, Envoy must discard its earlier route and evaluate the
header-based `HTTPRoute` again.

## Step 6: Install additional Istio configuration

Install the `DestinationRule` and gateway-scoped `EnvoyFilter` that connect the
Istio gateway to Semantic Router over ExtProc:

```bash
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/istio/destinationrule.yaml
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/istio/envoyfilter.yaml
```

The example filter sets `failure_mode_allow: false`. ExtProc is the inference
authentication, model-access, and quota decision point, so an unavailable
Router fails closed instead of allowing the request to continue without those
checks. It also sends response bodies to ExtProc so backend-authoritative token
usage can be settled; streaming responses switch to streamed processing at
response headers. Test both the outage and settlement paths before production
use.

## Step 7: Install gateway routes

Create the Istio-managed Gateway, then install the two `HTTPRoute` resources.

```bash
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/istio/gateway.yaml
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/istio/httproute-llama3-8b.yaml
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/istio/httproute-phi4-mini.yaml
```

## Step 8: Testing the Deployment

Follow [Test a Kubernetes Gateway Deployment](gateway-testing) to resolve the
actual gateway URL, compare direct and routed model requests, inspect routing
headers, and verify the selected backend. Do not copy the example IP or port;
they are assigned by your cluster.

## Troubleshooting

### Common Issues

**Gateway/ Front end not working:**

```bash
# Check istio gateway status
kubectl get gateway

# Check istio gw service status
kubectl get svc inference-gateway-istio

# Check Istio's Envoy logs
kubectl logs deploy/inference-gateway-istio -c istio-proxy
```

**Semantic router not responding:**

```bash
# Check semantic router pod
kubectl get pods -n vllm-semantic-router-system

# Check semantic router service
kubectl get svc -n vllm-semantic-router-system

# Check semantic router logs
kubectl logs -n vllm-semantic-router-system deployment/semantic-router
```

## Cleanup

To remove the entire deployment:

```bash
# Remove gateway routes
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/istio/httproute-llama3-8b.yaml
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/istio/httproute-phi4-mini.yaml
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/istio/gateway.yaml

# Remove Istio configuration
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/istio/envoyfilter.yaml
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/istio/destinationrule.yaml

# Remove semantic router
helm uninstall semantic-router -n vllm-semantic-router-system

# Remove Istio
istioctl uninstall --purge

# Remove LLMs
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/istio/vLlama3.yaml
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/istio/vPhi4.yaml

```

## Next Steps

- Replace the example models with pinned, production-operated backends.
- Add authentication, network policy, observability, and capacity controls.
- Use the [Gateway API Inference Extension guide](gateway-api-inference-extension)
  when each selected model needs an endpoint picker rather than a direct
  Service backend.

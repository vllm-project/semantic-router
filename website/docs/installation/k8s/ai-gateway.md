---
title: Deploy with Envoy AI Gateway
description: Use Semantic Router for model selection while Envoy AI Gateway owns provider and gateway policy.
---

# Deploy with Envoy AI Gateway

Use this topology when Envoy AI Gateway already owns north-south traffic and
provider integration, while Semantic Router should choose a model from the
request's meaning. Envoy AI Gateway remains responsible for Gateway API
resources, provider credentials, rate limits, and traffic policy. Semantic
Router runs as an ExtProc service and returns the routing decision.

For large request bodies or streamed immediate responses from Semantic Router, also see [Streamed ExtProc and immediate responses](./streamed-extproc). That guide shows how to switch the ExtProc filter from `BUFFERED` to `STREAMED` request bodies and how streamed Chat Completions clients receive looper or `fast_response` immediate responses.

## Responsibility split

The deployment consists of:

- **Semantic Router** evaluates the selected recipe and chooses the logical
  model or provider alias.
- **Envoy Gateway** provides the Kubernetes Gateway API data plane.
- **Envoy AI Gateway** translates provider APIs and applies gateway-owned
  authentication, rate limiting, and traffic policy.
- **Model providers** serve the selected model. This guide uses a demo backend;
  it does not install production inference capacity.

Provider support changes independently of Semantic Router. Use the
[Envoy AI Gateway provider documentation](https://aigateway.envoyproxy.io/docs/capabilities/llm-integrations/supported-providers/)
to choose an `AIServiceBackend` and credential policy, then bind the provider
names to the aliases used by your Semantic Router configuration.

## Prerequisites

You need:

- Kubernetes `1.32` or later for the pinned Envoy AI Gateway `v1.0.x` and
  Envoy Gateway `v1.8.x` compatibility set; [kind](https://kind.sigs.k8s.io/docs/user/quick-start/#installation)
  is sufficient for the demo;
- Gateway API `v1.5.x` CRDs. The default Envoy Gateway Helm installation below
  installs a compatible set; if your platform owns those CRDs, verify its
  versions before installing the chart;
- [kubectl](https://kubernetes.io/docs/tasks/tools/);
- [Helm](https://helm.sh/docs/intro/install/); and
- credentials and network access for every provider used by your config.

## Step 1: Create Kind Cluster (Optional)

Create a local Kubernetes cluster optimized for the semantic router workload:

```bash
kind create cluster --name semantic-router-cluster

# Verify cluster is ready
kubectl wait --for=condition=Ready nodes --all --timeout=300s
```

## Step 2: Deploy vLLM Semantic Router

Deploy Semantic Router with the integration values:

```bash
# Install with custom values from GHCR OCI registry
# (Optional) If you use a registry mirror/proxy, append: --set global.imageRegistry=<your-registry>
helm install semantic-router oci://ghcr.io/vllm-project/charts/semantic-router \
  --version 0.0.0-latest \
  --namespace vllm-semantic-router-system \
  --create-namespace \
  -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/ai-gateway/semantic-router-values/values.yaml

# Wait for deployment to be ready (this may take several minutes for model downloads)
kubectl wait --for=condition=Available deployment/semantic-router -n vllm-semantic-router-system --timeout=600s

# Verify deployment status
kubectl get pods -n vllm-semantic-router-system
```

**Note**: The values file contains the Semantic Router model bindings, signals,
decisions, and routing rules. Download and review
[values.yaml](https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/ai-gateway/semantic-router-values/values.yaml)
before adapting it to a real provider pool.

## Step 3: Install Envoy Gateway

Install the Envoy Gateway release supported by Envoy AI Gateway `v1.0.0`:

```bash
export AIGW_VERSION=v1.0.0
export ENVOY_GATEWAY_VERSION=v1.8.1

helm upgrade -i eg oci://docker.io/envoyproxy/gateway-helm \
  --version "${ENVOY_GATEWAY_VERSION}" \
  --namespace envoy-gateway-system \
  --create-namespace \
  -f "https://raw.githubusercontent.com/envoyproxy/ai-gateway/${AIGW_VERSION}/manifests/envoy-gateway-values.yaml"

kubectl wait --timeout=2m -n envoy-gateway-system deployment/envoy-gateway --for=condition=Available
```

## Step 4: Install Envoy AI Gateway

Install the AI Gateway CRDs before the controller. These versions follow the
upstream [`v1.0.x` compatibility matrix](https://aigateway.envoyproxy.io/docs/compatibility/).

```bash
# Install Envoy AI Gateway CRDs
helm upgrade -i aieg-crd oci://docker.io/envoyproxy/ai-gateway-crds-helm \
  --version "${AIGW_VERSION}" \
  --namespace envoy-ai-gateway-system \
  --create-namespace

# Install the controller
helm upgrade -i aieg oci://docker.io/envoyproxy/ai-gateway-helm \
  --version "${AIGW_VERSION}" \
  --namespace envoy-ai-gateway-system

# Wait for AI Gateway Controller to be ready
kubectl wait --timeout=300s -n envoy-ai-gateway-system deployment/ai-gateway-controller --for=condition=Available
```

## Step 5: Deploy Demo LLM

Create a demo LLM to serve as the backend for the semantic router:

```bash
# Deploy demo LLM
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/ai-gateway/aigw-resources/base-model.yaml
```

## Step 6: Create Gateway API Resources

Create the necessary Gateway API resources for the AI gateway:

```bash
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/ai-gateway/aigw-resources/gwapi-resources.yaml
```

## Testing the Deployment

### Method 1: Port Forwarding (Recommended for Local Testing)

Set up port forwarding to access the gateway locally:

```bash
# Get the Envoy service name
export ENVOY_SERVICE=$(kubectl get svc -n envoy-gateway-system \
  --selector=gateway.envoyproxy.io/owning-gateway-namespace=default,gateway.envoyproxy.io/owning-gateway-name=semantic-router \
  -o jsonpath='{.items[0].metadata.name}')

kubectl port-forward -n envoy-gateway-system svc/$ENVOY_SERVICE 8080:80
```

### Send Test Requests

Once the gateway is accessible, test the inference endpoint:

```bash
# Test math domain chat completions endpoint
curl -i -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vllm-sr/auto",
    "messages": [
      {"role": "user", "content": "What is the derivative of f(x) = x^3?"}
    ]
  }'
```

## Troubleshooting

### Common Issues

**Gateway not accessible:**

```bash
# Check gateway status
kubectl get gateway semantic-router -n default

# Check Envoy service
kubectl get svc -n envoy-gateway-system
```

**AI Gateway controller not ready:**

```bash
# Check AI gateway controller logs
kubectl logs -n envoy-ai-gateway-system deployment/ai-gateway-controller

# Check controller status
kubectl get deployment -n envoy-ai-gateway-system
```

**Semantic router not responding:**

```bash
# Check semantic router pod status
kubectl get pods -n vllm-semantic-router-system

# Check semantic router logs
kubectl logs -n vllm-semantic-router-system deployment/semantic-router
```

## Cleanup

To remove the entire deployment:

```bash
# Remove Gateway API resources and Demo LLM
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/ai-gateway/aigw-resources/gwapi-resources.yaml
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/ai-gateway/aigw-resources/base-model.yaml

# Remove semantic router
helm uninstall semantic-router -n vllm-semantic-router-system

# Remove AI gateway
helm uninstall aieg -n envoy-ai-gateway-system
helm uninstall aieg-crd -n envoy-ai-gateway-system

# Remove Envoy gateway
helm uninstall eg -n envoy-gateway-system

# Delete kind cluster (optional)
kind delete cluster --name semantic-router-cluster
```

## Next Steps

- Replace the demo backend with the provider resources and credentials owned by
  your gateway team.
- Keep the model aliases in `AIGatewayRoute` aligned with the names emitted by
  Semantic Router.
- Add authentication, rate limits, observability, and capacity policy at their
  owning layers before exposing the gateway.

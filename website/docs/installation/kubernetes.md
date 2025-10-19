# Install in Kubernetes

This guide provides step-by-step instructions for deploying the vLLM Semantic Router with Envoy AI Gateway on Kubernetes.

## Architecture Overview

The deployment consists of:

- **vLLM Semantic Router**: Provides intelligent request routing and semantic understanding
- **Envoy Gateway**: Core gateway functionality and traffic management
- **Envoy AI Gateway**: AI Gateway built on Envoy Gateway for LLM providers
- **Gateway API Inference Extension**: CRDs for managing inference pools

## Prerequisites

Before starting, ensure you have the following tools installed:

- [kind](https://kind.sigs.k8s.io/docs/user/quick-start/#installation) - Kubernetes in Docker (Optional)
- [kubectl](https://kubernetes.io/docs/tasks/tools/) - Kubernetes CLI
- [Helm](https://helm.sh/docs/intro/install/) - Package manager for Kubernetes

## Step 1: Create Kind Cluster (Optional)

Create a local Kubernetes cluster optimized for the semantic router workload:

```bash
# Create cluster with optimized resource settings
kind create cluster --name semantic-router-cluster --config tools/kind/kind-config.yaml

# Verify cluster is ready
kubectl wait --for=condition=Ready nodes --all --timeout=300s
```

**Note**: The kind configuration provides sufficient resources (8GB+ RAM, 4+ CPU cores) for running the semantic router and AI gateway components.

## Step 2: Deploy vLLM Semantic Router

Configure the semantic router by editing `deploy/kubernetes/config.yaml`. This file contains the vLLM configuration, including model config, endpoints, and policies.

Deploy the semantic router service with all required components:

```bash
# Deploy semantic router using Kustomize
kubectl apply -k deploy/kubernetes/

# Wait for deployment to be ready (this may take several minutes for model downloads)
kubectl wait --for=condition=Available deployment/semantic-router -n vllm-semantic-router-system --timeout=600s

# Verify deployment status
kubectl get pods -n vllm-semantic-router-system
```

## Step 3: Install Envoy Gateway

Install the core Envoy Gateway for traffic management:

```bash
# Install Envoy Gateway using Helm
helm upgrade -i eg oci://docker.io/envoyproxy/gateway-helm \
    --version v0.0.0-latest \
    --namespace envoy-gateway-system \
    --create-namespace

# Wait for Envoy Gateway to be ready
kubectl wait --timeout=300s -n envoy-gateway-system deployment/envoy-gateway --for=condition=Available
```

## Step 4: Install Envoy AI Gateway

Install the AI-specific extensions for inference workloads:

```bash
# Install Envoy AI Gateway using Helm
helm upgrade -i aieg oci://docker.io/envoyproxy/ai-gateway-helm \
    --version v0.0.0-latest \
    --namespace envoy-ai-gateway-system \
    --create-namespace

# Wait for AI Gateway Controller to be ready
kubectl wait --timeout=300s -n envoy-ai-gateway-system deployment/ai-gateway-controller --for=condition=Available
```

## Step 5: Install Gateway API Inference Extension

Install the Custom Resource Definitions (CRDs) for managing inference pools:

```bash
# Install Gateway API Inference Extension CRDs
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/v1.0.1/manifests.yaml

# Verify CRDs are installed
kubectl get crd | grep inference
```

## Step 6: Configure AI Gateway

Apply the AI Gateway configuration to connect with the semantic router:

```bash
# Apply AI Gateway configuration
kubectl apply -f deploy/kubernetes/ai-gateway/configuration

# Restart controllers to pick up new configuration
kubectl rollout restart -n envoy-gateway-system deployment/envoy-gateway
kubectl rollout restart -n envoy-ai-gateway-system deployment/ai-gateway-controller

# Wait for controllers to be ready
kubectl wait --timeout=120s -n envoy-gateway-system deployment/envoy-gateway --for=condition=Available
kubectl wait --timeout=120s -n envoy-ai-gateway-system deployment/ai-gateway-controller --for=condition=Available
```

## Step 7: Create Inference Pool

Create the inference pool that connects the gateway to the semantic router backend:

```bash
# Create inference pool configuration
kubectl apply -f deploy/kubernetes/ai-gateway/inference-pool

# Wait for inference pool to be ready
sleep 30
```

## Step 8: Verify Deployment

Verify that the inference pool has been created and is properly configured:

```bash
# Check inference pool status
kubectl get inferencepool vllm-semantic-router -n vllm-semantic-router-system -o yaml
```

Expected output should show the inference pool in `Accepted` state:

```yaml
status:
  parent:
  - conditions:
    - lastTransitionTime: "2025-09-27T09:27:32Z"
      message: 'InferencePool has been Accepted by controller ai-gateway-controller:
        InferencePool reconciled successfully'
      observedGeneration: 1
      reason: Accepted
      status: "True"
      type: Accepted
    - lastTransitionTime: "2025-09-27T09:27:32Z"
      message: 'Reference resolution by controller ai-gateway-controller: All references
        resolved successfully'
      observedGeneration: 1
      reason: ResolvedRefs
      status: "True"
      type: ResolvedRefs
    parentRef:
      group: gateway.networking.k8s.io
      kind: Gateway
      name: vllm-semantic-router
      namespace: vllm-semantic-router-system
```

## Testing the Deployment

### Method 1: Port Forwarding (Recommended for Local Testing)

Set up port forwarding to access the gateway locally:

```bash
# Set up environment variables
export GATEWAY_IP="localhost:8080"

# Get the Envoy service name
export ENVOY_SERVICE=$(kubectl get svc -n envoy-gateway-system \
    --selector=gateway.envoyproxy.io/owning-gateway-namespace=vllm-semantic-router-system,gateway.envoyproxy.io/owning-gateway-name=vllm-semantic-router \
    -o jsonpath='{.items[0].metadata.name}')

# Start port forwarding (run in background or separate terminal)
kubectl port-forward -n envoy-gateway-system svc/$ENVOY_SERVICE 8080:80
```

### Method 2: External IP (For Production Deployments)

For production deployments with external load balancers:

```bash
# Get the Gateway external IP
GATEWAY_IP=$(kubectl get gateway vllm-semantic-router -n vllm-semantic-router-system -o jsonpath='{.status.addresses[0].value}')
echo "Gateway IP: $GATEWAY_IP"
```

### Send Test Requests

Once the gateway is accessible, test the inference endpoint:

```bash
# Test math domain chat completions endpoint
curl -i -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MoM",
    "messages": [
      {"role": "user", "content": "What is the derivative of f(x) = x^3 + 2x^2 - 5x + 7?"}
    ]
  }'
```

## Troubleshooting

### Common Issues

**Gateway not accessible:**

```bash
# Check gateway status
kubectl get gateway vllm-semantic-router -n vllm-semantic-router-system

# Check Envoy service
kubectl get svc -n envoy-gateway-system
```

**Inference pool not ready:**

```bash
# Check inference pool events
kubectl describe inferencepool vllm-semantic-router -n vllm-semantic-router-system

# Check AI gateway controller logs
kubectl logs -n envoy-ai-gateway-system deployment/ai-gateway-controller
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
# Remove inference pool
kubectl delete -f deploy/kubernetes/ai-gateway/inference-pool

# Remove AI gateway configuration
kubectl delete -f deploy/kubernetes/ai-gateway/configuration

# Remove semantic router
kubectl delete -k deploy/kubernetes/

# Remove AI gateway
helm uninstall aieg -n envoy-ai-gateway-system

# Remove Envoy gateway
helm uninstall eg -n envoy-gateway-system

# Remove Gateway API CRDs (optional)
kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/v1.0.1/manifests.yaml

# Delete kind cluster
kind delete cluster --name semantic-router-cluster
```

## Next Steps

- Configure custom routing rules in the AI Gateway
- Set up monitoring and observability
- Implement authentication and authorization
- Scale the semantic router deployment for production workloads

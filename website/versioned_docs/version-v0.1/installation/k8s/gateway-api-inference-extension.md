# Install with Gateway API Inference Extension

This guide provides step-by-step instructions for integrating the vLLM Semantic Router (vSR) with Istio and the Kubernetes Gateway API Inference Extension (GIE). This powerful combination allows you to manage self-hosted, OpenAI-compatible models using Kubernetes-native APIs for advanced, load-aware routing.

## Architecture Overview

The deployment consists of three main components:

- **vLLM Semantic Router**: The brain that classifies incoming requests based on their content.
- **Istio & Gateway API**: The network mesh and the front door for all traffic entering the cluster.
- **Gateway API Inference Extension (GIE)**: A set of Kubernetes-native APIs (`InferencePool`, etc.) for managing and scaling self-hosted model backends.

## Benefits of Integration

Integrating vSR with Istio and GIE provides a robust, Kubernetes-native solution for serving LLMs with several key benefits:

### 1. **Kubernetes-Native LLM Management**

Manage your models, routing, and scaling policies directly through `kubectl` using familiar Custom Resource Definitions (CRDs).

### 2. **Intelligent Model and Replica Routing**

Combine vSR's prompt-based model routing with GIE's smart, load-aware replica selection. This ensures requests are sent not only to the right model but also to the healthiest replica, all in a single, efficient hop.

### 3. **Protect Your Models from Overload**

The built-in scheduler tracks GPU load and request queues, automatically shedding traffic to prevent your model servers from crashing under high demand.

### 4. **Deep Observability**

Gain insights from both high-level Gateway metrics and detailed vSR performance data (like token usage and classification accuracy) to monitor and troubleshoot your entire AI stack.

### 5. **Secure Multi-Tenancy**

Isolate tenant workloads using standard Kubernetes namespaces and `HTTPRoutes`. Apply rate limits and other policies while sharing a common, secure gateway infrastructure.

## Supported Backend Models

This architecture is designed to work with any self-hosted model that exposes an **OpenAI-compatible API**. The demo models in this guide use `vLLM` to serve Llama3 and Phi-3, but you can easily replace them with your own model servers.

## Prerequisites

Before starting, ensure you have the following tools installed:

- [Docker](https://docs.docker.com/get-docker/) or another container runtime.
- [kind](https://kind.sigs.k8s.io/) v0.22+ or any Kubernetes 1.29+ cluster.
- [kubectl](https://kubernetes.io/docs/tasks/tools/) v1.30+.
- [Helm](https://helm.sh/) v3.14+.
- [istioctl](https://istio.io/latest/docs/ops/diagnostic-tools/istioctl/) v1.28+.
- A Hugging Face token stored in the `HF_TOKEN` environment variable, required for the sample vLLM deployments to download models.

You can validate your toolchain versions with the following commands:

```bash
kind version
kubectl version --client --short
helm version --short
istioctl version --remote=false
```

## Step 1: Create a Kind Cluster (Optional)

If you don't have a Kubernetes cluster, create a local one for testing:

```bash
kind create cluster --name vsr-gie

# Verify the cluster is ready
kubectl wait --for=condition=Ready nodes --all --timeout=300s
```

## Step 2: Install Istio

Install Istio with support for the Gateway API and external processing:

```bash
# Download and install Istio
export ISTIO_VERSION=1.29.0
curl -L https://istio.io/downloadIstio | ISTIO_VERSION=$ISTIO_VERSION sh -
export PATH="$PWD/istio-$ISTIO_VERSION/bin:$PATH"
istioctl install -y --set profile=minimal --set values.pilot.env.ENABLE_GATEWAY_API=true

# Verify Istio is ready
kubectl wait --for=condition=Available deployment/istiod -n istio-system --timeout=300s
```

## Step 3: Install Gateway API & GIE CRDs

Install the Custom Resource Definitions (CRDs) for the standard Gateway API and the Inference Extension:

```bash
# Install Gateway API CRDs
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.2.0/standard-install.yaml

# Install Gateway API Inference Extension CRDs
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/v1.1.0/manifests.yaml

# Verify CRDs are installed
kubectl get crd | grep 'gateway.networking.k8s.io'
kubectl get crd | grep 'inference.networking.k8s.io'
```

## Step 4: Deploy Demo LLM Servers

Deploy two `vLLM` instances (Llama3 and Phi-3) to act as our backends. These will be automatically downloaded from Hugging Face.

```bash
# Create namespace and secret for the models
kubectl create namespace llm-backends --dry-run=client -o yaml | kubectl apply -f -
kubectl -n llm-backends create secret generic hf-token --from-literal=token=$HF_TOKEN

# Deploy the model servers
kubectl -n llm-backends apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/istio/vLlama3.yaml
kubectl -n llm-backends apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/istio/vPhi4.yaml

# Wait for models to be ready (this may take several minutes)
kubectl -n llm-backends wait --for=condition=Ready pods --all --timeout=10m
```

## Step 5: Deploy vLLM Semantic Router

Deploy the vLLM Semantic Router using its official Helm chart. This component will run as an `ext_proc` server that Istio calls for routing decisions.

```bash
helm upgrade -i semantic-router oci://ghcr.io/vllm-project/charts/semantic-router \
  --version v0.0.0-latest \
  --namespace vllm-semantic-router-system \
  --create-namespace \
  -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/ai-gateway/semantic-router-values/values.yaml

# Wait for the router to be ready
kubectl -n vllm-semantic-router-system wait --for=condition=Available deploy/semantic-router --timeout=10m
```

## Step 6: Deploy Gateway and Routing Logic

Apply the final set of resources to create the public-facing Gateway and wire everything together. This includes the `Gateway`, `InferencePools` for GIE, `HTTPRoutes` for traffic matching, and Istio's `EnvoyFilter`.

```bash
# Apply all routing and gateway resources
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/istio/gateway.yaml
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/inferencepool-llama.yaml
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/inferencepool-phi4.yaml
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/httproute-llama-pool.yaml
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/httproute-phi4-pool.yaml
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/istio/destinationrule.yaml
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/istio/envoyfilter.yaml

# Verify the Gateway is programmed by Istio
kubectl wait --for=condition=Programmed gateway/inference-gateway --timeout=120s
```

## Testing the Deployment

### Method 1: Port Forwarding

Set up port forwarding to access the gateway from your local machine.

```bash
# The Gateway service is named 'inference-gateway-istio' and lives in the default namespace
kubectl port-forward svc/inference-gateway-istio 8080:80
```

### Send Test Requests

Once port forwarding is active, you can send OpenAI-compatible requests to `localhost:8080`.

**Test 1: Explicitly request a model**
This request bypasses the semantic router's logic and goes directly to the specified model pool.

```bash
curl -sS http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "llama3-8b",
    "messages": [{"role": "user", "content": "Summarize the Kubernetes Gateway API in three sentences."}]
  }'
```

**Test 2: Let the Semantic Router choose the model**
By setting `"model": "auto"`, you ask vSR to classify the prompt. It will identify this as a "math" query and add the `x-selected-model: phi4-mini` header, which `HTTPRoute` uses to route the request.

```bash
curl -sS http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "What is 2+2 * (5-1)?"}],
    "max_tokens": 64
  }'
```

## Troubleshooting

**Problem: CRDs are missing**
If you see errors like `no matches for kind "InferencePool"`, check that the CRDs are installed.

```bash
# Check for GIE CRDs
kubectl get crd | grep inference.networking.k8s.io
```

**Problem: Gateway is not ready**
If `kubectl port-forward` fails or requests time out, check the Gateway status.

```bash
# The "Programmed" condition should be "True"
kubectl get gateway inference-gateway -o yaml
```

**Problem: vSR is not being called**
If requests work but routing seems incorrect, check the Istio proxy logs for `ext_proc` errors.

```bash
# Get the Istio gateway pod name
export ISTIO_GW_POD=$(kubectl get pod -l istio=ingressgateway -o jsonpath='{.items[0].metadata.name}')

# Check its logs
kubectl logs $ISTIO_GW_POD -c istio-proxy | grep ext_proc
```

**Problem: Requests are failing**
Check the logs for the vLLM Semantic Router and the backend models.

```bash
# Check vSR logs
kubectl logs deploy/semantic-router -n vllm-semantic-router-system

# Check Llama3 backend logs
kubectl logs -n llm-backends -l app=vllm-llama3-8b-instruct
```

## Cleanup

To remove all the resources created in this guide, run the following commands.

```bash
# 1. Delete all applied Kubernetes resources
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/istio/gateway.yaml
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/inferencepool-llama.yaml
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/inferencepool-phi4.yaml
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/httproute-llama-pool.yaml
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/httproute-phi4-pool.yaml
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/istio/destinationrule.yaml
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/istio/envoyfilter.yaml
kubectl delete ns llm-backends

# 2. Uninstall Helm releases
helm uninstall semantic-router -n vllm-semantic-router-system

# 3. Uninstall Istio
istioctl uninstall -y --purge

# 4. Delete the kind cluster (Optional)
kind delete cluster --name vsr-gie
```

## Next Steps

- **Customize Routing**: Modify the `values.yaml` file for the `semantic-router` Helm chart to define your own routing categories and rules.
- **Add Your Own Models**: Replace the demo Llama3 and Phi-3 deployments with your own OpenAI-compatible model servers.
- **Explore Advanced GIE Features**: Look into using `InferenceObjective` for more advanced autoscaling and scheduling policies.
- **Monitor Performance**: Integrate your Gateway and vSR with Prometheus and Grafana to build monitoring dashboards.

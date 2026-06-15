# Install with vLLM AIBrix

This guide provides step-by-step instructions for integrating the vLLM AIBrix.

## About vLLM AIBrix

[vLLM AIBrix](https://github.com/vllm-project/aibrix) is an open-source initiative designed to provide essential building blocks to construct scalable GenAI inference infrastructure. AIBrix delivers a cloud-native solution optimized for deploying, managing, and scaling large language model (LLM) inference, tailored specifically to enterprise needs.

### Key Features

- **High-Density LoRA Management**: Streamlined support for lightweight, low-rank adaptations of models
- **LLM Gateway and Routing**: Efficiently manage and direct traffic across multiple models and replicas
- **LLM App-Tailored Autoscaler**: Dynamically scale inference resources based on real-time demand
- **Unified AI Runtime**: A versatile sidecar enabling metric standardization, model downloading, and management
- **Distributed Inference**: Scalable architecture to handle large workloads across multiple nodes
- **Distributed KV Cache**: Enables high-capacity, cross-engine KV reuse
- **Cost-efficient Heterogeneous Serving**: Enables mixed GPU inference to reduce costs with SLO guarantees
- **GPU Hardware Failure Detection**: Proactive detection of GPU hardware issues

### Integration Benefits

Integrating vLLM Semantic Router with AIBrix provides several advantages:

1. **Intelligent Request Routing**: Semantic Router analyzes incoming requests and routes them to the most appropriate model based on content understanding, while AIBrix's gateway efficiently manages traffic distribution across model replicas

2. **Enhanced Scalability**: AIBrix's autoscaler works seamlessly with Semantic Router to dynamically adjust resources based on routing patterns and real-time demand

3. **Cost Optimization**: By combining Semantic Router's intelligent routing with AIBrix's heterogeneous serving capabilities, you can optimize GPU utilization and reduce infrastructure costs while maintaining SLO guarantees

4. **Production-Ready Infrastructure**: AIBrix provides enterprise-grade features like distributed KV cache, GPU failure detection, and unified runtime management, making it easier to deploy Semantic Router in production environments

5. **Simplified Operations**: The integration leverages Kubernetes-native patterns and Gateway API resources, providing a familiar operational model for DevOps teams

## Prerequisites

Before starting, ensure you have the following tools installed:

- [kind](https://kind.sigs.k8s.io/docs/user/quick-start/#installation) - Kubernetes in Docker (Optional)
- [kubectl](https://kubernetes.io/docs/tasks/tools/) - Kubernetes CLI
- [Helm](https://helm.sh/docs/intro/install/) - Package manager for Kubernetes

## Step 1: Create Kind Cluster (Optional)

Create a local Kubernetes cluster optimized for the semantic router workload:

```bash
kind create cluster --name semantic-router-cluster

# Verify cluster is ready
kubectl wait --for=condition=Ready nodes --all --timeout=300s
```

## Step 2: Deploy vLLM Semantic Router

Deploy the semantic router service with all required components using Helm:

```bash
# Install with custom values from GHCR OCI registry
# (Optional) If you use a registry mirror/proxy, append: --set global.imageRegistry=<your-registry>
helm install semantic-router oci://ghcr.io/vllm-project/charts/semantic-router \
  --version v0.0.0-latest \
  --namespace vllm-semantic-router-system \
  --create-namespace \
  -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/aibrix/semantic-router-values/values.yaml

# Wait for deployment to be ready (this may take several minutes for model downloads)
kubectl wait --for=condition=Available deployment/semantic-router -n vllm-semantic-router-system --timeout=600s

# Verify deployment status
kubectl get pods -n vllm-semantic-router-system
```

**Note**: The values file contains the configuration for the semantic router, including model settings, categories, and routing rules. You can download and customize it from [values.yaml](https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/aibrix/semantic-router-values/values.yaml).

## Step 3: Install vLLM AIBrix

Install the core vLLM AIBrix components:

```bash
# Install vLLM AIBrix
kubectl create -f https://github.com/vllm-project/aibrix/releases/download/v0.4.1/aibrix-dependency-v0.4.1.yaml

kubectl create -f https://github.com/vllm-project/aibrix/releases/download/v0.4.1/aibrix-core-v0.4.1.yaml

# wait for deployment to be ready
kubectl wait --timeout=2m -n aibrix-system deployment/aibrix-gateway-plugins --for=condition=Available
kubectl wait --timeout=2m -n aibrix-system deployment/aibrix-metadata-service --for=condition=Available
kubectl wait --timeout=2m -n aibrix-system deployment/aibrix-controller-manager --for=condition=Available
```

## Step 4: Deploy Demo LLM

Create a demo LLM to serve as the backend for the semantic router:

```bash
# Deploy demo LLM
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/aibrix/aigw-resources/base-model.yaml

kubectl wait --timeout=2m -n default deployment/vllm-llama3-8b-instruct --for=condition=Available
```

## Step 5: Create Gateway API Resources

Create the necessary Gateway API resources for the envoy gateway:

```bash
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/aibrix/aigw-resources/gwapi-resources.yaml
```

## Testing the Deployment

### Method 1: Port Forwarding (Recommended for Local Testing)

Set up port forwarding to access the gateway locally:

```bash
# Get the Envoy service name
export ENVOY_SERVICE=$(kubectl get svc -n envoy-gateway-system \
  --selector=gateway.envoyproxy.io/owning-gateway-namespace=aibrix-system,gateway.envoyproxy.io/owning-gateway-name=aibrix-eg \
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
    "model": "MoM",
    "messages": [
      {"role": "user", "content": "What is the derivative of f(x) = x^3?"}
    ]
  }'
```

You will see the response from the demo LLM, and additional headers injected by the semantic router.

```bash
HTTP/1.1 200 OK
server: fasthttp
date: Thu, 06 Nov 2025 06:38:08 GMT
content-type: application/json
x-inference-pod: vllm-llama3-8b-instruct-984659dbb-gp5l9
x-went-into-req-headers: true
request-id: b46b6f7b-5645-470f-9868-0dd8b99a7163
x-vsr-selected-category: math
x-vsr-selected-reasoning: on
x-vsr-selected-model: vllm-llama3-8b-instruct
x-vsr-injected-system-prompt: true
transfer-encoding: chunked

{"id":"chatcmpl-f390a0c6-b38f-4a73-b019-9374a3c5d69b","created":1762411088,"model":"vllm-llama3-8b-instruct","usage":{"prompt_tokens":42,"completion_tokens":48,"total_tokens":90},"object":"chat.completion","do_remote_decode":false,"do_remote_prefill":false,"remote_block_ids":null,"remote_engine_id":"","remote_host":"","remote_port":0,"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"I am your AI assistant, how can I help you today? To be or not to be that is the question. Alas, poor Yorick! I knew him, Horatio: A fellow of infinite jest Testing, testing 1,2,3"}}]}
```

## Cleanup

To remove the entire deployment:

```bash
# Remove Gateway API resources and Demo LLM
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/aibrix/aigw-resources/gwapi-resources.yaml
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/aibrix/aigw-resources/base-model.yaml

# Remove semantic router
helm uninstall semantic-router -n vllm-semantic-router-system

# Delete kind cluster (optional)
kind delete cluster --name semantic-router-cluster
```

## Next Steps

- Set up monitoring and observability
- Implement authentication and authorization
- Scale the semantic router deployment for production workloads

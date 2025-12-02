# Install with vLLM Production Stack

This tutorial is adapted from [vLLM production stack tutorials](https://github.com/vllm-project/production-stack/blob/main/tutorials/24-semantic-router-integration.md)

## What is vLLM Semantic Router?

The vLLM Semantic Router is an intelligent Mixture-of-Models (MoM) router that operates as an Envoy External Processor to semantically route OpenAI API–compatible requests to the most suitable backend model. Using BERT-based classification, it improves both quality and cost efficiency by matching requests (e.g., math, code, creative, general) to specialized models.

- **Auto-selection of models**: Routes math, creative writing, code, and general queries to the best-fit models.
- **Security & privacy**: PII detection, prompt guard, and safe routing for sensitive prompts.
- **Performance optimizations**: Semantic cache and better tool selection to cut latency and tokens.
- **Architecture**: Tight Envoy ExtProc integration; dual Go and Python implementations; production-ready and scalable.
- **Monitoring**: Grafana dashboards, Prometheus metrics, and tracing for full visibility.

Learn more: [vLLM Semantic Router](https://vllm-semantic-router.com/docs/intro)

## What are the benefits of integration?

The vLLM Production Stack provides several deployment ways that spin up vLLM servers which can direct traffic to different models, perform service discovery and fault tolerance through the Kubernetes API, and support round‑robin, session‑based, prefix‑aware, KV-aware and disaggregated-prefill routing with LMCache native support. The Semantic Router adds a system‑intelligence layer that classifies each user request, selects the most suitable model from a pool, injects domain‑specific system prompts, performs semantic caching and enforces enterprise‑grade security checks such as PII and jailbreak detection.

By combining these two systems we obtain a unified inference stack. Semantic routing ensures that each request is answered by the best possible model. Production‑Stack routing maximizes infrastructure and inference efficiency, and exposes rich metrics.

---

This tutorial will guide you:

- Deploy a minimal vLLM Production Stack
- Deploy vLLM Semantic Router and point it to your vLLM router Service
- Test the endpoint via the Envoy AI Gateway

## Prerequisites

- kubectl
- Helm
- A Kubernetes cluster (kind, minikube, GKE, etc.)

---

## Step 1: Deploy the vLLM Production Stack using your Helm values

Use your chart and the provided values file at `tutorials/assets/values-23-SR.yaml`.

```bash
helm repo add vllm-production-stack https://vllm-project.github.io/production-stack
helm install vllm-stack vllm-production-stack/vllm-stack -f ./tutorials/assets/values-23-SR.yaml
```

For reference, the following is the sample value file:

```yaml
servingEngineSpec:
  runtimeClassName: ""
  strategy:
    type: Recreate
  modelSpec:
  - name: "qwen3"
    repository: "lmcache/vllm-openai"
    tag: "v0.3.7"
    modelURL: "Qwen/Qwen3-8B"
    pvcStorage: "50Gi"
    vllmConfig:
      # maxModelLen: 131072
      extraArgs: ["--served-model-name", "Qwen/Qwen3-8B", "qwen3"]

    replicaCount: 2

    requestCPU: 8
    requestMemory: "16Gi"
    requestGPU: 1

routerSpec:
  repository: lmcache/lmstack-router
  tag: "latest"
  resources:
    requests:
      cpu: "1"
      memory: "2G"
    limits:
      cpu: "1"
      memory: "2G"
  routingLogic: "roundrobin"
  sessionKey: "x-user-id"
```

Identify the ClusterIP and port of your router Service created by the chart (name may vary):

```bash
kubectl get svc vllm-router-service
# Note the router service ClusterIP and port (e.g., 10.97.254.122:80)
```

---

## Step 2: Deploy vLLM Semantic Router and point it at your vLLM router Service

Follow the official guide from the official website with **the updated config file as the following**: [Install in Kubernetes](https://vllm-semantic-router.com/docs/installation/k8s/ai-gateway).

Deploy using Helm with custom values:

```bash
   # Deploy vLLM Semantic Router with custom values from GHCR OCI registry
   # (Optional) If you use a registry mirror/proxy, append: --set global.imageRegistry=<your-registry>
   helm install semantic-router oci://ghcr.io/vllm-project/charts/semantic-router \
     --version v0.0.0-latest \
     --namespace vllm-semantic-router-system \
     --create-namespace \
     -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/ai-gateway/semantic-router-values/values.yaml

   kubectl wait --for=condition=Available deployment/semantic-router \
     -n vllm-semantic-router-system --timeout=600s

   # Install Envoy Gateway
   helm upgrade -i eg oci://docker.io/envoyproxy/gateway-helm \
     --version v0.0.0-latest \
     --namespace envoy-gateway-system \
     --create-namespace \
     -f https://raw.githubusercontent.com/envoyproxy/ai-gateway/main/manifests/envoy-gateway-values.yaml

   # Install Envoy AI Gateway
   helm upgrade -i aieg oci://docker.io/envoyproxy/ai-gateway-helm \
     --version v0.0.0-latest \
     --namespace envoy-ai-gateway-system \
     --create-namespace

   # Install Envoy AI Gateway CRDs
   helm upgrade -i aieg-crd oci://docker.io/envoyproxy/ai-gateway-crds-helm \
     --version v0.0.0-latest \
     --namespace envoy-ai-gateway-system

   kubectl wait --timeout=300s -n envoy-ai-gateway-system \
     deployment/ai-gateway-controller --for=condition=Available
```

**Note**: The values file contains the configuration for the semantic router. You can download and customize it from [values.yaml](https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/ai-gateway/semantic-router-values/values.yaml) to match your vLLM Production Stack setup.

Create LLM Demo Backends and AI Gateway Routes:

```bash
   # Apply LLM demo backends
   kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/ai-gateway/aigw-resources/base-model.yaml
   # Apply AI Gateway routes
   kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/ai-gateway/aigw-resources/gwapi-resources.yaml
```

---

## Step 3: Test the deployment

Port-forward to the Envoy service and send a test request, following the guide:

```bash
  export ENVOY_SERVICE=$(kubectl get svc -n envoy-gateway-system \
    --selector=gateway.envoyproxy.io/owning-gateway-namespace=default,gateway.envoyproxy.io/owning-gateway-name=semantic-router \
    -o jsonpath='{.items[0].metadata.name}')

  kubectl port-forward -n envoy-gateway-system svc/$ENVOY_SERVICE 8080:80
```

Send a chat completions request:

```bash
  curl -i -X POST http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "MoM",
      "messages": [
        {"role": "user", "content": "What is the derivative of f(x) = x^3?"}
      ]
    }'
```

---

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

# Remove vLLM Production Stack
helm uninstall vllm-stack

# Delete kind cluster (optional)
kind delete cluster --name semantic-router-cluster
```

---

## Troubleshooting

- If the gateway is not accessible, check the Gateway and Envoy service per the guide.
- If the inference pool is not ready, `kubectl describe` the `InferencePool` and check controller logs.
- If the semantic router is not responding, check its pod status and logs.
- If it is returning error code, check the production stack router log.

# Integration with vLLM Semantic Router

- [Integration with vLLM Semantic Router](#integration-with-vllm-semantic-router)
  - [What is vLLM Semantic Router?](#what-is-vllm-semantic-router)
  - [What are the benefits of integration?](#what-are-the-benefits-of-integration)
  - [Prerequisites](#prerequisites)
  - [Step 1: Deploy the vLLM Production Stack using your Helm values](#step-1-deploy-the-vllm-production-stack-using-your-helm-values)
  - [Step 2: Deploy vLLM Semantic Router and point it at your vLLM router Service](#step-2-deploy-vllm-semantic-router-and-point-it-at-your-vllm-router-service)
  - [Step 3: Test the deployment](#step-3-test-the-deployment)
  - [Troubleshooting](#troubleshooting)

> This tutorial is adapted from [vLLM production stack tutorials](https://github.com/vllm-project/production-stack/blob/main/tutorials/24-semantic-router-integration.md)

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

Remember to update the semantic router config to include your vLLM router service as an endpoint. Edit `deploy/kubernetes/config.yaml` and set `vllm_endpoints` like this (replace the IP/port with your router Service ClusterIP/port from step 1):

```yaml
vllm_endpoints:
  - name: "endpoint1"
    address: <YOUR ROUTER SERVICE CLUSTERIP>
    port: <YOUR ROUTER SERVICE PORT>
    weight: 1
```

Minimal sequence (same as the guide):

```bash
# Deploy vLLM Semantic Router manifests
kubectl apply -k deploy/kubernetes/
kubectl wait --for=condition=Available deployment/semantic-router \
  -n vllm-semantic-router-system --timeout=600s

# Install Envoy Gateway
helm upgrade -i eg oci://docker.io/envoyproxy/gateway-helm \
  --version v0.0.0-latest \
  --namespace envoy-gateway-system \
  --create-namespace
kubectl wait --timeout=300s -n envoy-gateway-system \
  deployment/envoy-gateway --for=condition=Available

# Install Envoy AI Gateway
helm upgrade -i aieg oci://docker.io/envoyproxy/ai-gateway-helm \
  --version v0.0.0-latest \
  --namespace envoy-ai-gateway-system \
  --create-namespace
kubectl wait --timeout=300s -n envoy-ai-gateway-system \
  deployment/ai-gateway-controller --for=condition=Available

# Install Gateway API Inference Extension CRDs
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/v1.0.1/manifests.yaml
kubectl get crd | grep inference
```

Apply AI Gateway configuration and create the inference pool per the guide:

```bash
# Apply AI Gateway configuration
kubectl apply -f deploy/kubernetes/ai-gateway/configuration

# Restart controllers to pick up new config
kubectl rollout restart -n envoy-gateway-system deployment/envoy-gateway
kubectl rollout restart -n envoy-ai-gateway-system deployment/ai-gateway-controller
kubectl wait --timeout=120s -n envoy-gateway-system deployment/envoy-gateway --for=condition=Available
kubectl wait --timeout=120s -n envoy-ai-gateway-system deployment/ai-gateway-controller --for=condition=Available

# Create inference pool
kubectl apply -f deploy/kubernetes/ai-gateway/inference-pool
sleep 30

# Verify inference pool
kubectl get inferencepool vllm-semantic-router -n vllm-semantic-router-system -o yaml
```

---

## Step 3: Test the deployment

Port-forward to the Envoy service and send a test request, following the guide:

```bash
export GATEWAY_IP="localhost:8080"
export ENVOY_SERVICE=$(kubectl get svc -n envoy-gateway-system \
  --selector=gateway.envoyproxy.io/owning-gateway-namespace=vllm-semantic-router-system,gateway.envoyproxy.io/owning-gateway-name=vllm-semantic-router \
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
      {"role": "user", "content": "What is the derivative of f(x) = x^3 + 2x^2 - 5x + 7?"}
    ]
  }'
```

---

## Troubleshooting

- If the gateway is not accessible, check the Gateway and Envoy service per the guide.
- If the inference pool is not ready, `kubectl describe` the `InferencePool` and check controller logs.
- If the semantic router is not responding, check its pod status and logs.
- If it is returning error code, check the production stack router log.

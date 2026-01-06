# Install with Envoy AI Gateway

This guide provides step-by-step instructions for integrating the vLLM Semantic Router with Envoy AI Gateway on Kubernetes for advanced traffic management and AI-specific features.

## Architecture Overview

The deployment consists of:

- **vLLM Semantic Router**: Provides intelligent request routing and semantic understanding
- **Envoy Gateway**: Core gateway functionality and traffic management
- **Envoy AI Gateway**: AI Gateway built on Envoy Gateway for LLM providers

## Benefits of Integration

Integrating vLLM Semantic Router with Envoy AI Gateway provides enterprise-grade capabilities for production LLM deployments:

### 1. **Hybrid Model Selection**

Seamlessly route requests between cloud LLM providers (OpenAI, Anthropic, etc.) and self-hosted models.

### 2. **Token Rate Limiting**

Protect your infrastructure and control costs with fine-grained rate limiting:

- **Input token limits**: Control request size to prevent abuse
- **Output token limits**: Manage response generation costs
- **Total token limits**: Set overall usage quotas per user/tenant
- **Time-based windows**: Configure limits per second, minute, or hour

### 3. **Model/Provider Failover**

Ensure high availability with automatic failover mechanisms:

- Detect unhealthy backends and route traffic to healthy instances
- Support for active-passive and active-active failover strategies
- Graceful degradation when primary models are unavailable

### 4. **Traffic Splitting & Canary Testing**

Deploy new models safely with progressive rollout capabilities:

- **A/B Testing**: Split traffic between model versions to compare performance
- **Canary Deployments**: Gradually shift traffic to new models (e.g., 5% → 25% → 50% → 100%)
- **Shadow Traffic**: Send duplicate requests to new models without affecting production
- **Weight-based routing**: Fine-tune traffic distribution across model variants

### 5. **LLM Observability & Monitoring**

Gain deep insights into your LLM infrastructure:

- **Request/Response Metrics**: Track latency, throughput, token usage, and error rates
- **Model Performance**: Monitor accuracy, quality scores, and user satisfaction
- **Cost Analytics**: Analyze spending patterns across models and providers
- **Distributed Tracing**: End-to-end visibility with OpenTelemetry integration
- **Custom Dashboards**: Visualize metrics in Prometheus, Grafana, or your preferred monitoring stack

## Supported LLM Providers

| Provider Name                                                | API Schema Config on [AIServiceBackend](https://aigateway.envoyproxy.io/docs/api/#aiservicebackendspec) | Upstream Authentication Config on [BackendSecurityPolicy](https://aigateway.envoyproxy.io/docs/api/#backendsecuritypolicyspec) | Status |
| ------------------------------------------------------------ | :----------------------------------------------------------: | :----------------------------------------------------------: | :----: |
| [OpenAI](https://platform.openai.com/docs/api-reference)     |              `{"name":"OpenAI","version":"v1"}`              | [API Key](https://aigateway.envoyproxy.io/docs/api/#backendsecuritypolicyapikey) |   ✅    |
| [AWS Bedrock](https://docs.aws.amazon.com/bedrock/latest/APIReference/) |                   `{"name":"AWSBedrock"}`                    | [AWS Bedrock Credentials](https://aigateway.envoyproxy.io/docs/api/#backendsecuritypolicyawscredentials) |   ✅    |
| [Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-services/openai/reference) | `{"name":"AzureOpenAI","version":"2025-01-01-preview"}` or `{"name":"OpenAI", "version": "openai/v1"}` | [Azure Credentials](https://aigateway.envoyproxy.io/docs/api/#backendsecuritypolicyazurecredentials) or [Azure API Key](https://aigateway.envoyproxy.io/docs/api/#backendsecuritypolicyazureapikey) |   ✅    |
| [Google Gemini on AI Studio](https://ai.google.dev/gemini-api/docs/openai) |        `{"name":"OpenAI","version":"v1beta/openai"}`         | [API Key](https://aigateway.envoyproxy.io/docs/api/#backendsecuritypolicyapikey) |   ✅    |
| [Google Vertex AI](https://cloud.google.com/vertex-ai/docs/reference/rest) |                   `{"name":"GCPVertexAI"}`                   | [GCP Credentials](https://aigateway.envoyproxy.io/docs/api/#backendsecuritypolicygcpcredentials) |   ✅    |
| [Anthropic on GCP Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude) |   `{"name":"GCPAnthropic", "version":"vertex-2023-10-16"}`   | [GCP Credentials](https://aigateway.envoyproxy.io/docs/api/#backendsecuritypolicygcpcredentials) |   ✅    |
| [Groq](https://console.groq.com/docs/openai)                 |          `{"name":"OpenAI","version":"openai/v1"}`           | [API Key](https://aigateway.envoyproxy.io/docs/api/#backendsecuritypolicyapikey) |   ✅    |
| [Grok](https://docs.x.ai/docs/api-reference?utm_source=chatgpt.com#chat-completions) |              `{"name":"OpenAI","version":"v1"}`              | [API Key](https://aigateway.envoyproxy.io/docs/api/#backendsecuritypolicyapikey) |   ✅    |
| [Together AI](https://docs.together.ai/docs/openai-api-compatibility) |              `{"name":"OpenAI","version":"v1"}`              | [API Key](https://aigateway.envoyproxy.io/docs/api/#backendsecuritypolicyapikey) |   ✅    |
| [Cohere](https://docs.cohere.com/v2/docs/compatibility-api)  | `{"name":"Cohere","version":"v2"}` or `{"name":"OpenAI","version":"v1"}` | [API Key](https://aigateway.envoyproxy.io/docs/api/#backendsecuritypolicyapikey) |   ✅    |
| [Mistral](https://docs.mistral.ai/api/#tag/chat/operation/chat_completion_v1_chat_completions_post) |              `{"name":"OpenAI","version":"v1"}`              | [API Key](https://aigateway.envoyproxy.io/docs/api/#backendsecuritypolicyapikey) |   ✅    |
| [DeepInfra](https://deepinfra.com/docs/inference)            |          `{"name":"OpenAI","version":"v1/openai"}`           | [API Key](https://aigateway.envoyproxy.io/docs/api/#backendsecuritypolicyapikey) |   ✅    |
| [DeepSeek](https://api-docs.deepseek.com/)                   |              `{"name":"OpenAI","version":"v1"}`              | [API Key](https://aigateway.envoyproxy.io/docs/api/#backendsecuritypolicyapikey) |   ✅    |
| [Hunyuan](https://cloud.tencent.com/document/product/1729/111007) |              `{"name":"OpenAI","version":"v1"}`              | [API Key](https://aigateway.envoyproxy.io/docs/api/#backendsecuritypolicyapikey) |   ✅    |
| [Tencent LLM Knowledge Engine](https://www.tencentcloud.com/document/product/1255/70381?lang=en) |              `{"name":"OpenAI","version":"v1"}`              | [API Key](https://aigateway.envoyproxy.io/docs/api/#backendsecuritypolicyapikey) |   ✅    |
| [Tetrate Agent Router Service (TARS)](https://router.tetrate.ai/) |              `{"name":"OpenAI","version":"v1"}`              | [API Key](https://aigateway.envoyproxy.io/docs/api/#backendsecuritypolicyapikey) |   ✅    |
| [SambaNova](https://docs.sambanova.ai/sambastudio/latest/open-ai-api.html) |              `{"name":"OpenAI","version":"v1"}`              | [API Key](https://aigateway.envoyproxy.io/docs/api/#backendsecuritypolicyapikey) |   ✅    |
| [Anthropic](https://docs.claude.com/en/home)                 |                    `{"name":"Anthropic"}`                    | [Anthropic API Key](https://aigateway.envoyproxy.io/docs/api/#backendsecuritypolicyanthropicapikey) |   ✅    |
| Self-hosted-models                                           |              `{"name":"OpenAI","version":"v1"}`              |                             N/A                              |   ✅    |

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
  -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/ai-gateway/semantic-router-values/values.yaml

# Wait for deployment to be ready (this may take several minutes for model downloads)
kubectl wait --for=condition=Available deployment/semantic-router -n vllm-semantic-router-system --timeout=600s

# Verify deployment status
kubectl get pods -n vllm-semantic-router-system
```

**Note**: The values file contains the configuration for the semantic router, including model settings, categories, and routing rules. You can download and customize it from [values.yaml](https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/ai-gateway/semantic-router-values/values.yaml).

## Step 3: Install Envoy Gateway

Install the core Envoy Gateway for traffic management:

```bash
# Install Envoy Gateway using Helm
helm upgrade -i eg oci://docker.io/envoyproxy/gateway-helm \
  --version v0.0.0-latest \
  --namespace envoy-gateway-system \
  --create-namespace \
  -f https://raw.githubusercontent.com/envoyproxy/ai-gateway/main/manifests/envoy-gateway-values.yaml

kubectl wait --timeout=2m -n envoy-gateway-system deployment/envoy-gateway --for=condition=Available
```

## Step 4: Install Envoy AI Gateway

Install the AI-specific extensions for inference workloads:

```bash
# Install Envoy AI Gateway using Helm
helm upgrade -i aieg oci://docker.io/envoyproxy/ai-gateway-helm \
    --version v0.0.0-latest \
    --namespace envoy-ai-gateway-system \
    --create-namespace

# Install Envoy AI Gateway CRDs
helm upgrade -i aieg-crd oci://docker.io/envoyproxy/ai-gateway-crds-helm --version v0.0.0-latest --namespace envoy-ai-gateway-system

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
    "model": "MoM",
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

- Configure custom routing rules in the AI Gateway
- Set up monitoring and observability
- Implement authentication and authorization
- Scale the semantic router deployment for production workloads

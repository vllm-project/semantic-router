# Semantic Router Integration with OpenShift AI KServe

This directory contains Kubernetes manifests for deploying the vLLM Semantic Router to work with OpenShift AI's KServe InferenceService endpoints.

## Overview

The semantic router acts as an intelligent gateway that routes OpenAI-compatible API requests to appropriate vLLM models deployed via KServe InferenceServices. It provides:

- **Intelligent Model Selection**: Automatically routes requests to the best model based on semantic understanding
- **PII Detection & Protection**: Blocks or redacts sensitive information
- **Prompt Guard**: Detects and blocks jailbreak attempts
- **Semantic Caching**: Reduces latency and costs through intelligent caching
- **Category-Specific Prompts**: Injects domain-specific system prompts
- **Tools Auto-Selection**: Automatically selects relevant tools for function calling

## Architecture

```
Client Request (OpenAI API)
    ↓
[OpenShift Route - HTTPS]
    ↓
[Envoy Proxy Container] ← [Semantic Router Container]
    ↓                              ↓
    |                     [Classification & Selection]
    |                              ↓
    |                     [Sets x-gateway-destination-endpoint]
    ↓
[KServe InferenceService Predictor]
    ↓
[vLLM Model Response]
```

The deployment runs two containers in a single pod:
1. **Semantic Router**: ExtProc service that performs classification and routing logic
2. **Envoy Proxy**: HTTP proxy that integrates with the semantic router via gRPC

## Prerequisites

1. **OpenShift Cluster** with OpenShift AI (RHOAI) installed
2. **KServe InferenceServices** deployed in your namespace (see `inference-examples/` for sample configurations)
3. **Storage Class** available for PersistentVolumeClaims
4. **Namespace** where you want to deploy

### Verify Your InferenceServices

Check your deployed InferenceServices:

```bash
oc get inferenceservice
```

Example output:
```
NAME           URL                                        READY   PREV   LATEST
granite32-8b   https://granite32-8b-your-ns.apps...      True           100
```

Get the internal service URL for the predictor:

```bash
oc get inferenceservice granite32-8b -o jsonpath='{.status.components.predictor.address.url}'
```

Example output:
```
http://granite32-8b-predictor.your-namespace.svc.cluster.local
```

## Configuration

### Step 1: Configure InferenceService Endpoints

Edit `configmap-router-config.yaml` to add your InferenceService endpoints:

```yaml
vllm_endpoints:
  - name: "your-model-endpoint"
    address: "your-model-predictor.<namespace>.svc.cluster.local"  # Replace with your model and namespace
    port: 80  # KServe uses port 80 for internal service
    weight: 1
```

**Important**:
- Replace `<namespace>` with your actual namespace
- Replace `your-model` with your InferenceService name
- Use the **internal cluster URL** format: `<service-name>-predictor.<namespace>.svc.cluster.local`
- Use **port 80** for KServe internal services (not the external HTTPS port)

### Step 2: Configure Model Settings

Update the `model_config` section to match your models:

```yaml
model_config:
  "your-model-name":  # Must match the model name from your InferenceService
    reasoning_family: "qwen3"  # Options: qwen3, deepseek, gpt, gpt-oss - adjust based on your model family
    preferred_endpoints: ["your-model-endpoint"]
    pii_policy:
      allow_by_default: true
      pii_types_allowed: ["EMAIL_ADDRESS"]
```

### Step 3: Configure Category Routing

Update the `categories` section to define which models handle which types of queries:

```yaml
categories:
  - name: math
    system_prompt: "You are a mathematics expert..."
    model_scores:
      - model: your-model-name  # Must match model_config key
        score: 1.0  # Higher score = preferred for this category
        use_reasoning: true  # Enable extended reasoning
```

**Category Scoring**:
- Scores range from 0.0 to 1.0
- Higher scores indicate better suitability for the category
- The router selects the model with the highest score for each query category
- Use `use_reasoning: true` for complex tasks (math, chemistry, physics)

### Step 4: Adjust Storage Requirements

Edit `pvc.yaml` to set appropriate storage sizes:

```yaml
resources:
  requests:
    storage: 10Gi  # Adjust based on model sizes
```

Model storage requirements:
- Category classifier: ~500MB
- PII classifier: ~500MB
- Jailbreak classifier: ~500MB
- PII token classifier: ~500MB
- BERT embeddings: ~500MB
- **Total**: ~2.5GB minimum, recommend 10Gi for headroom

## Deployment

### Option 1: Deploy with Kustomize (Recommended)

```bash
# Switch to your namespace
oc project your-namespace

# Deploy all resources
oc apply -k deploy/kserve/

# Verify deployment
oc get pods -l app=semantic-router
oc get svc semantic-router-kserve
oc get route semantic-router-kserve
```

### Option 2: Deploy Individual Resources

```bash
# Switch to your namespace (or create it)
oc project your-namespace
# OR: oc new-project your-namespace

# Deploy in order
oc apply -f deploy/kserve/serviceaccount.yaml
oc apply -f deploy/kserve/pvc.yaml
oc apply -f deploy/kserve/configmap-router-config.yaml
oc apply -f deploy/kserve/configmap-envoy-config.yaml
oc apply -f deploy/kserve/deployment.yaml
oc apply -f deploy/kserve/service.yaml
oc apply -f deploy/kserve/route.yaml
```

### Monitor Deployment

Watch the pod initialization (model downloads take a few minutes):

```bash
# Watch pod status
oc get pods -l app=semantic-router -w

# Check init container logs (model download)
oc logs -l app=semantic-router -c model-downloader -f

# Check semantic router logs
oc logs -l app=semantic-router -c semantic-router -f

# Check Envoy logs
oc logs -l app=semantic-router -c envoy-proxy -f
```

### Verify Deployment

```bash
# Get the external route URL
ROUTER_URL=$(oc get route semantic-router-kserve -o jsonpath='{.spec.host}')
echo "https://$ROUTER_URL"

# Test health check
curl -k "https://$ROUTER_URL/v1/models"

# Test classification API
curl -k "https://$ROUTER_URL/v1/classify" \
  -H "Content-Type: application/json" \
  -d '{"text": "What is the derivative of x^2?"}'

# Test chat completion (replace 'your-model-name' with your actual model name)
curl -k "https://$ROUTER_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-name",
    "messages": [{"role": "user", "content": "Explain quantum entanglement"}]
  }'
```

## Testing with Different Categories

The router automatically classifies queries and routes to the best model. Test different categories:

```bash
ROUTER_URL=$(oc get route semantic-router-kserve -o jsonpath='{.spec.host}')
MODEL_NAME="your-model-name"  # Replace with your model name

# Math query (high reasoning enabled)
curl -k "https://$ROUTER_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL_NAME\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Solve the integral of x^2 dx\"}]
  }"

# Business query
curl -k "https://$ROUTER_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL_NAME\",
    \"messages\": [{\"role\": \"user\", \"content\": \"What is a good marketing strategy for SaaS?\"}]
  }"

# Test PII detection
curl -k "https://$ROUTER_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL_NAME\",
    \"messages\": [{\"role\": \"user\", \"content\": \"My SSN is 123-45-6789\"}]
  }"
```

## Monitoring

### Prometheus Metrics

Metrics are exposed on port 9190 at `/metrics`:

```bash
POD_NAME=$(oc get pods -l app=semantic-router -o jsonpath='{.items[0].metadata.name}')
oc port-forward $POD_NAME 9190:9190

# View metrics
curl http://localhost:9190/metrics
```

Key metrics:
- `semantic_router_classification_duration_seconds`: Classification latency
- `semantic_router_cache_hit_total`: Cache hit count
- `semantic_router_pii_detections_total`: PII detection count
- `semantic_router_requests_total`: Total requests processed

### Envoy Admin Interface

Access Envoy admin interface:

```bash
POD_NAME=$(oc get pods -l app=semantic-router -o jsonpath='{.items[0].metadata.name}')
oc port-forward $POD_NAME 19000:19000

# View stats
curl http://localhost:19000/stats
curl http://localhost:19000/clusters
```

### View Logs

```bash
# Combined logs from all containers
oc logs -l app=semantic-router --all-containers=true -f

# Semantic router only
oc logs -l app=semantic-router -c semantic-router -f

# Envoy only
oc logs -l app=semantic-router -c envoy-proxy -f
```

## Troubleshooting

### Pod Not Starting

```bash
# Check pod events
oc describe pod -l app=semantic-router

# Check PVC status
oc get pvc
```

**Common issues**:
- PVC pending: No storage class available or insufficient capacity
- ImagePullBackOff: Check image registry permissions
- Init container failing: Network issues downloading models from HuggingFace

### Model Download Issues

```bash
# Check init container logs
oc logs -l app=semantic-router -c model-downloader

# If models fail to download, you can pre-populate them:
# 1. Create a Job or pod with the model-downloader init container
# 2. Verify models exist in the PVC before starting the main deployment
```

### Routing Issues

```bash
# Check if semantic router can reach KServe predictors
POD_NAME=$(oc get pods -l app=semantic-router -o jsonpath='{.items[0].metadata.name}')
NAMESPACE=$(oc project -q)

# Test connectivity to InferenceService (replace 'your-model' with your InferenceService name)
oc exec $POD_NAME -c semantic-router -- \
  curl -v http://your-model-predictor.$NAMESPACE.svc.cluster.local/v1/models

# Check Envoy configuration
oc exec $POD_NAME -c envoy-proxy -- \
  curl http://localhost:19000/config_dump
```

### Classification Not Working

```bash
# Test the classification API directly
ROUTER_URL=$(oc get route semantic-router-kserve -o jsonpath='{.spec.host}')

curl -k "https://$ROUTER_URL/v1/classify" \
  -H "Content-Type: application/json" \
  -d '{"text": "What is 2+2?"}'

# Expected output should include category and model selection
```

### 503 Service Unavailable

**Possible causes**:
1. InferenceService is not ready
2. Incorrect endpoint address in config
3. Network policy blocking traffic

**Solutions**:
```bash
# Verify InferenceService is ready
oc get inferenceservice

# Check if predictor pods are running
oc get pods | grep predictor

# Verify network connectivity (replace 'your-model' with your InferenceService name)
POD_NAME=$(oc get pods -l app=semantic-router -o jsonpath='{.items[0].metadata.name}')
NAMESPACE=$(oc project -q)
oc exec $POD_NAME -c envoy-proxy -- \
  wget -O- http://your-model-predictor.$NAMESPACE.svc.cluster.local/v1/models
```

## Adding More InferenceServices

To add additional models:

1. **Deploy InferenceService** (if not already deployed)
2. **Update ConfigMap** (`configmap-router-config.yaml`):
   ```yaml
   vllm_endpoints:
     - name: "new-model-endpoint"
       address: "new-model-predictor.<namespace>.svc.cluster.local"  # Replace <namespace>
       port: 80
       weight: 1

   model_config:
     "new-model":
       reasoning_family: "qwen3"
       preferred_endpoints: ["new-model-endpoint"]
       pii_policy:
         allow_by_default: true

   categories:
     - name: coding
       system_prompt: "You are an expert programmer..."
       model_scores:
         - model: new-model
           score: 0.9
           use_reasoning: false
   ```

3. **Apply updated ConfigMap**:
   ```bash
   oc apply -f configmap-router-config.yaml

   # Restart deployment to pick up changes
   oc rollout restart deployment/semantic-router-kserve
   ```

## Performance Tuning

### Resource Limits

Adjust resource requests/limits in `deployment.yaml` based on load:

```yaml
resources:
  requests:
    memory: "3Gi"  # Increase for more models/cache
    cpu: "1"
  limits:
    memory: "6Gi"
    cpu: "2"
```

### Semantic Cache

Tune cache settings in `configmap-router-config.yaml`:

```yaml
semantic_cache:
  enabled: true
  similarity_threshold: 0.8  # Lower = more cache hits, higher = more accurate
  max_entries: 1000  # Increase for more cache capacity
  ttl_seconds: 3600  # Cache entry lifetime
```

### Scaling

Scale the deployment for high availability:

```bash
# Scale to multiple replicas
oc scale deployment/semantic-router-kserve --replicas=3

# Note: With multiple replicas, use Redis or Milvus for shared cache
```

## Integration with Applications

Point your OpenAI client to the semantic router:

**Python Example**:
```python
from openai import OpenAI

# Get your route URL from: oc get route semantic-router-kserve
client = OpenAI(
    base_url="https://semantic-router-your-namespace.apps.your-cluster.com/v1",
    api_key="not-needed"  # KServe doesn't require API key by default
)

response = client.chat.completions.create(
    model="your-model-name",  # Replace with your model name
    messages=[{"role": "user", "content": "Explain machine learning"}]
)
print(response.choices[0].message.content)
```

**cURL Example**:
```bash
curl -k "https://semantic-router-your-namespace.apps.your-cluster.com/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-name",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Cleanup

Remove all resources:

```bash
# Delete using kustomize
oc delete -k deploy/kserve/

# Or delete individual resources
oc delete route semantic-router-kserve
oc delete service semantic-router-kserve
oc delete deployment semantic-router-kserve
oc delete configmap semantic-router-kserve-config semantic-router-envoy-kserve-config
oc delete pvc semantic-router-models semantic-router-cache
oc delete serviceaccount semantic-router
```

## Additional Resources

- [vLLM Semantic Router Documentation](https://vllm-semantic-router.com)
- [OpenShift AI Documentation](https://access.redhat.com/documentation/en-us/red_hat_openshift_ai)
- [KServe Documentation](https://kserve.github.io/website/)
- [Envoy Proxy Documentation](https://www.envoyproxy.io/docs)

## Support

For issues and questions:
- GitHub Issues: https://github.com/vllm-project/semantic-router/issues
- Documentation: https://vllm-semantic-router.com/docs

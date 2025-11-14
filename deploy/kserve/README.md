# Semantic Router Integration with OpenShift AI KServe

Deploy vLLM Semantic Router as an intelligent gateway for your OpenShift AI KServe InferenceServices.

> **📍 Deployment Focus**: This guide is specifically for deploying semantic router on **OpenShift AI with KServe**.
>
> **🚀 Want to deploy quickly?** See [QUICKSTART.md](./QUICKSTART.md) for automated deployment in under 5 minutes.
>
> **📚 Learn about features?** See links to feature documentation throughout this guide.

## Overview

The semantic router acts as an intelligent API gateway that provides:

- **Intelligent Model Selection**: Automatically routes requests to the best model based on semantic understanding
  - *Learn more*: [Category Classification Training](../../src/training/classifier_model_fine_tuning/)
- **PII Detection & Protection**: Blocks or redacts sensitive information before sending to models
  - *Learn more*: [PII Detection Training](../../src/training/pii_model_fine_tuning/)
- **Prompt Guard**: Detects and blocks jailbreak attempts
  - *Learn more*: [Prompt Guard Training](../../src/training/prompt_guard_fine_tuning/)
- **Semantic Caching**: Reduces latency and costs through intelligent response caching
- **Category-Specific Prompts**: Injects domain-specific system prompts for better results
- **Tools Auto-Selection**: Automatically selects relevant tools for function calling

> **Note**: This directory focuses on **OpenShift deployment**. For general semantic router concepts, architecture, and feature details, see the [main project documentation](https://vllm-semantic-router.com).

## Prerequisites

Before deploying, ensure you have:

1. **OpenShift Cluster** with OpenShift AI (RHOAI) installed
2. **KServe InferenceService** already deployed and running
3. **OpenShift CLI (oc)** installed and logged in
4. **Cluster admin or namespace admin** permissions

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
    |                     [Sets routing headers]
    ↓
[KServe InferenceService Predictor]
    ↓
[vLLM Model Response]
```

### Components

- **Semantic Router**: ExtProc service that performs classification and routing logic
- **Envoy Proxy**: HTTP proxy that integrates with router via gRPC
- **Init Container**: Downloads ML classification models from HuggingFace (~2-3 min)

### Communication Flow

- **External**: HTTPS via OpenShift Route (TLS termination at edge)
- **Internal (Router ↔ Envoy)**: gRPC on port 50051
- **Internal (Envoy → KServe)**: HTTP on port 8080 (Istio provides mTLS)

### How Routing Works

1. Client sends OpenAI-compatible request to route
2. Envoy receives request and forwards to semantic router via ExtProc
3. Router performs:
   - Jailbreak detection (blocks malicious prompts)
   - PII detection (blocks/redacts sensitive data)
   - Semantic cache lookup (returns cached response if hit)
   - Category classification (math, coding, business, etc.)
   - Model selection based on category scores
4. Router sets routing headers for Envoy
5. Envoy routes to appropriate KServe predictor
6. Response flows back through Envoy to client
7. Router caches response for future queries

## Manual Deployment

### Step 1: Verify InferenceService

Check that your InferenceService is deployed and ready:

```bash
# Set your namespace
NAMESPACE=<your-namespace>

# List InferenceServices
oc get inferenceservice -n $NAMESPACE

# Example output:
# NAME           URL                                                  READY
# granite32-8b   http://granite32-8b-predictor.semantic.svc...       True
```

Create a stable ClusterIP service for the predictor:

```bash
INFERENCESERVICE_NAME=<your-inferenceservice-name>

# KServe creates a headless service by default (no stable ClusterIP)
# Create a stable ClusterIP service for consistent routing

# Option 1: Using the template file (recommended)
# Substitute variables and apply
sed -e "s/{{INFERENCESERVICE_NAME}}/$INFERENCESERVICE_NAME/g" \
    -e "s/{{NAMESPACE}}/$NAMESPACE/g" \
    service-predictor-stable.yaml | oc apply -f - -n $NAMESPACE

# Option 2: Using heredoc
cat <<EOF | oc apply -f - -n $NAMESPACE
apiVersion: v1
kind: Service
metadata:
  name: ${INFERENCESERVICE_NAME}-predictor-stable
  labels:
    app: ${INFERENCESERVICE_NAME}
    component: predictor-stable
spec:
  type: ClusterIP
  selector:
    serving.kserve.io/inferenceservice: ${INFERENCESERVICE_NAME}
  ports:
  - name: http
    port: 8080
    targetPort: 8080
    protocol: TCP
EOF

# Get the stable ClusterIP
PREDICTOR_SERVICE_IP=$(oc get svc "${INFERENCESERVICE_NAME}-predictor-stable" -n $NAMESPACE -o jsonpath='{.spec.clusterIP}')

echo "Predictor service ClusterIP: $PREDICTOR_SERVICE_IP"
```

> **Why a stable service?** KServe creates headless services by default (ClusterIP: None), which don't provide a stable IP. Pod IPs change on restart, requiring config updates. A ClusterIP service provides a stable IP that persists across pod restarts.

Verify the predictor is responding:

```bash
# Get pod name
PREDICTOR_POD=$(oc get pod -n $NAMESPACE \
  -l serving.kserve.io/inferenceservice=$INFERENCESERVICE_NAME \
  -o jsonpath='{.items[0].metadata.name}')

# Test the model endpoint
oc exec $PREDICTOR_POD -n $NAMESPACE -c kserve-container -- \
  curl -s http://localhost:8080/v1/models
```

### Step 2: Configure Router Settings

Edit `configmap-router-config.yaml` to configure your model:

#### A. Set vLLM Endpoint

Update the `vllm_endpoints` section with your predictor service IP:

```yaml
vllm_endpoints:
  - name: "my-model-endpoint"
    address: "172.30.45.97"  # Replace with your PREDICTOR_SERVICE_IP
    port: 8080
    weight: 1
```

> **Note**: The router requires an IP address format for validation. We use the **stable service ClusterIP** (not pod IP) because it persists across pod restarts.

#### B. Configure Model Settings

Update the `model_config` section:

```yaml
model_config:
  "my-model-name":  # Replace with your model name
    reasoning_family: "qwen3"  # Options: qwen3, deepseek, gpt, gpt-oss
    preferred_endpoints: ["my-model-endpoint"]
    pii_policy:
      allow_by_default: true
      pii_types_allowed: ["EMAIL_ADDRESS"]
```

**Reasoning Family Guide:**

| Family | Model Examples | Reasoning Parameter |
|--------|----------------|---------------------|
| `qwen3` | Qwen, Granite | `enable_thinking` |
| `deepseek` | DeepSeek | `thinking` |
| `gpt` | GPT-4 | `reasoning_effort` |
| `gpt-oss` | GPT-OSS variants | `reasoning_effort` |

#### C. Update Category Scores

Configure which categories route to your model:

```yaml
categories:
  - name: math
    system_prompt: "You are a mathematics expert..."
    model_scores:
      - model: my-model-name  # Must match model_config key
        score: 1.0  # 0.0-1.0, higher = preferred
        use_reasoning: true  # Enable for complex tasks

  - name: business
    system_prompt: "You are a business consultant..."
    model_scores:
      - model: my-model-name
        score: 0.8
        use_reasoning: false
```

**Score Guidelines:**

- `1.0`: Best suited for this category
- `0.7-0.9`: Good fit
- `0.4-0.6`: Moderate fit
- `0.0-0.3`: Not recommended

#### D. Set Default Model

```yaml
default_model: my-model-name
```

### Step 3: Configure Envoy Routing

Edit `configmap-envoy-config.yaml` to set the DNS endpoint.

Find the `kserve_dynamic_cluster` section and update:

```yaml
- name: kserve_dynamic_cluster
  type: STRICT_DNS
  load_assignment:
    cluster_name: kserve_dynamic_cluster
    endpoints:
    - lb_endpoints:
      - endpoint:
          address:
            socket_address:
              address: my-model-predictor.my-namespace.svc.cluster.local
              port_value: 8080
```

Replace:

- `my-model` with your InferenceService name
- `my-namespace` with your namespace

> **Note**: Envoy uses DNS (STRICT_DNS) for service discovery, so it will automatically resolve to the current pod IP even if it changes. This is different from the router config which requires the actual IP.

### Step 4: Configure Istio Security

Edit `peerauthentication.yaml` to set your namespace:

```yaml
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: semantic-router-kserve-permissive
  namespace: my-namespace  # Replace with your namespace
```

The `PERMISSIVE` mTLS mode allows both mTLS and plain HTTP, which is required for the router to communicate with both Envoy and the KServe predictor.

### Step 5: Configure Storage

Edit `pvc.yaml` to adjust storage sizes and class:

```yaml
# Models PVC
resources:
  requests:
    storage: 10Gi  # Adjust based on needs
storageClassName: gp3-csi  # Uncomment and set your storage class

# Cache PVC
resources:
  requests:
    storage: 5Gi  # Adjust based on cache requirements
```

**Storage Requirements:**

- **Models PVC**: ~2.5GB minimum for classification models, recommend 10Gi for headroom
- **Cache PVC**: Depends on cache size config, 5Gi is typically sufficient

### Step 6: Deploy Resources

Apply manifests in order:

```bash
# Set your namespace
NAMESPACE=<your-namespace>

# 1. ServiceAccount
oc apply -f serviceaccount.yaml -n $NAMESPACE

# 2. PersistentVolumeClaims
oc apply -f pvc.yaml -n $NAMESPACE

# 3. ConfigMaps
oc apply -f configmap-router-config.yaml -n $NAMESPACE
oc apply -f configmap-envoy-config.yaml -n $NAMESPACE

# 4. Istio Security
oc apply -f peerauthentication.yaml -n $NAMESPACE

# 5. Deployment
oc apply -f deployment.yaml -n $NAMESPACE

# 6. Service
oc apply -f service.yaml -n $NAMESPACE

# 7. Route
oc apply -f route.yaml -n $NAMESPACE
```

### Step 7: Monitor Deployment

Watch the pod initialization:

```bash
# Watch pod status
oc get pods -l app=semantic-router -n $NAMESPACE -w
```

The pod will go through these stages:

1. **Init:0/1** - Downloading models from HuggingFace (~2-3 minutes)
2. **PodInitializing** - Starting main containers
3. **Running (0/2)** - Containers starting
4. **Running (2/2)** - Ready to serve traffic

Monitor init container (model download):

```bash
oc logs -l app=semantic-router -c model-downloader -n $NAMESPACE -f
```

Check semantic router logs:

```bash
oc logs -l app=semantic-router -c semantic-router -n $NAMESPACE -f
```

Look for these log messages indicating successful startup:

```
{"level":"info","msg":"Starting vLLM Semantic Router ExtProc..."}
{"level":"info","msg":"Loaded category mapping with X categories"}
{"level":"info","msg":"Semantic cache enabled..."}
{"level":"info","msg":"Starting insecure LLM Router ExtProc server on port 50051..."}
```

Check Envoy logs:

```bash
oc logs -l app=semantic-router -c envoy-proxy -n $NAMESPACE -f
```

### Step 8: Get External URL

Retrieve the route URL:

```bash
ROUTER_URL=$(oc get route semantic-router-kserve -n $NAMESPACE -o jsonpath='{.spec.host}')
echo "External URL: https://$ROUTER_URL"
```

### Step 9: Test Deployment

Test the models endpoint:

```bash
curl -k "https://$ROUTER_URL/v1/models"
```

Expected response:

```json
{
  "object": "list",
  "data": [{
    "id": "MoM",
    "object": "model",
    "created": 1763143897,
    "owned_by": "vllm-semantic-router",
    "description": "Intelligent Router for Mixture-of-Models"
  }]
}
```

Test a chat completion:

```bash
curl -k "https://$ROUTER_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-model-name",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 50
  }'
```

Test semantic caching:

```bash
# First request (cache miss)
time curl -k -s "https://$ROUTER_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model": "my-model-name", "messages": [{"role": "user", "content": "What is the capital of France?"}], "max_tokens": 20}' \
  > /dev/null

# Second request (should be faster - cache hit)
time curl -k -s "https://$ROUTER_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model": "my-model-name", "messages": [{"role": "user", "content": "What is the capital of France?"}], "max_tokens": 20}' \
  > /dev/null
```

Run comprehensive validation tests:

```bash
# Set environment variables and run tests
NAMESPACE=$NAMESPACE MODEL_NAME=my-model-name ./test-semantic-routing.sh

# Or let the script auto-detect from config
cd deploy/kserve
./test-semantic-routing.sh
```

## Configuration Deep Dive

### Semantic Cache Configuration

The semantic cache stores responses based on embedding similarity:

```yaml
semantic_cache:
  enabled: true
  backend_type: "memory"         # Options: memory, milvus
  similarity_threshold: 0.8      # 0.0-1.0 (higher = more strict)
  max_entries: 1000              # Maximum cached responses
  ttl_seconds: 3600              # Entry lifetime (1 hour)
  eviction_policy: "fifo"        # Options: fifo, lru, lfu
  use_hnsw: true                 # Use HNSW index for fast similarity search
  hnsw_m: 16                     # HNSW parameter
  hnsw_ef_construction: 200      # HNSW parameter
  embedding_model: "bert"        # Model for embeddings
```

**Threshold Guidelines:**

- `0.95-1.0`: Very strict - only exact or near-exact matches
- `0.85-0.94`: Strict - recommended for accuracy (default: 0.8)
- `0.75-0.84`: Moderate - balance between hit rate and accuracy
- `0.60-0.74`: Loose - maximize cache hits, lower accuracy

**Backend Types:**

- **memory**: In-memory cache (default) - fast but not shared across replicas
- **milvus**: Distributed vector database - required for multi-replica deployments

### PII Detection Configuration

Configure what types of personally identifiable information to detect:

```yaml
classifier:
  pii_model:
    model_id: "models/pii_classifier_modernbert-base_presidio_token_model"
    use_modernbert: true
    threshold: 0.7  # Confidence threshold (0.0-1.0)
    use_cpu: true
```

> **Learn More**: For details on PII detection models and training, see [PII Model Fine-Tuning](../../src/training/pii_model_fine_tuning/).

**Per-Model PII Policies:**

```yaml
model_config:
  "my-model":
    pii_policy:
      allow_by_default: true  # Allow requests unless PII detected
      pii_types_allowed:      # Whitelist specific PII types
        - "EMAIL_ADDRESS"
        - "PHONE_NUMBER"
      # pii_types_allowed: []  # Empty list = block all PII
```

**Detected PII Types:**

- `CREDIT_CARD`
- `SSN` (Social Security Number)
- `EMAIL_ADDRESS`
- `PHONE_NUMBER`
- `PERSON` (names)
- `LOCATION`
- `DATE_TIME`
- `MEDICAL_LICENSE`
- `IP_ADDRESS`
- `IBAN_CODE`
- `US_DRIVER_LICENSE`
- `US_PASSPORT`

### Prompt Guard Configuration

Detect and block jailbreak/adversarial prompts:

```yaml
prompt_guard:
  enabled: true
  use_modernbert: true
  model_id: "models/jailbreak_classifier_modernbert-base_model"
  threshold: 0.7  # Confidence threshold (higher = more strict)
  use_cpu: true
```

When a jailbreak is detected, the request is blocked with an error response.

> **Learn More**: For details on jailbreak detection models and training, see [Prompt Guard Fine-Tuning](../../src/training/prompt_guard_fine_tuning/).

### Tools Auto-Selection

Automatically select relevant tools based on query similarity:

```yaml
tools:
  enabled: true
  top_k: 3                      # Number of tools to select
  similarity_threshold: 0.2     # Minimum similarity score
  tools_db_path: "config/tools_db.json"
  fallback_to_empty: true       # Return empty list if no matches
```

The tools database (`tools_db.json`) contains tool descriptions and the router uses semantic similarity to select the most relevant tools for each query.

### Category Classification

Categories determine routing decisions and system prompts:

```yaml
categories:
  - name: math
    system_prompt: "You are a mathematics expert. Provide step-by-step solutions."
    semantic_cache_enabled: false     # Override global cache setting
    semantic_cache_similarity_threshold: 0.9  # Override threshold
    model_scores:
      - model: small-model
        score: 0.7
        use_reasoning: true
      - model: large-model
        score: 1.0
        use_reasoning: true
```

**Per-Category Settings:**

- `semantic_cache_enabled`: Override global cache setting for this category
- `semantic_cache_similarity_threshold`: Custom threshold for category
- `model_scores`: List of models with scores and reasoning settings

The router selects the model with the highest score for the detected category.

> **Learn More**: For details on category classification models and training your own, see [Category Classifier Fine-Tuning](../../src/training/classifier_model_fine_tuning/).

## Multi-Model Configuration

To route between multiple InferenceServices:

### Step 1: Create Stable Services and Get ClusterIPs for All Models

```bash
# Create stable service for Model 1
cat <<EOF | oc apply -f - -n $NAMESPACE
apiVersion: v1
kind: Service
metadata:
  name: model1-predictor-stable
spec:
  type: ClusterIP
  selector:
    serving.kserve.io/inferenceservice: model1
  ports:
  - name: http
    port: 8080
    targetPort: 8080
EOF

# Create stable service for Model 2
cat <<EOF | oc apply -f - -n $NAMESPACE
apiVersion: v1
kind: Service
metadata:
  name: model2-predictor-stable
spec:
  type: ClusterIP
  selector:
    serving.kserve.io/inferenceservice: model2
  ports:
  - name: http
    port: 8080
    targetPort: 8080
EOF

# Get ClusterIPs
MODEL1_IP=$(oc get svc model1-predictor-stable -n $NAMESPACE -o jsonpath='{.spec.clusterIP}')
MODEL2_IP=$(oc get svc model2-predictor-stable -n $NAMESPACE -o jsonpath='{.spec.clusterIP}')

echo "Model 1 ClusterIP: $MODEL1_IP (stable)"
echo "Model 2 ClusterIP: $MODEL2_IP (stable)"
```

### Step 2: Update Router Config

Edit `configmap-router-config.yaml`:

```yaml
vllm_endpoints:
  - name: "small-model-endpoint"
    address: "172.30.10.50"  # Small model stable service ClusterIP
    port: 8080
    weight: 1

  - name: "large-model-endpoint"
    address: "172.30.20.100"  # Large model stable service ClusterIP
    port: 8080
    weight: 1

model_config:
  "small-model":
    reasoning_family: "qwen3"
    preferred_endpoints: ["small-model-endpoint"]
    pii_policy:
      allow_by_default: true

  "large-model":
    reasoning_family: "qwen3"
    preferred_endpoints: ["large-model-endpoint"]
    pii_policy:
      allow_by_default: true

categories:
  - name: simple-qa
    system_prompt: "You are a helpful assistant."
    model_scores:
      - model: small-model
        score: 1.0          # Prefer small model
        use_reasoning: false
      - model: large-model
        score: 0.5          # Fallback to large model
        use_reasoning: false

  - name: complex-reasoning
    system_prompt: "You are an expert problem solver."
    model_scores:
      - model: small-model
        score: 0.3          # Small model not preferred
        use_reasoning: true
      - model: large-model
        score: 1.0          # Prefer large model
        use_reasoning: true  # Enable reasoning for complex tasks
```

### Step 3: Update Envoy Config (Optional)

For multiple models in the same namespace, Envoy DNS resolution works automatically. If models are in different namespaces, you may need to create multiple clusters.

### Step 4: Apply Changes

```bash
oc apply -f configmap-router-config.yaml -n $NAMESPACE
oc rollout restart deployment/semantic-router-kserve -n $NAMESPACE
```

## Monitoring and Observability

### Prometheus Metrics

Metrics are exposed on port 9190 at `/metrics`:

```bash
# Port-forward to access metrics
POD=$(oc get pods -l app=semantic-router -n $NAMESPACE -o jsonpath='{.items[0].metadata.name}')
oc port-forward $POD 9190:9190 -n $NAMESPACE

# View metrics
curl http://localhost:9190/metrics
```

**Key Metrics:**

| Metric | Type | Description |
|--------|------|-------------|
| `semantic_router_requests_total` | Counter | Total requests processed |
| `semantic_router_classification_duration_seconds` | Histogram | Classification latency |
| `semantic_router_routing_latency_ms` | Histogram | Routing decision latency |
| `semantic_router_cache_hit_total` | Counter | Cache hits |
| `semantic_router_cache_miss_total` | Counter | Cache misses |
| `semantic_router_pii_detections_total` | Counter | PII detections by type |
| `semantic_router_jailbreak_detections_total` | Counter | Jailbreak detections |

### Application Logs

The router emits structured JSON logs with detailed request information:

```bash
# View router logs
oc logs -l app=semantic-router -c semantic-router -n $NAMESPACE -f
```

**Important Log Events:**

- `routing_decision`: Which model was selected and why

  ```json
  {
    "msg": "routing_decision",
    "selected_model": "granite32-8b",
    "category": "math",
    "reasoning_enabled": true,
    "routing_latency_ms": 45
  }
  ```

- `cache_hit`/`cache_miss`: Cache performance

  ```json
  {
    "msg": "cache_hit",
    "similarity": 0.98,
    "threshold": 0.8,
    "model": "granite32-8b"
  }
  ```

- `llm_usage`: Token usage and costs

  ```json
  {
    "msg": "llm_usage",
    "model": "granite32-8b",
    "prompt_tokens": 150,
    "completion_tokens": 200,
    "total_tokens": 350,
    "completion_latency_ms": 1250
  }
  ```

- `pii_detection`: PII found in requests

  ```json
  {
    "msg": "pii_detection",
    "pii_types": ["SSN", "CREDIT_CARD"],
    "action": "blocked"
  }
  ```

### Envoy Admin Interface

Access Envoy's admin interface for detailed routing statistics:

```bash
POD=$(oc get pods -l app=semantic-router -n $NAMESPACE -o jsonpath='{.items[0].metadata.name}')
oc port-forward $POD 19000:19000 -n $NAMESPACE
```

**Useful Endpoints:**

```bash
# Overall stats
curl http://localhost:19000/stats

# Cluster health
curl http://localhost:19000/clusters

# Full config dump
curl http://localhost:19000/config_dump

# Check upstream endpoints
curl http://localhost:19000/clusters | grep -A 10 "kserve_dynamic_cluster"
```

### Distributed Tracing (Optional)

Enable OpenTelemetry tracing in `configmap-router-config.yaml`:

```yaml
observability:
  tracing:
    enabled: true
    provider: "opentelemetry"
    exporter:
      type: "otlp"  # Options: stdout, otlp, jaeger
      endpoint: "jaeger-collector.observability.svc.cluster.local:4317"
      insecure: true
    sampling:
      type: "always_on"  # Options: always_on, always_off, trace_id_ratio
      rate: 1.0  # Sample rate (0.0-1.0)
```

## Troubleshooting

### Pod Stuck in Init

**Symptoms**: Pod stuck in `Init:0/1` state

**Diagnosis**:

```bash
# Check init container logs
oc logs -l app=semantic-router -c model-downloader -n $NAMESPACE

# Check events
oc describe pod -l app=semantic-router -n $NAMESPACE
```

**Common Causes**:

1. **Network issues**: Cannot reach HuggingFace
   - Solution: Check network policies, proxy settings

2. **PVC not bound**: Storage not provisioned

   ```bash
   oc get pvc -n $NAMESPACE
   ```

   - Solution: Check StorageClass, provision capacity

3. **OOM during model download**: Insufficient memory
   - Solution: Increase init container memory limits in `deployment.yaml`

### Router Container Crashing

**Symptoms**: Pod shows `CrashLoopBackOff`

**Diagnosis**:

```bash
oc logs -l app=semantic-router -c semantic-router -n $NAMESPACE --previous
```

**Common Causes**:

1. **Configuration error**: Invalid YAML or missing fields

   ```
   Failed to load config: yaml: unmarshal errors
   ```

   - Solution: Validate YAML syntax, check required fields

2. **Invalid IP address**: Router validation failed

   ```
   invalid IP address format, got: my-model.svc.cluster.local
   ```

   - Solution: Use service ClusterIP (not DNS) in `vllm_endpoints.address` - see Step 1 for creating stable service

3. **Missing models**: Classification models not downloaded

   ```
   failed to read mapping file: no such file or directory
   ```

   - Solution: Check init container completed successfully

### Cannot Connect to InferenceService

**Symptoms**: 503 errors, upstream connect errors in logs

**Diagnosis**:

```bash
# Test from router pod
POD=$(oc get pods -l app=semantic-router -n $NAMESPACE -o jsonpath='{.items[0].metadata.name}')

oc exec $POD -c semantic-router -n $NAMESPACE -- \
  curl -v http://my-model-predictor.$NAMESPACE.svc.cluster.local:8080/v1/models
```

**Common Causes**:

1. **InferenceService not ready**:

   ```bash
   oc get inferenceservice -n $NAMESPACE
   ```

   - Solution: Wait for READY=True, check predictor logs

2. **Wrong DNS name**: Incorrect service name in Envoy config
   - Solution: Verify format: `<inferenceservice>-predictor.<namespace>.svc.cluster.local`

3. **Network policy blocking**: Istio/NetworkPolicy restrictions

   ```bash
   oc get networkpolicies -n $NAMESPACE
   ```

   - Solution: Add policy to allow traffic from router to predictor

4. **PeerAuthentication conflict**: mTLS mode mismatch

   ```bash
   oc get peerauthentication -n $NAMESPACE
   ```

   - Solution: Ensure PERMISSIVE mode or adjust Envoy TLS config

### Predictor Pod IP Changed (If Using Pod IP Instead of Service IP)

> **Note**: This issue should not occur if you're using the **stable ClusterIP service** approach (recommended). Service ClusterIPs persist across pod restarts.

**If you used pod IP directly** (not recommended):

**Symptoms**: Router logs show connection refused after predictor restart

**Solution**:

1. Switch to stable service approach (recommended):

   ```bash
   # Create stable service
   cat <<EOF | oc apply -f - -n $NAMESPACE
   apiVersion: v1
   kind: Service
   metadata:
     name: my-model-predictor-stable
   spec:
     type: ClusterIP
     selector:
       serving.kserve.io/inferenceservice: my-model
     ports:
     - name: http
       port: 8080
       targetPort: 8080
   EOF

   # Get ClusterIP and update config
   NEW_IP=$(oc get svc my-model-predictor-stable -n $NAMESPACE -o jsonpath='{.spec.clusterIP}')
   oc edit configmap semantic-router-kserve-config -n $NAMESPACE
   # Update vllm_endpoints.address to $NEW_IP

   # Restart router
   oc rollout restart deployment/semantic-router-kserve -n $NAMESPACE
   ```

> **Best Practice**: Always use a stable ClusterIP service instead of pod IPs to avoid this issue entirely.

### Cache Not Working

**Symptoms**: No cache hits in logs, all requests show `cache_miss`

**Diagnosis**:

```bash
# Check logs for cache events
oc logs -l app=semantic-router -c semantic-router -n $NAMESPACE \
  | grep -E "cache_hit|cache_miss"
```

**Common Causes**:

1. **Threshold too high**: Similarity threshold prevents matches

   ```yaml
   similarity_threshold: 0.99  # Too strict
   ```

   - Solution: Lower threshold to 0.8-0.85

2. **Cache disabled**: Not enabled in config
   - Solution: Set `semantic_cache.enabled: true`

3. **Different model parameter**: Requests use different `max_tokens`, `temperature`, etc.
   - Cache considers full request context, not just the prompt

4. **Cache expired**: TTL too short
   - Solution: Increase `ttl_seconds`

## Scaling and High Availability

### Horizontal Scaling

Scale the router for high availability:

```bash
oc scale deployment/semantic-router-kserve --replicas=3 -n $NAMESPACE
```

**Important Considerations**:

- **Cache**: With multiple replicas, each has its own in-memory cache
  - For shared cache, configure Milvus backend
  - Or use session affinity to route users to same replica

- **Resource Requirements**: Each replica needs ~3Gi memory
  - Plan capacity accordingly

### Vertical Scaling

Adjust resources in `deployment.yaml`:

```yaml
containers:
- name: semantic-router
  resources:
    requests:
      memory: "4Gi"  # Increase for larger models
      cpu: "2"       # Increase for higher throughput
    limits:
      memory: "8Gi"
      cpu: "4"
```

Apply changes:

```bash
oc apply -f deployment.yaml -n $NAMESPACE
```

### Auto-Scaling with HPA

Create HorizontalPodAutoscaler:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: semantic-router-kserve-hpa
  namespace: <your-namespace>
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: semantic-router-kserve
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

Apply:

```bash
oc apply -f hpa.yaml -n $NAMESPACE
```

Monitor autoscaling:

```bash
oc get hpa -n $NAMESPACE -w
```

### Load Balancing

OpenShift Route automatically load balances across healthy pods. For additional control:

```yaml
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  name: semantic-router-kserve
  annotations:
    haproxy.router.openshift.io/balance: roundrobin  # leastconn, source
spec:
  # ... rest of route config
```

## Advanced Topics

### Using Milvus for Shared Cache

For multi-replica deployments with shared cache:

1. Deploy Milvus in your cluster
2. Update `configmap-router-config.yaml`:

   ```yaml
   semantic_cache:
     enabled: true
     backend_type: "milvus"
     milvus:
       host: "milvus.semantic.svc.cluster.local"
       port: 19530
       collection_name: "semantic_cache"
   ```

3. Apply and restart:

   ```bash
   oc apply -f configmap-router-config.yaml -n $NAMESPACE
   oc rollout restart deployment/semantic-router-kserve -n $NAMESPACE
   ```

### Custom Classification Models

To use your own fine-tuned classification models:

1. Train your custom models:
   - [Category Classifier](../../src/training/classifier_model_fine_tuning/)
   - [PII Detector](../../src/training/pii_model_fine_tuning/)
   - [Prompt Guard](../../src/training/prompt_guard_fine_tuning/)
2. Upload to HuggingFace or internal registry
3. Update `deployment.yaml` init container to download your model
4. Update model paths in `configmap-router-config.yaml`

> **Training Documentation**: Each training directory contains detailed guides for fine-tuning models on your own datasets.

### Integration with Service Mesh

The deployment includes Istio integration:

- `sidecar.istio.io/inject: "true"` enables Envoy sidecar
- `PeerAuthentication` configures mTLS mode
- Distributed tracing propagates through Istio

For custom Istio configuration, edit `deployment.yaml` annotations.

## Cleanup

Remove all deployed resources:

```bash
NAMESPACE=<your-namespace>

oc delete route semantic-router-kserve -n $NAMESPACE
oc delete service semantic-router-kserve -n $NAMESPACE
oc delete deployment semantic-router-kserve -n $NAMESPACE
oc delete configmap semantic-router-kserve-config semantic-router-envoy-kserve-config -n $NAMESPACE
oc delete pvc semantic-router-models semantic-router-cache -n $NAMESPACE
oc delete peerauthentication semantic-router-kserve-permissive -n $NAMESPACE
oc delete serviceaccount semantic-router -n $NAMESPACE
```

> **Warning**: Deleting PVCs will remove downloaded models and cache data. To preserve data, skip PVC deletion.

## Related Documentation

### Within This Repository

- **[Category Classifier Training](../../src/training/classifier_model_fine_tuning/)** - Train custom category classification models
- **[PII Detector Training](../../src/training/pii_model_fine_tuning/)** - Train custom PII detection models
- **[Prompt Guard Training](../../src/training/prompt_guard_fine_tuning/)** - Train custom jailbreak detection models
- **[Main Project README](../../README.md)** - Project overview and general documentation
- **[CLAUDE.md](../../CLAUDE.md)** - Development guide and architecture details

### Other Deployment Options

- **[OpenShift Deployment](../openshift/)** - Deploy with standalone vLLM containers (not KServe)
- *This directory* - OpenShift AI KServe deployment (you are here)

### External Resources

- **Main Project**: https://github.com/vllm-project/semantic-router
- **Full Documentation**: https://vllm-semantic-router.com
- **OpenShift AI Docs**: https://access.redhat.com/documentation/en-us/red_hat_openshift_ai
- **KServe Docs**: https://kserve.github.io/website/
- **Envoy Proxy Docs**: https://www.envoyproxy.io/docs

## Getting Help

- 📖 **Quick Start**: See [QUICKSTART.md](./QUICKSTART.md) for automated deployment
- 💬 **GitHub Issues**: https://github.com/vllm-project/semantic-router/issues
- 📚 **Discussions**: https://github.com/vllm-project/semantic-router/discussions

## License

This project follows the vLLM Semantic Router license. See the main repository for details.

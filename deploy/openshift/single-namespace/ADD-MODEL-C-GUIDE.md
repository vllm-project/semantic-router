# Adding Model-C to Semantic Router Configuration

This guide explains the theoretical configuration changes needed to add a third model (Model-C) to the semantic router deployment.

## Overview

The semantic router uses a configuration-driven approach to route requests to different LLM models based on intent classification. Adding a new model requires updates to:

1. Service definition (networking)
2. Router configuration (routing logic)
3. Envoy configuration (optional, only if needed)

## Architecture Context

```
Client Request
    ↓
Envoy Proxy (8801)
    ↓
ExtProc gRPC → Semantic Router (50051)
    ├─ Classify intent
    ├─ Check PII/Jailbreak
    └─ Route decision → Set header: x-gateway-destination-endpoint
    ↓
Envoy routes based on header
    ↓
Model-A (ClusterIP:8000) OR Model-B (ClusterIP:8000) OR Model-C (ClusterIP:8000)
```

## Required Configuration Changes

### 1. Service Definition (`services.yaml`)

**Purpose**: Create a Kubernetes Service to expose Model-C pods via a stable ClusterIP address.

**Location**: `deploy/openshift/single-namespace/services.yaml`

**What to add**:

```yaml
---
apiVersion: v1
kind: Service
metadata:
  name: model-c
  namespace: vllm-semantic-router
  labels:
    app: model-c
spec:
  type: ClusterIP
  ports:
    - port: 8000
      targetPort: 8000
      protocol: TCP
      name: http
  selector:
    app: model-c
```

**Key points**:
- Service name: `model-c`
- Port: `8000` (standard vLLM/llm-katan inference port)
- Selector: Must match the labels on your Model-C pods
- ClusterIP will be dynamically assigned by Kubernetes

---

### 2. Router Configuration (`configmap-router.yaml`)

**Purpose**: Configure the semantic router to recognize Model-C as a valid routing destination.

**Location**: `deploy/openshift/single-namespace/configmap-router.yaml`

This file has three sections that need updates:

#### Section A: Add Model-C Endpoint

**Location in file**: Under `vllm_endpoints:` array

**What to add**:

```yaml
vllm_endpoints:
  - name: model-a-endpoint
    address: "172.30.87.247"  # Example - will be dynamically set
    port: 8000
    model: Model-A

  - name: model-b-endpoint
    address: "172.30.246.170"  # Example - will be dynamically set
    port: 8000
    model: Model-B

  # ADD THIS:
  - name: model-c-endpoint
    address: "172.30.XXX.XXX"  # Will be dynamically set from Service ClusterIP
    port: 8000
    model: Model-C
```

**Key points**:
- `name`: Unique identifier for this endpoint
- `address`: ClusterIP of the model-c Service (auto-populated by deploy script)
- `port`: 8000 (standard)
- `model`: Display name used in responses

---

#### Section B: Add Model Configuration

**Location in file**: Under `model_config:` section

**What to add**:

```yaml
model_config:
  Model-A:
    reasoning_family: "qwen3"
    supports_reasoning: true

  Model-B:
    reasoning_family: "qwen3"
    supports_reasoning: true

  # ADD THIS:
  Model-C:
    reasoning_family: "qwen3"  # Or whatever reasoning model family Model-C uses
    supports_reasoning: true    # Set to false if model doesn't support reasoning
```

**Key points**:
- `reasoning_family`: Defines which reasoning prompt format to use
  - Options: `qwen3`, `deepseek`, `o1`, `generic`
  - Must match the model's actual reasoning capabilities
- `supports_reasoning`: Boolean flag
  - `true`: Router can enable reasoning mode for this model
  - `false`: Router will not use reasoning, even if category requests it

---

#### Section C: Update Categories with Model-C Scores

**Location in file**: Under `categories:` array

**Option 1: Create New Categories for Model-C**

```yaml
categories:
  # ... existing categories (math, coding, history, etc.) ...

  # ADD NEW CATEGORIES:
  - label: science
    system_prompt: "You are an expert in scientific analysis and research."
    use_reasoning: true
    model_scores:
      Model-C: 1.0
      Model-A: 0.3
      Model-B: 0.3

  - label: creative_writing
    system_prompt: "You are a creative writing assistant."
    use_reasoning: false
    model_scores:
      Model-C: 1.0
      Model-A: 0.2
      Model-B: 0.5
```

**Option 2: Update Existing Categories to Include Model-C**

```yaml
categories:
  - label: math
    system_prompt: "You are a mathematical reasoning expert."
    use_reasoning: true
    model_scores:
      Model-A: 1.0
      Model-B: 0.3
      Model-C: 0.5  # ADD THIS - now math queries can route to Model-C

  - label: coding
    system_prompt: "You are an expert programmer."
    use_reasoning: true
    model_scores:
      Model-A: 1.0
      Model-B: 0.3
      Model-C: 0.8  # ADD THIS - Model-C gets high score for coding
```

**How routing works**:
1. User sends query: "Write a Python function to calculate fibonacci"
2. Classifier identifies category: `coding` (confidence: 0.85)
3. Router looks up `coding` category's `model_scores`:
   - Model-A: 1.0
   - Model-B: 0.3
   - Model-C: 0.8
4. If confidence >= 0.6 threshold:
   - Router selects Model-A (highest score: 1.0)
5. If confidence < 0.6:
   - Fallback to default model (Model-A)

**Key points**:
- Scores range from 0.0 to 1.0
- Higher score = more likely to be selected for that category
- Model with highest score wins (when confidence is high enough)
- `use_reasoning: true` enables reasoning mode if model supports it

---

### 3. Envoy Configuration (`configmap-envoy.yaml`)

**Purpose**: Envoy handles the actual HTTP routing to backend models.

**Location**: `deploy/openshift/single-namespace/configmap-envoy.yaml`

**GOOD NEWS**: **No changes required!**

**Why?**

The current Envoy configuration uses a special cluster type called `ORIGINAL_DST`:

```yaml
clusters:
  - name: vllm_dynamic_cluster
    type: ORIGINAL_DST
    lb_policy: CLUSTER_PROVIDED
```

This means:
- Semantic router sets a header: `x-gateway-destination-endpoint: 172.30.XXX.XXX:8000`
- Envoy reads this header and routes directly to that IP:port
- No need to pre-define Model-C in Envoy configuration
- Completely dynamic routing based on router decisions

**If you wanted static Envoy configuration** (not recommended):

```yaml
# NOT NEEDED - just for reference
clusters:
  - name: model-c-cluster
    type: STRICT_DNS
    lb_policy: ROUND_ROBIN
    load_assignment:
      cluster_name: model-c-cluster
      endpoints:
        - lb_endpoints:
            - endpoint:
                address:
                  socket_address:
                    address: model-c.vllm-semantic-router.svc.cluster.local
                    port_value: 8000
```

---

## Configuration Summary

### Minimum Required Changes

| File | Section | Action |
|------|---------|--------|
| `services.yaml` | Root | Add Service definition for model-c |
| `configmap-router.yaml` | `vllm_endpoints` | Add model-c-endpoint entry |
| `configmap-router.yaml` | `model_config` | Add Model-C configuration |
| `configmap-router.yaml` | `categories` | Add Model-C scores to categories |
| `configmap-envoy.yaml` | N/A | **No changes needed** |

### What Gets Auto-Populated

The `deploy-all.sh` script automatically:

1. Creates the Service and waits for ClusterIP assignment
2. Retrieves the ClusterIP: `oc get svc model-c -o jsonpath='{.spec.clusterIP}'`
3. Updates `configmap-router.yaml` with the actual IP address
4. Applies the updated ConfigMap to the cluster

---

## Example: Complete Model-C Configuration

### Scenario
- Model-C: Specialized for scientific and creative tasks
- Service ClusterIP: 172.30.100.50 (example)
- Reasoning: Supports Qwen3 reasoning format

### services.yaml
```yaml
---
apiVersion: v1
kind: Service
metadata:
  name: model-c
  namespace: vllm-semantic-router
spec:
  type: ClusterIP
  ports:
    - port: 8000
      targetPort: 8000
      name: http
  selector:
    app: model-c
```

### configmap-router.yaml
```yaml
vllm_endpoints:
  - name: model-a-endpoint
    address: "172.30.87.247"
    port: 8000
    model: Model-A
  - name: model-b-endpoint
    address: "172.30.246.170"
    port: 8000
    model: Model-B
  - name: model-c-endpoint
    address: "172.30.100.50"  # Auto-populated
    port: 8000
    model: Model-C

model_config:
  Model-A:
    reasoning_family: "qwen3"
    supports_reasoning: true
  Model-B:
    reasoning_family: "qwen3"
    supports_reasoning: true
  Model-C:
    reasoning_family: "qwen3"
    supports_reasoning: true

categories:
  - label: science
    system_prompt: "You are an expert in scientific analysis."
    use_reasoning: true
    model_scores:
      Model-C: 1.0  # Highest - Model-C is best for science
      Model-A: 0.4
      Model-B: 0.4

  - label: creative_writing
    system_prompt: "You are a creative writing assistant."
    use_reasoning: false
    model_scores:
      Model-C: 1.0  # Highest - Model-C is best for creative writing
      Model-A: 0.3
      Model-B: 0.6

  - label: math
    system_prompt: "You are a mathematical reasoning expert."
    use_reasoning: true
    model_scores:
      Model-A: 1.0  # Model-A remains best for math
      Model-B: 0.3
      Model-C: 0.5
```

---

## How Routing Decisions Work

### Example Query Flow

**Query**: "Explain the process of photosynthesis"

1. **Classification**
   - Classifier identifies: `science` category
   - Confidence: 0.82

2. **Model Selection**
   - Lookup `science` category scores:
     - Model-C: 1.0 ← **Winner**
     - Model-A: 0.4
     - Model-B: 0.4
   - Confidence 0.82 >= 0.6 threshold ✓

3. **Routing**
   - Router sets header: `x-gateway-destination-endpoint: 172.30.100.50:8000`
   - Envoy routes request to Model-C Service

4. **Response**
   - Model-C generates response
   - Response includes: `"model": "Model-C"`

---

## Key Concepts

### ClusterIP vs Pod IP
- **ClusterIP**: Stable virtual IP for the Service (doesn't change)
- **Pod IP**: Direct IP of the pod (changes when pod restarts)
- Always use ClusterIP in configuration for stability

### Confidence Threshold
- Default: 0.6
- If classification confidence < 0.6 → fallback to default model
- Configured in `configmap-router.yaml` under `classifier.confidence_threshold`

### Model Scores
- Not probabilities - just relative preferences
- Highest score wins (if confidence is high enough)
- Can give same score to multiple models (router picks first)

### Reasoning Mode
- `use_reasoning: true` in category → enables chain-of-thought
- `supports_reasoning: true` in model_config → model can handle reasoning
- Both must be true for reasoning to activate

---

## Applying Changes to Running Deployment

You can apply configuration changes **without re-running `deploy-all.sh`** by following these steps.

### Method 1: Manual IP Update (Recommended)

```bash
# Step 1: Apply the new/updated Service for Model-C
oc apply -f deploy/openshift/single-namespace/services.yaml

# Step 2: Get the ClusterIP that was assigned to Model-C
MODEL_C_IP=$(oc get svc model-c -n vllm-semantic-router -o jsonpath='{.spec.clusterIP}')
echo "Model-C ClusterIP: $MODEL_C_IP"

# Step 3: Manually edit configmap-router.yaml
# Replace "172.30.XXX.XXX" with the actual Model-C IP from Step 2

# Step 4: Apply the updated ConfigMap
oc apply -f deploy/openshift/single-namespace/configmap-router.yaml

# Step 5: Restart semantic-router deployment to load new config
oc rollout restart deployment/semantic-router -n vllm-semantic-router

# Step 6: Wait for rollout to complete
oc rollout status deployment/semantic-router -n vllm-semantic-router
```

### Method 2: Automated IP Update (Like deploy-all.sh)

```bash
# Step 1: Apply Service
oc apply -f deploy/openshift/single-namespace/services.yaml

# Step 2: Get Model-C ClusterIP
MODEL_C_IP=$(oc get svc model-c -n vllm-semantic-router -o jsonpath='{.spec.clusterIP}')
echo "Model-C ClusterIP: $MODEL_C_IP"

# Step 3: Create temporary ConfigMap with updated IP
TEMP_CONFIG="/tmp/configmap-router-updated.yaml"
cp deploy/openshift/single-namespace/configmap-router.yaml "$TEMP_CONFIG"

# Replace the Model-C placeholder IP (adjust sed pattern to match your placeholder)
sed -i "s/address: \"172\.30\.XXX\.XXX\"/address: \"$MODEL_C_IP\"/" "$TEMP_CONFIG"

# Step 4: Apply the updated ConfigMap
oc apply -f "$TEMP_CONFIG"

# Step 5: Restart semantic-router deployment
oc rollout restart deployment/semantic-router -n vllm-semantic-router

# Step 6: Clean up temporary file
rm -f "$TEMP_CONFIG"
```

### Method 3: Apply Both Files Together

```bash
# Apply both files at once (if IPs are already correct in configmap-router.yaml)
oc apply -f deploy/openshift/single-namespace/services.yaml \
          -f deploy/openshift/single-namespace/configmap-router.yaml

# Restart to pick up ConfigMap changes
oc rollout restart deployment/semantic-router -n vllm-semantic-router
```

### Why Restart is Required

**Important**: ConfigMaps don't automatically update running pods!

- **ConfigMaps** are mounted into pods at pod startup
- Changing a ConfigMap updates the object in Kubernetes
- But running pods keep their original mounted version
- **You MUST restart** the deployment: `oc rollout restart deployment/semantic-router`
- The restart does a rolling update (zero downtime)

### Verification Steps

After applying changes:

```bash
# 1. Verify ConfigMap was updated
oc get configmap router-config -n vllm-semantic-router -o yaml | grep -A 5 "model-c-endpoint"

# 2. Verify Service exists and has ClusterIP
oc get svc model-c -n vllm-semantic-router

# 3. Watch pods restart (wait for 2/2 Ready)
oc get pods -n vllm-semantic-router -l app=semantic-router -w

# 4. Check logs to confirm new config loaded
oc logs deployment/semantic-router -c semantic-router -n vllm-semantic-router | head -50

# 5. Test routing to Model-C
ROUTE=$(oc get route envoy-proxy -n vllm-semantic-router -o jsonpath='{.spec.host}')
curl -X POST http://$ROUTE/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "Explain photosynthesis"}]}'
```

### What Happens During Apply

| Resource | Command | Effect |
|----------|---------|--------|
| **Service** | `oc apply -f services.yaml` | Creates/updates Service, assigns ClusterIP immediately |
| **ConfigMap** | `oc apply -f configmap-router.yaml` | Updates ConfigMap object in Kubernetes |
| **Pods** | `oc rollout restart deployment/...` | Triggers rolling restart to mount updated ConfigMap |

### Summary

**Key Insight**: No need to re-run `deploy-all.sh`!

1. **Services**: Applied with `oc apply -f services.yaml` → takes effect immediately
2. **ConfigMaps**: Applied with `oc apply -f configmap-router.yaml` → updates the object
3. **Pods**: Must be restarted with `oc rollout restart` → mounts the updated ConfigMap

The restart is a **rolling update** with zero downtime - Kubernetes will:
- Start new pod with updated config
- Wait for it to be ready (2/2 containers)
- Terminate old pod
- All traffic continues flowing during the update

---

## Troubleshooting

### Model-C not receiving requests
- Check confidence threshold (may be too low)
- Verify Model-C has highest score in relevant categories
- Check logs: `oc logs -f deployment/semantic-router -c semantic-router`

### Classification not detecting categories
- Train new classifier with Model-C categories
- Update category mapping JSON files
- Lower confidence threshold temporarily for testing

### Envoy routing errors
- Verify ClusterIP is correct in configmap
- Check Service selector matches Model-C pod labels
- Verify Model-C pod is Running: `oc get pods -l app=model-c`

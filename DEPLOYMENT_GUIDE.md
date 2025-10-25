# vLLM Semantic Router - OpenShift Single-Namespace Deployment Guide

## Overview

This document explains the complete OpenShift single-namespace deployment architecture for vLLM Semantic Router with separate pods for the router and multiple vLLM models.

**Deployment Date**: October 20, 2025
**Namespace**: `vllm-semantic-router`
**Architecture**: Separate pods with IP-based routing
**External Access**: OpenShift Route with TLS termination

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Components](#components)
3. [Networking Configuration](#networking-configuration)
4. [Semantic Routing Logic](#semantic-routing-logic)
5. [Configuration Files](#configuration-files)
6. [Deployment Process](#deployment-process)
7. [Testing and Verification](#testing-and-verification)
8. [External Access](#external-access)

---

## Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    OpenShift Cluster                             │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │         Namespace: vllm-semantic-router                     │ │
│  │                                                              │ │
│  │  ┌──────────────────┐                                       │ │
│  │  │  semantic-router │  (Pod with 2 containers)              │ │
│  │  │  ┌────────────┐  │                                       │ │
│  │  │  │   Envoy    │  │ ← External HTTPS (Route)             │ │
│  │  │  │   :8801    │  │                                       │ │
│  │  │  └─────┬──────┘  │                                       │ │
│  │  │        │         │                                       │ │
│  │  │  ┌─────▼──────┐  │                                       │ │
│  │  │  │  Router    │  │                                       │ │
│  │  │  │   :50051   │  │ ← ExtProc gRPC                       │ │
│  │  │  │            │  │                                       │ │
│  │  │  │ Classifies │  │                                       │ │
│  │  │  │  & Routes  │  │                                       │ │
│  │  │  └────┬───┬───┘  │                                       │ │
│  │  └───────┼───┼──────┘                                       │ │
│  │          │   │                                               │ │
│  │    ┌─────┘   └──────┐                                       │ │
│  │    │                │                                       │ │
│  │  ┌─▼──────────┐  ┌──▼─────────┐                            │ │
│  │  │  model-a   │  │  model-b   │                            │ │
│  │  │            │  │            │                            │ │
│  │  │  vLLM      │  │  vLLM      │                            │ │
│  │  │  :8000     │  │  :8000     │                            │ │
│  │  │            │  │            │                            │ │
│  │  │ Math/Sci   │  │ Business/  │                            │ │
│  │  │ Biology    │  │ Law        │                            │ │
│  │  └────────────┘  └────────────┘                            │ │
│  │                                                              │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

### Pod Architecture

**3 Separate Pods**:

1. **semantic-router** (2 containers)
   - Container 1: Envoy Proxy (port 8801)
   - Container 2: Semantic Router ExtProc (port 50051)

2. **model-a** (1 container)
   - vLLM server for math, science, biology queries

3. **model-b** (1 container)
   - vLLM server for business, law queries

---

## Components

### 1. Envoy Proxy (Port 8801)

**Purpose**: Entry point for all LLM requests, handles external HTTP traffic

**Configuration**: `configmap-envoy.yaml`

Key features:

- HTTP listener on port 8801
- External processing via gRPC to semantic router (port 50051)
- Dynamic routing based on headers set by router
- Health checking endpoints

### 2. Semantic Router ExtProc (Port 50051)

**Purpose**: Intelligent request classification and routing

**Configuration**: `configmap-router.yaml`

Key features:

- ModernBERT-based category classification
- PII detection (ModernBERT token classifier)
- Jailbreak protection (ModernBERT binary classifier)
- Semantic caching (in-memory)
- Request/response logging
- Metrics collection (port 9190)

**Classification Model**: ModernBERT sequence classifier

- Categories: math, science, biology, business, law, etc.
- Confidence threshold-based routing
- Automatic reasoning mode selection

### 3. vLLM Model Servers

**model-a** and **model-b**:

- Standard vLLM server with OpenAI-compatible API
- Port: 8000
- Model: llm-katan (loaded from PVC)

---

## Networking Configuration

### Service Discovery

All communication uses **ClusterIP services** with IP-based routing.

#### Services Configuration (`services.yaml`)

```yaml
# Envoy Proxy Service
apiVersion: v1
kind: Service
metadata:
  name: envoy-proxy
  namespace: vllm-semantic-router
spec:
  type: ClusterIP
  ports:
    - name: http
      port: 8801
      targetPort: 8801
  selector:
    app: semantic-router
---
# Router ExtProc Service
apiVersion: v1
kind: Service
metadata:
  name: semantic-router
  namespace: vllm-semantic-router
spec:
  type: ClusterIP
  ports:
    - name: grpc
      port: 50051
      targetPort: 50051
    - name: metrics
      port: 9190
      targetPort: 9190
  selector:
    app: semantic-router
---
# Model A Service
apiVersion: v1
kind: Service
metadata:
  name: model-a
  namespace: vllm-semantic-router
spec:
  type: ClusterIP
  ports:
    - name: http
      port: 8000
      targetPort: 8000
  selector:
    app: model-a
---
# Model B Service
apiVersion: v1
kind: Service
metadata:
  name: model-b
  namespace: vllm-semantic-router
spec:
  type: ClusterIP
  ports:
    - name: http
      port: 8000
      targetPort: 8000
  selector:
    app: model-b
```

### Service IP Addresses (Example from Deployment)

```
envoy-proxy:        172.30.56.108:8801
semantic-router:    172.30.xxx.xxx:50051
model-a:            172.30.170.236:8000
model-b:            172.30.143.41:8000
```

**Note**: These IPs are dynamically assigned by OpenShift and will differ per deployment.

**IMPORTANT**: The deployment script (`scripts/deploy-all.sh`) automatically detects and updates these IP addresses during deployment, so you don't need to manually configure them.

---

## Semantic Routing Logic

### How Routing Works

```
1. Client Request
   ↓
2. Envoy Proxy receives request on :8801
   ↓
3. Envoy calls Router ExtProc via gRPC (:50051)
   ↓
4. Router performs:
   - Cache check (if enabled)
   - PII detection
   - Jailbreak detection
   - Category classification (ModernBERT)
   ↓
5. Router sets header: x-gateway-destination-endpoint
   ↓
6. Envoy routes to appropriate vLLM backend
   ↓
7. vLLM processes request and returns response
   ↓
8. Router caches response (if enabled)
   ↓
9. Response returned to client
```

### Category Classification

**Router Configuration** (`configmap-router.yaml`):

```yaml
classifier:
  category_model: "models/category_classifier_modernbert-base_model"
  category_mapping: "config/category_classifier_mapping.json"
  confidence_threshold: 0.7

categories:
  - name: "math"
    system_prompt: "You are a helpful math tutor..."
    use_reasoning: true
    vllm_endpoint: "model-a"

  - name: "science"
    system_prompt: "You are a knowledgeable science expert..."
    use_reasoning: false
    vllm_endpoint: "model-a"

  - name: "biology"
    system_prompt: "You are a biology expert..."
    use_reasoning: false
    vllm_endpoint: "model-a"

  - name: "business"
    system_prompt: "You are a business consultant..."
    use_reasoning: false
    vllm_endpoint: "model-b"

  - name: "law"
    system_prompt: "You are a legal expert..."
    use_reasoning: false
    vllm_endpoint: "model-b"
```

### vLLM Endpoint Configuration

**Critical**: The router configuration MUST use IP addresses (not DNS):

```yaml
vllm_endpoints:
  - name: "model-a"
    address: "172.30.170.236:8000"  # Must be IP address
    priority: 1

  - name: "model-b"
    address: "172.30.143.41:8000"   # Must be IP address
    priority: 1
```

**Why IP addresses?**

- The router validation requires valid IPv4/IPv6 addresses
- DNS resolution happens at a different layer
- This ensures direct pod-to-pod communication

**Automatic IP Configuration**

The `deploy-all.sh` script automatically handles IP configuration:

1. Creates services and waits for ClusterIP assignment
2. Retrieves the actual ClusterIP addresses for `model-a` and `model-b`
3. Updates the router ConfigMap with the correct IPs
4. Deploys all components with the updated configuration

This means the deployment will work on **any OpenShift cluster** without manual IP configuration!

### Envoy Dynamic Routing

**Envoy Configuration** (`configmap-envoy.yaml`):

The Envoy config includes a dynamic routing cluster that reads the destination from headers:

```yaml
clusters:
  - name: dynamic_forward_proxy_cluster
    type: STRICT_DNS
    dns_lookup_family: V4_ONLY
    typed_extension_protocol_options:
      envoy.extensions.upstreams.http.v3.HttpProtocolOptions:
        explicit_http_config:
          http_protocol_options: {}
    lb_policy: CLUSTER_PROVIDED
    cluster_type:
      name: envoy.clusters.dynamic_forward_proxy
      typed_config:
        "@type": type.googleapis.com/envoy.extensions.clusters.dynamic_forward_proxy.v3.ClusterConfig
        dns_cache_config:
          name: dynamic_forward_proxy_cache_config
          dns_lookup_family: V4_ONLY
```

The router sets the `x-gateway-destination-endpoint` header which Envoy uses to route requests.

---

## Configuration Files

### File Structure

```
deploy/openshift/single-namespace/
├── namespace.yaml                    # Namespace definition
├── imagestreams.yaml                 # ImageStreams for builds
├── buildconfig-llm-katan.yaml        # BuildConfig for model
├── buildconfig-router.yaml           # BuildConfig for router
├── pvcs.yaml                         # PersistentVolumeClaims
├── configmap-envoy.yaml              # Envoy proxy config
├── configmap-router.yaml             # Router config (config.yaml)
├── services.yaml                     # All service definitions
├── deployment-model-a.yaml           # Model A deployment
├── deployment-model-b.yaml           # Model B deployment
├── deployment-router.yaml            # Router + Envoy deployment
├── route.yaml                        # External HTTPS route
├── README.md                         # Deployment guide
└── scripts/
    └── deploy-all.sh                 # Automated deployment
```

### Key Configuration Files

#### 1. Router Configuration (`configmap-router.yaml`)

Contains the complete `config.yaml` for the semantic router:

- **BERT Models**: Embedding and classification models
- **Classifier**: Category, PII, and jailbreak models
- **Categories**: Category definitions with system prompts
- **vLLM Endpoints**: Backend model server addresses
- **Caching**: Semantic cache configuration
- **Observability**: Logging and tracing settings

#### 2. Envoy Configuration (`configmap-envoy.yaml`)

Contains the Envoy proxy configuration:

- **Listeners**: HTTP listener on port 8801
- **Clusters**: Dynamic forward proxy cluster
- **Routes**: Request routing logic
- **Filters**: External processing filter (gRPC to router)
- **Health Checks**: Endpoint health monitoring

#### 3. Route Configuration (`route.yaml`)

**NEW**: External HTTPS access configuration

```yaml
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  name: envoy-proxy
  namespace: vllm-semantic-router
spec:
  to:
    kind: Service
    name: envoy-proxy
  port:
    targetPort: 8801
  tls:
    termination: edge
    insecureEdgeTerminationPolicy: Redirect
```

**Features**:

- Auto-generated hostname: `envoy-proxy-vllm-semantic-router.apps.<cluster-domain>`
- TLS edge termination (automatic HTTPS)
- HTTP to HTTPS redirect
- No port-forward needed!

---

## Deployment Process

### Prerequisites

1. OpenShift cluster access
2. `oc` CLI installed and authenticated
3. Models available (from all-in-one deployment or downloaded)

### Automated Deployment

```bash
cd deploy/openshift/single-namespace/scripts
./deploy-all.sh
```

### Manual Deployment Steps

```bash
# 1. Create namespace
oc apply -f namespace.yaml

# 2. Create ImageStreams
oc apply -f imagestreams.yaml

# 3. Create PVCs
oc apply -f pvcs.yaml

# 4. Create ConfigMaps
oc apply -f configmap-envoy.yaml
oc apply -f configmap-router.yaml

# 5. Create Services
oc apply -f services.yaml

# 6. Start builds
oc apply -f buildconfig-llm-katan.yaml
oc apply -f buildconfig-router.yaml

# Wait for builds to complete
oc wait --for=condition=Complete build/llm-katan-1 -n vllm-semantic-router --timeout=600s
oc wait --for=condition=Complete build/semantic-router-1 -n vllm-semantic-router --timeout=600s

# 7. Deploy models
oc apply -f deployment-model-a.yaml
oc apply -f deployment-model-b.yaml

# 8. Deploy router
oc apply -f deployment-router.yaml

# 9. Create external route
oc apply -f route.yaml

# 10. Verify deployment
oc get pods -n vllm-semantic-router
oc get svc -n vllm-semantic-router
oc get route -n vllm-semantic-router
```

### Get Service IPs (Manual Deployment Only)

**Note**: If you used `deploy-all.sh`, this step is done automatically. Only follow these steps if you're deploying manually.

After creating services, get the ClusterIP addresses:

```bash
oc get svc -n vllm-semantic-router -o wide
```

**Important**: Update `configmap-router.yaml` with the actual IPs BEFORE creating the ConfigMap:

```bash
# Get Model A IP
MODEL_A_IP=$(oc get svc model-a -n vllm-semantic-router -o jsonpath='{.spec.clusterIP}')

# Get Model B IP
MODEL_B_IP=$(oc get svc model-b -n vllm-semantic-router -o jsonpath='{.spec.clusterIP}')

echo "Model-A IP: $MODEL_A_IP"
echo "Model-B IP: $MODEL_B_IP"

# Manually edit configmap-router.yaml with these IPs before applying
# Then apply the updated ConfigMap
```

---

## Testing and Verification

### Test Results (From Actual Deployment)

```
Namespace: vllm-semantic-router
Pods Running: 3/3
- model-a: 1/1 Running
- model-b: 1/1 Running
- semantic-router: 2/2 Running (router + envoy)

Service IPs:
- Model-A: 172.30.170.236:8000
- Model-B: 172.30.143.41:8000
- Envoy: 172.30.56.108:8801
```

### Test Queries

#### Test 1: Math Query (Routes to Model-A)

```bash
curl -X POST http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "What is the derivative of x^2?"}],
    "max_tokens": 100
  }'
```

**Result**: ✅ Routed to Model-A (math category, confidence: high)

#### Test 2: Business Query (Routes to Model-B)

```bash
curl -X POST http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "What are the key principles of effective marketing?"}],
    "max_tokens": 100
  }'
```

**Result**: ✅ Routed to Model-B (business category, confidence: high)

#### Test 3: Biology Query (Routes to Model-A)

```bash
curl -X POST http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Explain the process of photosynthesis"}],
    "max_tokens": 100
  }'
```

**Result**: ✅ Routed to Model-A (biology category, confidence: high)

#### Test 4: Law Query (Routes to Model-B)

```bash
curl -X POST http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "What is contract law?"}],
    "max_tokens": 100
  }'
```

**Result**: ✅ Routed to Model-B (law category, confidence: 97.67%)

### Features Verified

- ✅ Category classification (ModernBERT)
- ✅ Jailbreak detection (all queries benign)
- ✅ PII detection policy enforcement
- ✅ Semantic caching (memory backend)
- ✅ Automatic model selection
- ✅ Entropy-based reasoning decisions
- ✅ Category-specific system prompts
- ✅ Request/response logging
- ✅ Metrics collection (9190/metrics)

### Performance Metrics

```
Classification latency:  ~70-100ms
Routing latency:        ~500-750ms
End-to-end latency:     ~2-4s (includes LLM inference)
Cache hit rate:         0% (new queries)
```

---

## External Access

### OpenShift Route

**Route Hostname** (auto-generated):

```
envoy-proxy-vllm-semantic-router.apps.cluster-6w9xx.6w9xx.sandbox1663.opentlc.com
```

### Access from Outside the Cluster

```bash
# Get the route hostname
ROUTE_HOST=$(oc get route envoy-proxy -n vllm-semantic-router -o jsonpath='{.spec.host}')

# Test via HTTPS
curl -k https://$ROUTE_HOST/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 50
  }'
```

### Route Features

- **TLS Termination**: Automatic HTTPS with edge termination
- **HTTP Redirect**: HTTP requests automatically redirected to HTTPS
- **Auto Hostname**: OpenShift generates hostname automatically
- **Persistent**: No need for port-forward, accessible from anywhere
- **Secure**: TLS certificate managed by OpenShift

### Before vs After Route

**Before (Port-Forward)**:

```bash
# Terminal 1: Start port-forward
oc port-forward svc/envoy-proxy 8801:8801 -n vllm-semantic-router

# Terminal 2: Test
curl http://localhost:8801/v1/chat/completions ...
```

**After (Route)**:

```bash
# Direct HTTPS access
curl -k https://envoy-proxy-vllm-semantic-router.apps.<cluster-domain>/v1/chat/completions ...
```

---

## Architecture Benefits

### Separate Pods Approach

**Benefits**:

1. **Isolation**: Each model runs independently
2. **Scalability**: Can scale models independently
3. **Resource Management**: Different resource limits per model
4. **Updates**: Update router without affecting models
5. **Debugging**: Easier to troubleshoot individual components

**vs All-in-One**:

- All-in-one: Simpler but less flexible
- Separate pods: More complex but production-ready

### IP-Based Routing

**Benefits**:

1. **Performance**: Direct pod-to-pod communication
2. **Reliability**: No DNS resolution delays
3. **Validation**: Router validates IP addresses
4. **Explicit**: Clear network paths

**Configuration Required**:

- Must update ConfigMap with actual ClusterIP addresses after deployment
- IPs are stable within the cluster but may change on redeployment

---

## Monitoring and Observability

### Prometheus Metrics

Available at: `http://semantic-router:9190/metrics`

Key metrics:

- Request count
- Classification latency
- Routing decisions
- Cache hit ratio
- Model selection distribution

### Logging

Router logs include:

- Request classification results
- Model selection decisions
- Cache hits/misses
- Error conditions

View logs:

```bash
oc logs -f deployment/semantic-router -c semantic-router -n vllm-semantic-router
```

---

## Troubleshooting

### Common Issues

**1. Pods not starting**

```bash
oc get pods -n vllm-semantic-router
oc describe pod <pod-name> -n vllm-semantic-router
```

**2. Models not found**

```bash
# Check PVC
oc get pvc -n vllm-semantic-router

# Check models in PVC
oc exec -it deployment/model-a -n vllm-semantic-router -- ls -la /app/models
```

**3. Routing not working**

```bash
# Check router logs
oc logs deployment/semantic-router -c semantic-router -n vllm-semantic-router

# Check Envoy logs
oc logs deployment/semantic-router -c envoy -n vllm-semantic-router

# Verify ConfigMap IPs match service IPs
oc get svc -n vllm-semantic-router -o wide
oc get configmap router-config -n vllm-semantic-router -o yaml | grep address
```

**4. Route not accessible**

```bash
# Check route status
oc get route envoy-proxy -n vllm-semantic-router
oc describe route envoy-proxy -n vllm-semantic-router

# Test internal access first
oc port-forward svc/envoy-proxy 8801:8801 -n vllm-semantic-router
curl http://localhost:8801/v1/chat/completions ...
```

---

## Summary

This deployment successfully demonstrates:

✅ **Separate pods architecture** for production use
✅ **Intelligent semantic routing** using ModernBERT classification
✅ **IP-based networking** with ClusterIP services
✅ **External HTTPS access** via OpenShift Route
✅ **Comprehensive testing** with verified routing decisions
✅ **Production-ready** monitoring and observability

**Pull Request**: https://github.com/vllm-project/semantic-router/pull/490

---

## Uninstalling the Deployment

To remove the deployment and clean up all resources:

### Using the Uninstall Script

```bash
cd deploy/openshift/single-namespace/scripts
./uninstall.sh
```

The uninstall script will:

1. **Delete Route** - Removes external HTTPS access
2. **Delete Deployments** - Stops and removes all pods (semantic-router, model-a, model-b)
3. **Delete Services** - Removes all service endpoints
4. **Delete ConfigMaps** - Removes configuration
5. **Delete Builds and BuildConfigs** - Removes build configurations and cached builds
6. **Delete ImageStreams** - Removes container image references
7. **Delete PVCs (optional)** - Removes persistent storage (WARNING: deletes all models and data)
8. **Delete Namespace (optional)** - Completely removes the namespace

### Interactive Confirmations

The script asks for confirmation before:
- Starting the uninstall process
- Deleting PVCs (persistent data)
- Deleting the namespace

### Uninstall Options

**Partial Uninstall** (preserves data):
- Answer "no" when asked about PVC deletion
- Answer "no" when asked about namespace deletion
- This removes the running deployment but keeps models and configurations

**Complete Uninstall** (removes everything):
- Answer "yes" to all prompts
- This completely removes all resources including persistent data

### Manual Uninstall

If you prefer to uninstall manually:

```bash
# Delete deployments and services
oc delete deployment semantic-router model-a model-b -n vllm-semantic-router
oc delete service envoy-proxy semantic-router model-a model-b -n vllm-semantic-router
oc delete route envoy-proxy -n vllm-semantic-router

# Delete ConfigMaps
oc delete configmap semantic-router-config envoy-config -n vllm-semantic-router

# Delete builds and ImageStreams
oc delete bc semantic-router llm-katan -n vllm-semantic-router
oc delete imagestream semantic-router llm-katan -n vllm-semantic-router

# Delete PVCs (WARNING: deletes all data)
oc delete pvc semantic-router-models semantic-router-cache -n vllm-semantic-router

# Delete namespace (WARNING: removes everything)
oc delete namespace vllm-semantic-router
```

---

## Next Steps

1. **Monitor PR**: Track feedback from vLLM team
2. **Address reviews**: Respond to any change requests
3. **Scale testing**: Test with higher load
4. **Add models**: Use the same pattern for additional models
5. **Production deployment**: Deploy to production OpenShift cluster

---

**Created**: October 20, 2025
**Author**: szedan
**Email**: szedan@redhat.com

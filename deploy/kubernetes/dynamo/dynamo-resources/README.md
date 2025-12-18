# Dynamo E2E Testing Resources

This directory contains Kubernetes manifests for E2E testing of **Semantic Router with NVIDIA Dynamo integration**.

## 🎥 Demo Video

▶️ **[Watch the E2E Demo on YouTube](https://www.youtube.com/watch?v=rRULSR9gTds&list=PLmrddZ45wYcuPrXisC-yl7bMI39PLo4LO&index=2)**

## ⚠️ GPU Requirements

**This test requires a VM with at least 3 GPUs:**

| Component | GPU | Description |
|-----------|-----|-------------|
| Frontend | GPU 0 | Dynamo Frontend with KV-aware routing (`--router-mode kv`) |
| VLLMPrefillWorker | GPU 1 | Handles prefill phase of inference (`--is-prefill-worker`) |
| VLLMDecodeWorker | GPU 2 | Handles decode phase of inference |

## 🏗️ Deployment Pattern: Disaggregated Router Deployment

This integration uses **Pattern 4: Disaggregated Router Deployment** (`disagg_router.yaml`) - the most advanced configuration from [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/deploy/README.md).

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DYNAMO FRONTEND                          │
│              (KV-aware routing enabled)                     │
│                  --router-mode kv                           │
└─────────────────────────┬───────────────────────────────────┘
                          │
            ┌─────────────┴─────────────┐
            │                           │
            ▼                           ▼
┌───────────────────────┐   ┌───────────────────────┐
│   PREFILL WORKER      │   │   DECODE WORKER       │
│  --is-prefill-worker  │   │  (decode-only)        │
│  Processes input      │──▶│  Generates output     │
│  tokens               │   │  tokens               │
└───────────────────────┘   └───────────────────────┘
```

### Why Disaggregated Router Deployment?

| Feature | Benefit |
|---------|---------|
| **Disaggregated Serving** | Separate Prefill/Decode workers for optimal GPU utilization |
| **KV-Aware Routing** | Routes requests to workers with relevant KV cache (prefix cache optimization) |
| **Reduced Latency** | 2-4x faster responses on repeated/similar prompts |
| **Better Throughput** | Workers specialize in their tasks |

### KV Cache Routing Benefits

The Frontend with `--router-mode kv` enables:

- **Prefix Cache Hits**: Routes to workers with cached prefixes
- **Load Balancing**: Considers worker busyness and cache state
- **Optimal Worker Selection**: Uses cost formula based on prefill/decode load

## 🔧 Prerequisites (One-Time Setup)

> **Tested on:** RHEL 9 VM with NVIDIA GPUs

Before running the E2E tests, you must configure Docker to use the NVIDIA runtime as the default. This is a **one-time setup** that persists across reboots.

### Step 1: Configure NVIDIA Runtime as Default

```bash
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
```

### Step 2: Restart Docker

```bash
sudo systemctl restart docker
```

### Step 3: Verify Configuration

```bash
docker info | grep -i "default runtime"
```

You should see: `Default Runtime: nvidia`

> **Note:** The E2E framework verifies this configuration but does not set it automatically (requires sudo privileges).

## What the E2E Framework Does

The E2E framework automatically:

1. **Verifies** Docker runtime is set to `nvidia` (fails with instructions if not configured)
2. Creates Kind cluster with GPU support
3. Copies NVIDIA libraries to the Kind worker node
4. Deploys the NVIDIA device plugin

## What We Test

✅ **What IS Tested:**

- Dynamo CRD deployment (`DynamoGraphDeployment`)
- Dynamo Frontend coordination with workers via etcd/NATS
- Semantic Router ExtProc integration with Dynamo
- RBAC for accessing Dynamo CRDs
- Request routing through Dynamo Frontend
- **Real vLLM inference with GPU** (TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- Disaggregated serving (Prefill + Decode workers)
- GPU utilization and memory management
- Model name rewriting via Semantic Router

## Complete Architecture & Request Flow

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          CLIENT (You)                                    │
│  curl -X POST http://localhost:8080/v1/chat/completions                 │
│       -d '{"model": "MoM", "messages": [...]}'                          │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   ENVOY GATEWAY (Gateway Proxy)                         │
│  • Namespace: envoy-gateway-system                                       │
│  • Service: envoy-default-semantic-router-31cbd78c:80                   │
│  • Role: API Gateway, routes traffic, applies policies                  │
│  • EnvoyPatchPolicy: ENABLED (critical for ExtProc)                     │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              SEMANTIC ROUTER (ExtProc Filter)                           │
│  • Namespace: vllm-semantic-router-system                               │
│  • Service: semantic-router-ext-proc:9002                               │
│  • Role: Intelligent request routing & classification                   │
│                                                                          │
│  Processing Steps:                                                      │
│  1. Receives request from Envoy (ExtProc protocol)                      │
│  2. Parses body: {"model": "MoM", "messages": [...]}                    │
│  3. Classifies query: "What is 2+2?" → category="math" (93.3%)         │
│  4. Looks up category in config:                                        │
│     categories:                                                         │
│       - name: math                                                      │
│         model_scores:                                                   │
│           - model: TinyLlama/TinyLlama-1.1B-Chat-v1.0                                    │
│             score: 1.0                                                  │
│  5. Selects model: TinyLlama/TinyLlama-1.1B-Chat-v1.0 (score: 1.0)                       │
│  6. Rewrites request: model="MoM" → model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"           │
│  7. Returns modified request to Envoy                                   │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  ENVOY GATEWAY (After ExtProc)                          │
│  • Request now has: model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"                           │
│  • Consults HTTPRoute for routing decision                              │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              GATEWAY API HTTPRoute (Routing Rules)                      │
│  • Name: semantic-router-to-dynamo                                      │
│  • Namespace: default                                                   │
│  • Rule: Match /v1/* → forward to Dynamo Frontend                       │
│  • Backend: Service/vllm-frontend (dynamo-system:8000)                 │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              DYNAMO FRONTEND (Coordination Layer)                       │
│  • Created by: DynamoGraphDeployment CRD                                │
│  • Service: vllm-frontend                                               │
│  • Namespace: dynamo-system                                             │
│  • Role: Intelligent routing to workers via etcd/NATS                   │
│  • Features: Request queuing, worker selection, coordination            │
└─────────────────────────────────────────────────────────────────────────┘
                        │                          │
                        ▼                          ▼
    ┌───────────────────────────┐    ┌───────────────────────────┐
    │  PREFILL WORKER (GPU 1)   │    │  DECODE WORKER (GPU 2)    │
    │  • Created by: Dynamo CRD │    │  • Created by: Dynamo CRD │
    │  • Namespace: dynamo-sys  │    │  • Namespace: dynamo-sys  │
    │  • Image: vllm-runtime    │    │  • Image: vllm-runtime    │
    │  • Model: TinyLlama       │    │  • Model: TinyLlama       │
    │  • Port: 9090             │    │  • Port: 9090             │
    │  • Coordination: etcd+NATS│    │  • Coordination: etcd+NATS│
    │                           │    │                           │
    │  Processing:              │    │  Processing:              │
    │  1. Receives from Frontend│    │  1. Receives from Frontend│
    │  2. Prefill phase (KV)    │    │  2. Decode phase (tokens) │
    │  3. Real GPU inference    │    │  3. Real GPU inference    │
    │  4. Returns via Frontend  │    │  4. Returns via Frontend  │
    └───────────────────────────┘    └───────────────────────────┘
                        │                          │
                        └──────────┬───────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      RESPONSE FLOW (Backwards)                          │
│                                                                          │
│  Worker → Service → HTTPRoute → Envoy Gateway                           │
│                                       │                                  │
│                                       ▼                                  │
│                         SEMANTIC ROUTER (Response Processing)           │
│                         • Intercepts response via ExtProc               │
│                         • Logs usage metrics                            │
│                         • Updates cache                                 │
│                         • Returns to Envoy                              │
│                                       │                                  │
│                                       ▼                                  │
│                                  ENVOY GATEWAY                           │
│                         • Forwards response to client                   │
│                                       │                                  │
│                                       ▼                                  │
│                                    CLIENT                                │
│  Receives: {"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "choices": [...]}             │
└─────────────────────────────────────────────────────────────────────────┘
```

## Detailed Component Breakdown

### 1. Envoy Gateway (API Gateway Layer)

**What it is**: A Kubernetes-native API Gateway based on Envoy Proxy

**Configuration**:

- Deployed via Helm with custom values (`envoy-gateway-values.yaml`)
- **Critical setting**: `extensionApis.enableEnvoyPatchPolicy: true`
- Without this, EnvoyPatchPolicy resources are rejected!

**Role**:

- Entry point for all HTTP traffic
- Applies routing rules from Gateway API resources
- Integrates with external processors (ExtProc) like Semantic Router
- Handles TLS termination, rate limiting, etc.

**How it works**:

1. Receives client request on port 80
2. Checks EnvoyPatchPolicy for custom filters
3. Calls Semantic Router ExtProc service (if configured)
4. Applies HTTPRoute rules to forward to backend
5. Returns response to client

### 2. Semantic Router (Intelligent Routing Layer)

**What it is**: An AI-powered router that classifies queries and routes to optimal models

**Configuration**:

- Config: `../semantic-router/config.yaml`
- Values: `../semantic-router-values/values.yaml`

**Role**:

- Intercepts requests via Envoy ExtProc protocol
- Classifies user queries into categories (math, science, general, etc.)
- Selects the best model based on category and scores
- Rewrites the `model` field in the request
- Logs routing decisions and metrics

**Classification Models**:

- Category classifier: `models/category_classifier_modernbert-base_model`
- Embedding model: `models/all-MiniLM-L12-v2`
- Jailbreak detector: `models/jailbreak_classifier_modernbert-base_model`

**How it works**:

1. Envoy sends request to ExtProc gRPC service on port 9002
2. Semantic Router parses JSON body
3. Runs classification: query text → category (with confidence score)
4. Looks up category in config to find best model
5. Rewrites `model` field: "MoM" → "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
6. Returns modified request to Envoy
7. On response path: logs metrics, updates cache

**Example Classification**:

```
Query: "What is 2+2?"
  ↓
Category: "math" (confidence: 93.3%)
  ↓
Config lookup:
  categories:
    - name: math
      model_scores:
        - model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
          score: 1.0
  ↓
Selected: TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

### 3. Gateway API Resources (Routing Configuration)

**What it is**: Kubernetes-native API for configuring traffic routing

**Resources deployed** (`gwapi-resources.yaml`):

#### a. GatewayClass

- Name: `envoy-gateway`
- Controller: `gateway.envoyproxy.io/gatewayclass-controller`
- Defines the type of Gateway implementation

#### b. Gateway

- Name: `semantic-router`
- Namespace: `default`
- Listeners: HTTP on port 80
- Links to GatewayClass and allows HTTPRoutes

#### c. HTTPRoute

- Name: `semantic-router-to-dynamo`
- Matches: All paths starting with `/v1/`
- Backend: `vllm-frontend` (dynamo-system:8000)
- Routes requests through Dynamo Frontend (created by DynamoGraphDeployment)

#### d. EnvoyPatchPolicy

- Name: `semantic-router-extproc-patch-policy`
- Applies ExtProc filter to Gateway
- Configures Semantic Router as external processor
- **Status must be "Accepted: True"** or ExtProc won't work!

#### e. ReferenceGrant

- Allows HTTPRoute in `default` namespace to reference Service in `dynamo-system`
- Required for cross-namespace routing

### 4. Dynamo Platform (Coordination Layer)

**What it is**: NVIDIA's distributed inference platform

**Components** (deployed via Helm chart `dynamo-platform`):

#### a. etcd

- Distributed key-value store
- Stores worker state, model metadata, KV cache mappings
- Workers register themselves in etcd
- Enables KV cache sharing across workers

#### b. NATS

- Message queue for asynchronous communication
- Workers subscribe to model-specific topics
- Router publishes requests to appropriate topics
- Enables dynamic load balancing and request batching

#### c. Dynamo Operator

- Kubernetes operator that manages Dynamo lifecycle
- Watches for DynamoGraphDeployment CRDs (**actively used in E2E**)
- Creates Frontend and Worker deployments/services
- Manages worker scaling and health

**Role in E2E**:

- ✅ **Dynamo Operator**: Actively used to create Frontend and Workers from CRD
- ✅ **Dynamo Frontend**: Coordinates routing to workers via etcd/NATS
- ✅ **etcd**: Used for worker registration and coordination
- ✅ **NATS**: Used for message queuing between Frontend and Workers
- ✅ **DynamoGraphDeployment CRD**: Defines Frontend and Worker specifications
- ✅ **Real vLLM Inference**: GPU-enabled with TinyLlama model

### 5. DynamoGraphDeployment CRD (Dynamo Configuration)

**What it is**: Kubernetes CRD that defines Dynamo Frontend and Workers

**Configuration** (`dynamo-graph-deployment.yaml`):

- **CRD**: `nvidia.com/v1alpha1/DynamoGraphDeployment`
- **Name**: `vllm`
- **Namespace**: `dynamo-system`

**Frontend Service** (GPU 0):

- Image: `nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.6.1.post1`
- Command: `python3 -m dynamo.frontend --http-port 8000`
- Replicas: 1
- Role: HTTP API server, coordinates requests to workers via etcd/NATS
- Service Name (created by operator): `vllm-frontend`

**VLLMPrefillWorker Service** (GPU 1):

- **Image**: `nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.6.1.post1`
- **Command**: `python3 -m dynamo.vllm --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --is-prefill-worker`
- **Replicas**: 1
- **Role**: Handles prefill phase (prompt processing, KV cache generation)
- Service Name (created by operator): `vllm-vllmprefillworker`

**VLLMDecodeWorker Service** (GPU 2):

- **Image**: `nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.6.1.post1`
- **Command**: `python3 -m dynamo.vllm --model TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Replicas**: 1
- **Role**: Handles decode phase (token generation)
- Service Name (created by operator): `vllm-vllmdecodeworker`

### 6. RBAC (Permissions)

**What it is**: Role-based access control for Semantic Router

**Configuration** (`rbac.yaml`):

- **ClusterRole**: `dynamo-extproc-access`
  - Allows Semantic Router to `get`, `list`, `watch` DynamoGraphDeployments
  - Allows access to pods, services, endpoints for monitoring
- **ClusterRoleBinding**: Links role to `semantic-router` ServiceAccount
- **ServiceAccount**: `semantic-router` (in `vllm-semantic-router-system`)

**Why needed**: Semantic Router may query Dynamo CRDs for routing decisions

- Port: 8000
- Selector: `app=vllm-worker-demo`
- Load balances across 2 pods (round-robin)

**How workers process requests** (disaggregated serving):

1. Frontend receives POST to `/v1/chat/completions`
2. Routes to Prefill Worker via etcd/NATS
3. Prefill Worker processes prompt, generates KV cache
4. Decode Worker receives KV cache, generates tokens
5. Response flows back through Frontend to client

### 6. Kubernetes Service (Load Balancing Layer)

**What it is**: Kubernetes native load balancer

**How it works**:

- Service selector matches pods with `app=vllm-worker-demo`
- Maintains list of healthy pod IPs (Endpoints)
- Uses round-robin to distribute requests
- Health checks via readiness probes
- Automatic failover if pod becomes unhealthy

**Why this matters**:

- Without service: Client needs to know all pod IPs
- With service: Single stable endpoint, automatic load balancing
- Enables horizontal scaling (add more replicas)

## What This Tests

### ✅ Semantic Router Capabilities:

1. **Request classification** - Classifies "What is 2+2?" → "math"
2. **Model selection** - Selects TinyLlama/TinyLlama-1.1B-Chat-v1.0 based on category
3. **Request rewriting** - Changes model="MoM" → model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
4. **ExtProc integration** - Works with Envoy Gateway via gRPC
5. **Response processing** - Logs metrics, updates cache
6. **Fallback behavior** - Uses default_model when no category matches

### ✅ Infrastructure Capabilities:

1. **Load balancing** - Requests distributed across 2 workers
2. **Service discovery** - HTTPRoute finds workers via Service
3. **High availability** - One worker fails, traffic goes to the other
4. **Cross-namespace routing** - default → dynamo-system via ReferenceGrant
5. **EnvoyPatchPolicy** - Custom filters applied to Gateway

## Step-by-Step Request Flow Example

Let's trace a single request through the entire system:

### Request: "What is 2+2?"

```
STEP 1: Client sends request
  POST http://localhost:8080/v1/chat/completions
  Body: {"model": "MoM", "messages": [{"role": "user", "content": "What is 2+2?"}]}
  ↓

STEP 2: Port-forward routes to Envoy Gateway pod
  kubectl port-forward → envoy-gateway-system/envoy-xxx:80
  ↓

STEP 3: Envoy Gateway receives request
  • Checks Gateway resource for listeners
  • Finds EnvoyPatchPolicy: semantic-router-extproc-patch-policy
  • Configures ExtProc filter pointing to semantic-router-ext-proc:9002
  ↓

STEP 4: Envoy calls Semantic Router (ExtProc request phase)
  gRPC call to semantic-router-ext-proc:9002
  Sends: HTTP headers + JSON body
  ↓

STEP 5: Semantic Router processes request
  a) Parse body: model="MoM", query="What is 2+2?"
  b) Run jailbreak detection: BENIGN ✅
  c) Run category classification:
     - Input: "What is 2+2?"
     - Model: category_classifier_modernbert-base_model
     - Output: category="math", confidence=0.933
  d) Look up category in config.yaml:
     categories:
       - name: math
         model_scores:
           - model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
             score: 1.0
  e) Select model: TinyLlama/TinyLlama-1.1B-Chat-v1.0 (highest score)
  f) Rewrite request body: model="MoM" → model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  g) Return modified request to Envoy
  ↓

STEP 6: Envoy routes request using HTTPRoute
  • HTTPRoute "dynamo-worker-route" matches path /v1/*
  • Backend: Service/vllm-worker-demo-svc in dynamo-system namespace
  • ReferenceGrant allows cross-namespace reference
  ↓

STEP 7: Kubernetes Service load balances
  • Service: vllm-worker-demo-svc
  • Endpoints: [worker-pod-1:8000, worker-pod-2:8000]
  • Algorithm: Round-robin
  • Selects: worker-pod-1 (assume)
  ↓

STEP 8: Worker Pod 1 receives request
  POST /v1/chat/completions
  Body: {"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "messages": [...]}  ← rewritten!
  a) Validates: model="TinyLlama/TinyLlama-1.1B-Chat-v1.0" matches configured model ✅
  b) Simulator generates random text response
  c) Returns JSON:
     {
       "id": "chatcmpl-xxx",
       "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
       "choices": [{
         "message": {"role": "assistant", "content": "4"}
       }],
       "usage": {"prompt_tokens": 15, "completion_tokens": 1, "total_tokens": 16}
     }
  ↓

STEP 9: Response flows back through Service → HTTPRoute → Envoy
  ↓

STEP 10: Envoy calls Semantic Router (ExtProc response phase)
  gRPC call to semantic-router-ext-proc:9002
  Sends: Response headers + JSON body
  ↓

STEP 11: Semantic Router processes response
  a) Extract usage metrics: prompt_tokens=15, completion_tokens=1
  b) Log metrics: llm_usage event with model, tokens, latency
  c) Update cache: Store (query, response) for future cache hits
  d) Return response unmodified to Envoy
  ↓

STEP 12: Envoy returns response to client
  ↓

STEP 13: Client receives final response
  {
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  ← NOT "MoM" anymore!
    "choices": [...],
    "usage": {...}
  }
```

### Key Observations

1. **Model rewriting happened**: Client sent "MoM", worker received "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
2. **Classification worked**: Query classified as "math" with 93.3% confidence
3. **Load balancing worked**: One of 2 workers processed the request
4. **Metrics logged**: Semantic Router tracked tokens, latency, model used
5. **Cache updated**: Next identical query will be a cache hit

### Components

1. **Envoy Gateway** (deployed via Helm with custom values):
   - `envoy-gateway-values.yaml`: Enables `extensionApis.enableEnvoyPatchPolicy: true`
   - **Critical**: EnvoyPatchPolicy MUST be enabled for Semantic Router ExtProc to work

2. **Dynamo Platform** (deployed via Helm):
   - etcd: Distributed key-value store for coordination
   - NATS: Message queue for request routing
   - Dynamo Operator: Manages Dynamo lifecycle

3. **DynamoGraphDeployment** (`dynamo-graph-deployment.yaml`):
   - Frontend: HTTP API server on port 8000 (GPU 0)
   - VLLMPrefillWorker: Prefill phase worker (GPU 1)
   - VLLMDecodeWorker: Decode phase worker (GPU 2)
   - Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

4. **Gateway API Resources** (`gwapi-resources.yaml`):
   - HTTPRoute routing traffic to Dynamo Frontend
   - Semantic Router integration via ExtProc
   - EnvoyPatchPolicy for request/response interception

## GPU-Enabled Testing

This E2E profile uses **real vLLM inference with GPU** instead of simulators:

- **Real GPU inference** - Tests actual model loading and inference
- **Disaggregated serving** - Prefill and Decode workers on separate GPUs
- **TinyLlama model** - Uses `TinyLlama/TinyLlama-1.1B-Chat-v1.0` for fast testing
- **Full Dynamo stack** - Frontend coordinates workers via etcd/NATS

**Note:** This requires a VM with at least 3 GPUs available.

## Deployment

The E2E profile automatically deploys these resources in this order:

1. Dynamo Platform (Helm: `dynamo-crds` + `dynamo-platform`)
2. Worker Pool (2 replicas)
3. Gateway API Resources

## Testing Dynamo Functionality

With 2 worker replicas, you can test:

- **Load balancing** - Requests distributed across workers
- **Dynamic batching** - Multiple requests batched together
- **Failover** - One worker fails, traffic goes to the other
- **KV cache coordination** - Workers share KV cache state via etcd

## Production Deployment

For production with larger models, update the deployment:

```yaml
image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.6.1.post1
command: ["python3", "-m", "dynamo.vllm"]
args:
  - --model
  - meta-llama/Llama-3-8b-hf
  - --tensor-parallel-size
  - "1"
  - --enforce-eager
resources:
  requests:
    nvidia.com/gpu: 1  # Or more for tensor parallelism
```

**Note:** The E2E test uses `TinyLlama/TinyLlama-1.1B-Chat-v1.0` for faster testing.

## Manual Testing

After deployment, test the Semantic Router + Dynamo integration:

### 1. Port Forward to Envoy Gateway

```bash
kubectl port-forward -n envoy-gateway-system service/envoy-default-semantic-router-31cbd78c 8080:80
```

### 2. Send Test Request with "MoM" Model

**IMPORTANT**: Use `/v1/chat/completions` endpoint (not `/v1/completions`):

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MoM",
    "messages": [
      {
        "role": "user",
        "content": "What is 2+2?"
      }
    ]
  }'
```

### 3. Verify the Response

```json
{
  "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  // ← Rewritten from "MoM" ✅
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 54,
    "total_tokens": 69
  },
  "choices": [...]
}
```

**Success indicators:**

- ✅ Request sent with `model="MoM"`
- ✅ Response shows `model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"` (rewritten by Semantic Router)
- ✅ No "model does not exist" error

### 4. Check Semantic Router Logs

```bash
# See classification and routing decisions
kubectl logs -n vllm-semantic-router-system deployment/semantic-router -f | grep -E "category|routing_decision"
```

Expected log output:

```
Classified as category: math (confidence=0.933)
Selected model TinyLlama/TinyLlama-1.1B-Chat-v1.0 for category math with score 1.0000
routing_decision: original_model="MoM", selected_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

### 5. Verify EnvoyPatchPolicy is Enabled

```bash
kubectl get envoypatchpolicy -n default -o yaml | grep -A 5 "status:"
```

Expected status:

```yaml
status:
  conditions:
  - type: Accepted
    status: "True"  # ← Must be True!
  - type: Programmed
    status: "True"
```

If `Accepted: False` with message "EnvoyPatchPolicy is disabled", the Envoy Gateway was not deployed with the correct values file.

## Files

- `dynamo-graph-deployment.yaml` - DynamoGraphDeployment CRD (Frontend + Prefill Worker + Decode Worker with GPU)
- `rbac.yaml` - RBAC permissions for Semantic Router to access Dynamo CRDs
- `gwapi-resources.yaml` - Gateway, GatewayClass, HTTPRoute, EnvoyPatchPolicy, ReferenceGrant
- `envoy-gateway-values.yaml` - Envoy Gateway Helm values (enables EnvoyPatchPolicy)
- `README.md` - This file (you are here)

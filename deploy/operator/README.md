# Semantic Router Operator

Kubernetes operator for managing [vLLM Semantic Router](https://github.com/vllm-project/semantic-router) instances.

## Quick Start

### Prerequisites

- Kubernetes 1.25+ or OpenShift 4.12+
- `kubectl` or `oc` CLI
- Go 1.23+ (for building from source)

### Building

```bash
cd deploy/operator

# Build operator binary
make build

# Build and push Docker image
make docker-build docker-push IMG=<your-registry>/semantic-router-operator:latest
```

### Deploying the Operator

#### Option 1: Direct Install (Kubernetes)

```bash
# Install CRDs
make install

# Deploy operator
make deploy IMG=ghcr.io/vllm-project/semantic-router-operator:latest
```

#### Option 2: OpenShift OperatorHub

1. Navigate to **Operators** → **OperatorHub** in OpenShift Console
2. Search for "Semantic Router"
3. Click **Install**

#### Option 3: Manual OLM Install

```bash
# Build and push bundle
make bundle-build bundle-push BUNDLE_IMG=<your-registry>/semantic-router-operator-bundle:latest

# Build and push catalog
make catalog-build catalog-push CATALOG_IMG=<your-registry>/semantic-router-operator-catalog:latest

# Deploy to OpenShift
make openshift-deploy
```

### Deploying a SemanticRouter Instance

The operator supports multiple deployment modes and backend configurations. Choose the approach that best fits your infrastructure.

#### Quick Start Examples

For quick deployment, use one of the curated sample configurations:

```bash
# Simple standalone deployment with KServe backend (minimal config)
kubectl apply -f config/samples/vllm.ai_v1alpha1_semanticrouter_simple.yaml

# Full-featured OpenShift deployment with Routes
kubectl apply -f config/samples/vllm.ai_v1alpha1_semanticrouter_openshift.yaml

# Gateway integration mode (Istio/Envoy Gateway)
kubectl apply -f config/samples/vllm.ai_v1alpha1_semanticrouter_gateway.yaml

# Llama Stack backend discovery
kubectl apply -f config/samples/vllm.ai_v1alpha1_semanticrouter_llamastack.yaml

# OpenShift Route for external access
kubectl apply -f config/samples/vllm.ai_v1alpha1_semanticrouter_route.yaml

# Redis cache backend (production caching with persistence)
kubectl apply -f config/samples/vllm.ai_v1alpha1_semanticrouter_redis_cache.yaml

# Milvus cache backend (enterprise-grade vector database)
kubectl apply -f config/samples/vllm.ai_v1alpha1_semanticrouter_milvus_cache.yaml

# Hybrid cache backend (in-memory HNSW + persistent Milvus)
kubectl apply -f config/samples/vllm.ai_v1alpha1_semanticrouter_hybrid_cache.yaml

# mmBERT 2D Matryoshka embeddings with layer early exit
kubectl apply -f config/samples/vllm.ai_v1alpha1_semanticrouter_mmbert.yaml

# Complexity-aware routing for intelligent model selection
kubectl apply -f config/samples/vllm.ai_v1alpha1_semanticrouter_complexity.yaml
```

**Note:** All cache backend samples include the required `bert_model` or `embedding_models` configuration and will automatically download embedding models on startup. Update the Redis/Milvus hostnames to match your deployment environment.

#### Backend Discovery Types

The semantic router supports three types of backend discovery for connecting to vLLM model servers:

##### 1. KServe InferenceService Discovery

For RHOAI 3.x or standalone KServe deployments. The operator automatically discovers the predictor service created by KServe:

```yaml
vllmEndpoints:
  - name: llama3-8b-endpoint
    model: llama3-8b
    reasoningFamily: qwen3
    backend:
      type: kserve
      inferenceServiceName: llama-3-8b  # InferenceService in same namespace
    weight: 1
```

**When to use:**

- Running on Red Hat OpenShift AI (RHOAI) 3.x
- Using KServe for model serving
- Want automatic service discovery

##### 2. Llama Stack Service Discovery

Discovers Llama Stack deployments using Kubernetes label selectors:

```yaml
vllmEndpoints:
  - name: llama-405b-endpoint
    model: llama-3.3-70b-instruct
    reasoningFamily: gpt
    backend:
      type: llamastack
      discoveryLabels:
        app: llama-stack
        model: llama-3.3-70b
    weight: 1
```

**When to use:**

- Using Meta's Llama Stack for model serving
- Multiple Llama Stack services with different models
- Want label-based service discovery

##### 3. Direct Kubernetes Service

Direct connection to any Kubernetes service (vLLM, TGI, etc.):

```yaml
vllmEndpoints:
  - name: custom-vllm-endpoint
    model: deepseek-r1-distill-qwen-7b
    reasoningFamily: deepseek
    backend:
      type: service
      service:
        name: vllm-deepseek
        namespace: vllm-serving  # Can reference service in another namespace
        port: 8000
    weight: 1
```

**When to use:**

- Direct vLLM deployments
- Custom model servers with OpenAI-compatible API
- Cross-namespace service references
- Maximum control over service endpoints

## Semantic Cache Backends

The semantic router supports multiple cache backends for semantic caching, which significantly improves latency and reduces token usage by caching similar queries and their responses.

:::warning Prerequisites
The operator does **not** deploy Redis or Milvus. You must deploy these services separately in your cluster before using them as cache backends. The operator only configures the SemanticRouter to connect to your existing Redis/Milvus deployment.

**Note:** If you prefer automatic deployment of Redis/Milvus, consider using the [Helm chart](../helm/README.md), which can deploy cache backends as Helm chart dependencies.
:::

### Supported Backends

#### 1. Memory Cache (Default)

Simple in-memory cache suitable for development and small deployments.

**Characteristics:**

- No external dependencies
- Fast access
- Not persistent (cleared on restart)
- Limited by pod memory

**Configuration:**

```yaml
spec:
  config:
    semantic_cache:
      enabled: true
      backend_type: memory  # Default
      similarity_threshold: "0.8"
      max_entries: 1000
      ttl_seconds: 3600
      eviction_policy: fifo  # fifo, lru, or lfu
```

**When to use:**

- Development and testing
- Small deployments (<1000 cached queries)
- No persistence requirements

#### 2. Redis Cache

High-performance distributed cache using Redis with vector search capabilities. Requires Redis 7.0+ with RediSearch module.

**Characteristics:**

- Distributed and scalable
- Persistent storage (with AOF/RDB)
- HNSW or FLAT indexing
- Wide ecosystem support

**Configuration:**

```yaml
spec:
  config:
    semantic_cache:
      enabled: true
      backend_type: redis
      similarity_threshold: "0.85"
      ttl_seconds: 3600

      redis:
        connection:
          host: redis.default.svc.cluster.local
          port: 6379
          database: 0
          # Use Secret reference (recommended)
          password_secret_ref:
            name: redis-credentials
            key: password
          # OR use plaintext (not recommended)
          # password: "mypassword"
          timeout: 30
          tls:
            enabled: false

        index:
          name: semantic_cache_idx
          prefix: "cache:"
          vector_field:
            name: embedding
            dimension: 384  # Match your embedding model
            metric_type: COSINE
          index_type: HNSW
          params:
            M: 16
            efConstruction: 64

        search:
          topk: 1

        development:
          auto_create_index: true
          verbose_errors: true
```

**Prerequisites:**

- Redis 7.0+ with RediSearch module
- Create Kubernetes Secret for password:

```bash
kubectl create secret generic redis-credentials \
  --from-literal=password='your-redis-password'
```

**Example:** See [config/samples/vllm.ai_v1alpha1_semanticrouter_redis_cache.yaml](config/samples/vllm.ai_v1alpha1_semanticrouter_redis_cache.yaml)

**When to use:**

- Production deployments with moderate scale
- Need persistence and high availability
- Existing Redis infrastructure
- Fast in-memory performance required

#### 3. Milvus Cache

Enterprise-grade vector database for production deployments with large cache volumes. Supports advanced features like TTL, compaction, and distributed architecture.

**Characteristics:**

- Highly scalable and distributed
- Advanced indexing (HNSW, IVF, etc.)
- Built-in data lifecycle management
- High availability support

**Configuration:**

```yaml
spec:
  config:
    semantic_cache:
      enabled: true
      backend_type: milvus
      similarity_threshold: "0.90"
      ttl_seconds: 7200
      embedding_model: qwen3  # bert, qwen3, or gemma

      milvus:
        connection:
          host: milvus-standalone.default.svc.cluster.local
          port: 19530
          database: semantic_router_cache
          timeout: 30
          auth:
            enabled: true
            username: root
            password_secret_ref:
              name: milvus-credentials
              key: password

        collection:
          name: semantic_cache
          description: "Semantic cache for LLM responses"
          vector_field:
            name: embedding
            dimension: 1024  # Match your embedding model
            metric_type: IP
          index:
            type: HNSW
            params:
              M: 16
              efConstruction: 64

        search:
          params:
            ef: 64
          topk: 10
          consistency_level: Session

        performance:
          connection_pool:
            max_connections: 10
            max_idle_connections: 5
          batch:
            insert_batch_size: 100

        data_management:
          ttl:
            enabled: true
            timestamp_field: created_at
            cleanup_interval: 3600

        development:
          auto_create_collection: true
```

**Prerequisites:**

- Milvus 2.3+ (standalone or cluster)
- Create Kubernetes Secret for credentials:

```bash
kubectl create secret generic milvus-credentials \
  --from-literal=password='your-milvus-password'
```

**Example:** See [config/samples/vllm.ai_v1alpha1_semanticrouter_milvus_cache.yaml](config/samples/vllm.ai_v1alpha1_semanticrouter_milvus_cache.yaml)

**When to use:**

- Large-scale production deployments
- Need advanced vector search capabilities
- Require data lifecycle management (TTL, compaction)
- High availability and scalability requirements

#### 4. Hybrid Cache

Combines in-memory HNSW index with persistent Milvus storage for optimal performance and durability.

**Characteristics:**

- Fast in-memory search with HNSW
- Persistent storage in Milvus
- Best of both worlds
- Automatic synchronization

**Configuration:**

```yaml
spec:
  config:
    semantic_cache:
      enabled: true
      backend_type: hybrid
      similarity_threshold: "0.85"
      ttl_seconds: 3600
      max_entries: 5000
      eviction_policy: lru

      # HNSW in-memory configuration
      hnsw:
        use_hnsw: true
        hnsw_m: 32
        hnsw_ef_construction: 128
        max_memory_entries: 5000

      # Milvus persistent storage (same config as milvus backend)
      milvus:
        connection:
          host: milvus-standalone.default.svc.cluster.local
          port: 19530
          # ... rest of milvus config
```

**Example:** See [config/samples/vllm.ai_v1alpha1_semanticrouter_hybrid_cache.yaml](config/samples/vllm.ai_v1alpha1_semanticrouter_hybrid_cache.yaml)

**When to use:**

- Need fastest possible cache lookups
- Require persistence and durability
- Willing to trade memory for performance
- High-throughput production deployments

### Cache Configuration Reference

For detailed configuration options, use:

```bash
# Explore Redis cache configuration
kubectl explain semanticrouter.spec.config.semantic_cache.redis

# Explore Milvus cache configuration
kubectl explain semanticrouter.spec.config.semantic_cache.milvus

# Explore HNSW configuration
kubectl explain semanticrouter.spec.config.semantic_cache.hnsw
```

### Embedding Models

The operator supports advanced embedding models through the unified `embedding_models` configuration. These models provide semantic understanding for caching, classification, and routing decisions.

#### Available Models

1. **BERT (Legacy)** - 384 dimensions
   - Lightweight, fast, good for general use
   - Legacy configuration via `bert_model` section

2. **Qwen3-Embedding** - 1024 dimensions, 32K context
   - High-quality semantic understanding
   - Best for: Complex queries, research documents, detailed analysis
   - Use case: Production deployments requiring maximum accuracy

3. **EmbeddingGemma** - 768 dimensions, 8K context
   - Balanced performance and accuracy
   - Best for: Fast performance with good quality
   - Use case: Real-time applications, high-throughput scenarios

4. **mmBERT 2D Matryoshka** - 64-768 dimensions, multilingual
   - Adaptive quality/speed trade-offs via layer early exit
   - Layer 3: ~7x speedup, Layer 6: ~3.6x speedup, Layer 11: ~2x speedup, Layer 22: full accuracy
   - Dimension reduction: 64, 128, 256, 512, 768
   - Best for: Multilingual deployments, flexible performance tuning
   - Use case: Multi-language support, budget-constrained environments

#### Configuration Examples

**Using mmBERT with layer early exit:**

```yaml
spec:
  config:
    embedding_models:
      mmbert_model_path: "models/mmbert-embedding"
      use_cpu: true

      hnsw_config:
        model_type: "mmbert"
        target_layer: 6      # Balanced speed/quality (3.6x speedup)
        target_dimension: 256  # Reduced dimension for faster search
        preload_embeddings: true
        enable_soft_matching: true
        min_score_threshold: "0.5"

    semantic_cache:
      enabled: true
      embedding_model: "mmbert"
      similarity_threshold: "0.85"
```

**Using Qwen3 with Redis cache:**

```yaml
spec:
  config:
    embedding_models:
      qwen3_model_path: "models/qwen3-embedding"
      use_cpu: true

    semantic_cache:
      enabled: true
      backend_type: "redis"
      embedding_model: "qwen3"
      redis:
        index:
          vector_field:
            dimension: 1024  # Match Qwen3 dimension
```

**Using Gemma with Milvus cache:**

```yaml
spec:
  config:
    embedding_models:
      gemma_model_path: "models/gemma-embedding"
      use_cpu: true

    semantic_cache:
      enabled: true
      backend_type: "milvus"
      embedding_model: "gemma"
      milvus:
        collection:
          vector_field:
            dimension: 768  # Match Gemma dimension
```

**Dimension Reference:**

| Model | Dimensions | Context | Performance |
|-------|-----------|---------|-------------|
| BERT | 384 | 512 | Fast |
| Gemma | 768 | 8K | Balanced |
| Qwen3 | 1024 | 32K | High Quality |
| mmBERT | 64-768 (adaptive) | Varies | Tunable |

**Important:** Ensure `dimension` in cache config matches your chosen embedding model's dimension.

### Migration Path

Migrating from memory cache to Redis or Milvus is straightforward:

1. Deploy Redis or Milvus in your cluster
2. Create the credentials Secret
3. Update SemanticRouter CR with new backend configuration
4. Apply the changes - operator will perform rolling update

The cache will be empty after migration but will populate naturally as queries are processed.

## Complexity-Aware Routing

Route queries to different models based on complexity classification using few-shot learning. This enables cost optimization by sending simple queries to fast models and complex queries to powerful models.

### Configuration

```yaml
spec:
  # Configure multiple backends with different capabilities
  vllmEndpoints:
    - name: llama-8b-fast
      model: llama3-8b
      reasoningFamily: qwen3
      backend:
        type: kserve
        inferenceServiceName: llama-3-8b
      weight: 2  # Prefer for simple queries

    - name: llama-70b-reasoning
      model: llama3-70b
      reasoningFamily: deepseek
      backend:
        type: kserve
        inferenceServiceName: llama-3-70b
      weight: 1  # Use for complex queries

  config:
    # Define complexity rules with few-shot examples
    complexity_rules:
      - name: "code-complexity"
        description: "Classify coding tasks by complexity"
        threshold: "0.3"  # Lower threshold works better for embedding-based similarity

        # Examples of complex coding tasks
        hard:
          candidates:
            - "Implement a distributed lock manager with leader election"
            - "Design a database migration system with rollback support"
            - "Create a compiler optimization pass for loop unrolling"

        # Examples of simple coding tasks
        easy:
          candidates:
            - "Write a function to reverse a string"
            - "Create a class to represent a rectangle"
            - "Implement a simple counter with increment/decrement"

      - name: "reasoning-complexity"
        description: "Classify reasoning tasks"
        threshold: "0.3"  # Lower threshold works better for embedding-based similarity

        hard:
          candidates:
            - "Analyze geopolitical implications of renewable energy adoption"
            - "Evaluate ethical considerations of AI in healthcare"

        easy:
          candidates:
            - "What is the capital of France?"
            - "How many days are in a week?"
```

### How It Works

1. **Query Analysis**: Incoming query is compared against `hard` and `easy` candidate examples using embedding similarity
2. **Complexity Scoring**: Similarity scores determine if query is closer to hard or easy examples
3. **Signal Generation**: Outputs classification signals: `{rule-name}:hard`, `{rule-name}:easy`, or `{rule-name}:medium`
4. **Routing Decision**: Router uses complexity signals to select appropriate backend model
5. **Cost Optimization**: Simple queries → fast/cheap models, Complex queries → powerful/expensive models

### Advanced: Conditional Rules with Composer

Apply rules conditionally based on other signals (e.g., domain, language):

```yaml
complexity_rules:
  - name: "medical-complexity"
    description: "Classify medical queries (only for medical domain)"
    threshold: "0.7"

    hard:
      candidates:
        - "Differential diagnosis for chest pain with dyspnea"
        - "Treatment protocol for multi-drug resistant tuberculosis"

    easy:
      candidates:
        - "What is the normal body temperature?"
        - "What are common symptoms of a cold?"

    # Only apply this rule if domain:medical signal is present
    composer:
      operator: "AND"
      conditions:
        - type: "domain"
          name: "medical"
```

**Example:** See [config/samples/vllm.ai_v1alpha1_semanticrouter_complexity.yaml](config/samples/vllm.ai_v1alpha1_semanticrouter_complexity.yaml)

## Verification

```bash
# Check status
kubectl get semanticrouter test-router

# Check deployment
kubectl get deployment test-router

# Check logs
kubectl logs -l app.kubernetes.io/instance=test-router

# Port forward to access locally
kubectl port-forward svc/test-router 50051:50051 8080:8080
```

## Development

```bash
# Install CRDs
make install

# Run operator locally (outside cluster)
make run

# Run tests
make test

# Generate code after API changes
make manifests generate
```

## Deployment Modes

The operator supports two deployment modes:

### Standalone Mode (Default)

Deploys semantic router with an **Envoy sidecar container** that acts as an ingress gateway. Envoy forwards requests to the semantic router via ExtProc gRPC protocol.

**Architecture:**

```
Client → Service (port 8080) → Envoy Sidecar → ExtProc (semantic router) → vLLM Backend
```

**When to use:**

- Simple deployments without existing service mesh
- Testing and development
- Self-contained deployment with minimal dependencies

**Configuration:**

```yaml
spec:
  # No gateway configuration - defaults to standalone mode
  service:
    type: ClusterIP
    api:
      port: 8080  # Client traffic enters here
      targetPort: 8080  # Envoy ingress port
    grpc:
      port: 50051  # ExtProc communication
      targetPort: 50051
```

### Gateway Integration Mode

Reuses an **existing Gateway** (Istio, Envoy Gateway, etc.) and creates an HTTPRoute. The operator skips deploying the Envoy sidecar container.

**Architecture:**

```
Client → Gateway → HTTPRoute → Service (port 8080) → Semantic Router API → vLLM Backend
```

**When to use:**

- Existing Istio or Envoy Gateway deployment
- Centralized ingress management
- Multi-tenancy with shared gateway
- Advanced traffic management (circuit breaking, retries, rate limiting)

**Configuration:**

```yaml
spec:
  gateway:
    existingRef:
      name: istio-ingressgateway  # Or your Envoy Gateway name
      namespace: istio-system

  # Service only needs API port in gateway mode
  service:
    type: ClusterIP
    api:
      port: 8080
      targetPort: 8080
```

**Operator behavior in gateway mode:**

1. Creates HTTPRoute resource pointing to the specified Gateway
2. Skips Envoy sidecar container in pod spec
3. Sets `status.gatewayMode: "gateway-integration"`
4. Semantic router operates in pure API mode (no ExtProc)

## OpenShift Routes

For OpenShift deployments, the operator can create Routes for external access with TLS termination:

```yaml
spec:
  openshift:
    routes:
      enabled: true
      hostname: semantic-router.apps.openshift.example.com  # Optional - auto-generated if omitted
      tls:
        termination: edge  # edge, passthrough, or reencrypt
        insecureEdgeTerminationPolicy: Redirect  # Redirect HTTP to HTTPS
```

**TLS termination options:**

- **edge**: TLS terminates at Route, plain HTTP to backend (recommended)
- **passthrough**: TLS passthrough to backend (requires backend TLS)
- **reencrypt**: TLS terminates at Route, re-encrypts to backend

**When to use:**

- Running on OpenShift 4.x
- Need external access without configuring Ingress
- Want auto-generated hostnames
- Require OpenShift-native TLS management

**Operator behavior:**

1. Creates OpenShift Route resource
2. Configures TLS based on spec
3. Sets `status.openshiftFeatures.routesEnabled: true`
4. Sets `status.openshiftFeatures.routeHostname` with actual hostname

## Configuration

For production deployments, enable persistence:

```yaml
spec:
  persistence:
    enabled: true
    size: 10Gi
    storageClassName: "fast-ssd"  # Adjust for your cluster
```

The operator validates that the specified StorageClass exists before creating the PVC.

For high availability:

```yaml
spec:
  replicas: 3
  autoscaling:
    enabled: true
    minReplicas: 3
    maxReplicas: 20
    targetCPUUtilizationPercentage: 70
```

## Troubleshooting

```bash
# Check operator logs
kubectl logs -n semantic-router-operator-system \
  -l app.kubernetes.io/name=semantic-router-operator

# Check resource status
kubectl describe semanticrouter test-router

# Check events
kubectl get events --sort-by='.lastTimestamp'
```

## Documentation

Full documentation: https://vllm-semantic-router.com

## License

Apache License 2.0

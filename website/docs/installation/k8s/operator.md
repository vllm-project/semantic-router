---
sidebar_position: 8
---

# Kubernetes Operator

The Semantic Router Operator provides a Kubernetes-native way to deploy and manage vLLM Semantic Router instances using Custom Resource Definitions (CRDs). It simplifies deployment, configuration, and lifecycle management across Kubernetes and OpenShift platforms.

## Features

- **🚀 Declarative Deployment**: Define semantic router instances using Kubernetes CRDs
- **🔄 Automatic Configuration**: Generates and manages ConfigMaps for semantic router configuration
- **📦 Persistent Storage**: Manages PVCs for ML model storage with automatic lifecycle
- **🔐 Platform Detection**: Automatically detects and configures for OpenShift or standard Kubernetes
- **📊 Built-in Observability**: Metrics, tracing, and monitoring support out of the box
- **🎯 Production Features**: HPA, ingress, service mesh integration, and pod disruption budgets
- **🛡️ Secure by Default**: Drops all capabilities, prevents privilege escalation

## Prerequisites

- Kubernetes 1.24+ or OpenShift 4.12+
- `kubectl` or `oc` CLI configured
- Cluster admin access (for CRD installation)

## Installation

### Option 1: Using Kustomize (Standard Kubernetes)

```bash
# Clone the repository
git clone https://github.com/vllm-project/semantic-router
cd semantic-router/deploy/operator

# Install CRDs
make install

# Deploy the operator
make deploy IMG=ghcr.io/vllm-project/semantic-router-operator:latest
```

Verify the operator is running:

```bash
kubectl get pods -n semantic-router-operator-system
```

### Option 2: Using OLM (OpenShift)

For OpenShift deployments using Operator Lifecycle Manager:

```bash
cd semantic-router/deploy/operator

# Build and push to your registry (Quay, internal registry, etc.)
podman login quay.io
make podman-build IMG=quay.io/<your-org>/semantic-router-operator:latest
make podman-push IMG=quay.io/<your-org>/semantic-router-operator:latest

# Deploy using OLM
make openshift-deploy
```

## Deploy Your First Router

### Quick Start with Sample Configurations

Choose a pre-configured sample based on your infrastructure:

```bash
# Simple standalone deployment with KServe backend
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/operator/config/samples/vllm.ai_v1alpha1_semanticrouter_simple.yaml

# Full-featured OpenShift deployment with Routes
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/operator/config/samples/vllm.ai_v1alpha1_semanticrouter_openshift.yaml

# Gateway integration mode (Istio/Envoy Gateway)
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/operator/config/samples/vllm.ai_v1alpha1_semanticrouter_gateway.yaml

# Llama Stack backend discovery
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/operator/config/samples/vllm.ai_v1alpha1_semanticrouter_llamastack.yaml
```

### Custom Configuration

Create a `my-router.yaml` file:

```yaml
apiVersion: vllm.ai/v1alpha1
kind: SemanticRouter
metadata:
  name: my-router
  namespace: default
spec:
  replicas: 2

  image:
    repository: ghcr.io/vllm-project/semantic-router/extproc
    tag: latest

  # Configure vLLM backend endpoints
  vllmEndpoints:
    # KServe InferenceService (RHOAI 3.x)
    - name: llama3-8b-endpoint
      model: llama3-8b
      reasoningFamily: qwen3
      backend:
        type: kserve
        inferenceServiceName: llama-3-8b
      weight: 1

  resources:
    limits:
      memory: "7Gi"
      cpu: "2"
    requests:
      memory: "3Gi"
      cpu: "1"

  persistence:
    enabled: true
    size: 10Gi
    storageClassName: "standard"

  config:
    bert_model:
      model_id: "models/mom-embedding-light"
      threshold: "0.6"
      use_cpu: true

    semantic_cache:
      enabled: true
      backend_type: "memory"
      max_entries: 1000
      ttl_seconds: 3600

    tools:
      enabled: true
      top_k: 3
      similarity_threshold: "0.2"

    prompt_guard:
      enabled: true
      threshold: "0.7"

  toolsDb:
    - tool:
        type: "function"
        function:
          name: "get_weather"
          description: "Get weather information for a location"
          parameters:
            type: "object"
            properties:
              location:
                type: "string"
                description: "City and state, e.g. San Francisco, CA"
            required: ["location"]
      description: "Weather information tool"
      category: "weather"
      tags: ["weather", "temperature"]
```

Apply the configuration:

```bash
kubectl apply -f my-router.yaml
```

## Verify Deployment

```bash
# Check the SemanticRouter resource
kubectl get semanticrouter my-router

# Check created resources
kubectl get deployment,service,configmap -l app.kubernetes.io/instance=my-router

# View status
kubectl describe semanticrouter my-router

# View logs
kubectl logs -f deployment/my-router
```

Expected output:

```
NAME                        PHASE     REPLICAS   READY   AGE
semanticrouter.vllm.ai/my-router   Running   2          2       5m
```

## Backend Discovery Types

The operator supports three types of backend discovery for connecting semantic router to vLLM model servers. Choose the type that matches your infrastructure.

### KServe InferenceService Discovery

For RHOAI 3.x or standalone KServe deployments. The operator automatically discovers the predictor service created by KServe.

```yaml
spec:
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

**How it works:**

- Discovers the predictor service: `{inferenceServiceName}-predictor`
- Uses port 8443 (KServe default HTTPS port)
- Works in the same namespace as SemanticRouter

### Llama Stack Service Discovery

Discovers Llama Stack deployments using Kubernetes label selectors.

```yaml
spec:
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

**How it works:**

- Lists services matching the label selector
- Uses first matching service if multiple found
- Extracts port from service definition

### Direct Kubernetes Service

Direct connection to any Kubernetes service (vLLM, TGI, etc.).

```yaml
spec:
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

**How it works:**

- Connects to specified service directly
- No discovery - uses explicit configuration
- Supports cross-namespace references

### Multiple Backends

You can configure multiple backends with load balancing weights:

```yaml
spec:
  vllmEndpoints:
    # KServe backend
    - name: llama3-8b
      model: llama3-8b
      reasoningFamily: qwen3
      backend:
        type: kserve
        inferenceServiceName: llama-3-8b
      weight: 2  # Higher weight = more traffic

    # Direct service backend
    - name: qwen-7b
      model: qwen2.5-7b
      reasoningFamily: qwen3
      backend:
        type: service
        service:
          name: vllm-qwen
          port: 8000
      weight: 1
```

## Deployment Modes

The operator supports two deployment modes with different architectures.

### Standalone Mode (Default)

Deploys semantic router with an **Envoy sidecar container** that acts as an ingress gateway.

**Architecture:**

```
Client → Service (8080) → Envoy Sidecar → ExtProc gRPC → Semantic Router → vLLM
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

**Operator behavior:**

- Deploys pod with two containers: semantic router + Envoy sidecar
- Envoy handles ingress and forwards to semantic router via ExtProc gRPC
- Status shows `gatewayMode: "standalone"`

### Gateway Integration Mode

Reuses an **existing Gateway** (Istio, Envoy Gateway, etc.) and creates an HTTPRoute.

**Architecture:**

```
Client → Gateway (Istio/Envoy) → HTTPRoute → Service (8080) → Semantic Router API → vLLM
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

**Operator behavior:**

1. Creates HTTPRoute resource pointing to the specified Gateway
2. Skips Envoy sidecar container in pod spec
3. Sets `status.gatewayMode: "gateway-integration"`
4. Semantic router operates in pure API mode (no ExtProc)

**Example:** See [`vllm.ai_v1alpha1_semanticrouter_gateway.yaml`](https://github.com/vllm-project/semantic-router/blob/main/deploy/operator/config/samples/vllm.ai_v1alpha1_semanticrouter_gateway.yaml)

## OpenShift Routes

For OpenShift deployments, the operator can create Routes for external access with TLS termination.

### Basic OpenShift Route

```yaml
spec:
  openshift:
    routes:
      enabled: true
      hostname: semantic-router.apps.openshift.example.com  # Optional - auto-generated if omitted
      tls:
        termination: edge  # TLS terminates at Route, plain HTTP to backend
        insecureEdgeTerminationPolicy: Redirect  # Redirect HTTP to HTTPS
```

### TLS Termination Options

- **edge** (recommended): TLS terminates at Route, plain HTTP to backend
- **passthrough**: TLS passthrough to backend (requires backend TLS)
- **reencrypt**: TLS terminates at Route, re-encrypts to backend

### When to Use OpenShift Routes

- Running on OpenShift 4.x
- Need external access without configuring Ingress
- Want auto-generated hostnames
- Require OpenShift-native TLS management

### Status Information

After creating a Route, check the status:

```bash
kubectl get semanticrouter my-router -o jsonpath='{.status.openshiftFeatures}'
```

Output:

```json
{
  "routesEnabled": true,
  "routeHostname": "semantic-router-default.apps.openshift.example.com"
}
```

**Example:** See [`vllm.ai_v1alpha1_semanticrouter_route.yaml`](https://github.com/vllm-project/semantic-router/blob/main/deploy/operator/config/samples/vllm.ai_v1alpha1_semanticrouter_route.yaml)

## Choosing Your Configuration

Use this decision tree to select the right configuration:

```
┌─ Need to run on OpenShift?
│  ├─ YES → Use openshift sample (Routes + KServe/service backends)
│  └─ NO ↓
│
├─ Have existing Gateway (Istio/Envoy)?
│  ├─ YES → Use gateway sample (Gateway integration mode)
│  └─ NO ↓
│
├─ Using Meta Llama Stack?
│  ├─ YES → Use llamastack sample
│  └─ NO ↓
│
└─ Simple deployment → Use simple sample (standalone mode)
```

**Backend choice:**

```
┌─ Running RHOAI 3.x or KServe?
│  ├─ YES → Use KServe backend type
│  └─ NO ↓
│
├─ Using Meta Llama Stack?
│  ├─ YES → Use llamastack backend type
│  └─ NO ↓
│
└─ Have direct vLLM service? → Use service backend type
```

## Architecture

The operator manages a complete stack of resources for each SemanticRouter:

```
┌─────────────────────────────────────────────────────┐
│              SemanticRouter CR                       │
│  apiVersion: vllm.ai/v1alpha1                       │
│  kind: SemanticRouter                               │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  Operator Controller │
        │  - Watches CR        │
        │  - Reconciles state  │
        │  - Platform detection│
        └─────────┬────────────┘
                  │
     ┌────────────┼────────────┬──────────────┐
     ▼            ▼            ▼              ▼
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│Deployment│  │ Service │  │ConfigMap│  │   PVC   │
│         │  │ - gRPC  │  │ - config│  │ - models│
│         │  │ - API   │  │ - tools │  │         │
│         │  │ - metrics│  │         │  │         │
└─────────┘  └─────────┘  └─────────┘  └─────────┘
```

**Managed Resources:**

- **Deployment**: Runs semantic router pods with configurable replicas
- **Service**: Exposes gRPC (50051), HTTP API (8080), and metrics (9190)
- **ConfigMap**: Contains semantic router configuration and tools database
- **ServiceAccount**: For RBAC (optional, created when specified)
- **PersistentVolumeClaim**: For ML model storage (optional, when persistence enabled)
- **HorizontalPodAutoscaler**: For auto-scaling (optional, when autoscaling enabled)
- **Ingress**: For external access (optional, when ingress enabled)

## Platform Detection and Security

The operator automatically detects the platform and configures security contexts appropriately.

### OpenShift Platform

When running on OpenShift, the operator:

- **Detects**: Checks for `route.openshift.io` API resources
- **Security Context**: Does NOT set `runAsUser`, `runAsGroup`, or `fsGroup`
- **Rationale**: Lets OpenShift SCCs assign UIDs/GIDs from the namespace's allowed range
- **Compatible with**: `restricted` SCC (default) and custom SCCs

### Standard Kubernetes

When running on standard Kubernetes, the operator:

- **Security Context**: Sets `runAsUser: 1000`, `fsGroup: 1000`, `runAsNonRoot: true`
- **Rationale**: Provides secure defaults for pod security policies/standards

### Both Platforms

Regardless of platform:

- Drops ALL capabilities (`drop: [ALL]`)
- Prevents privilege escalation (`allowPrivilegeEscalation: false`)
- No special permissions or SCCs required beyond defaults

### Override Security Context

You can override automatic security contexts in your CR:

```yaml
spec:
  # Container security context
  securityContext:
    runAsNonRoot: true
    runAsUser: 2000
    allowPrivilegeEscalation: false
    capabilities:
      drop:
        - ALL

  # Pod security context
  podSecurityContext:
    runAsNonRoot: true
    runAsUser: 2000
    fsGroup: 2000
```

:::caution OpenShift Note
When running on OpenShift, it's recommended to omit `runAsUser` and `fsGroup` and let SCCs handle UID/GID assignment automatically.
:::

## Configuration Reference

### Image Configuration

```yaml
spec:
  image:
    repository: ghcr.io/vllm-project/semantic-router/extproc
    tag: latest
    pullPolicy: IfNotPresent
    imageRegistry: ""  # Optional: custom registry prefix

  # Optional: Image pull secrets
  imagePullSecrets:
    - name: ghcr-secret
```

### Service Configuration

```yaml
spec:
  service:
    type: ClusterIP  # or NodePort, LoadBalancer

    grpc:
      port: 50051
      targetPort: 50051

    api:
      port: 8080
      targetPort: 8080

    metrics:
      enabled: true
      port: 9190
      targetPort: 9190
```

### Persistence Configuration

```yaml
spec:
  persistence:
    enabled: true
    storageClassName: "standard"  # Adjust for your cluster
    accessMode: ReadWriteOnce
    size: 10Gi

    # Optional: Use existing PVC
    existingClaim: "my-existing-pvc"

    # Optional: PVC annotations
    annotations:
      backup.velero.io/backup-volumes: "models"
```

:::info Storage Validation
The operator validates that the specified StorageClass exists before creating the PVC. If `storageClassName` is omitted, the cluster's default StorageClass is used.
:::

**Storage Class Examples:**

- **AWS EKS**: `gp3-csi`, `gp2`
- **GKE**: `standard`, `premium-rwo`
- **Azure AKS**: `managed`, `managed-premium`
- **OpenShift**: `gp3-csi`, `thin`, `ocs-storagecluster-ceph-rbd`

### Semantic Router Configuration

Full semantic router configuration is embedded in the CR. See the complete example in [`deploy/operator/config/samples/vllm_v1alpha1_semanticrouter.yaml`](https://github.com/vllm-project/semantic-router/blob/main/deploy/operator/config/samples/vllm_v1alpha1_semanticrouter.yaml).

Key configuration sections:

```yaml
spec:
  config:
    # BERT model for embeddings
    bert_model:
      model_id: "models/mom-embedding-light"
      threshold: 0.6
      use_cpu: true

    # Semantic cache
    semantic_cache:
      enabled: true
      backend_type: "memory"  # or "milvus"
      similarity_threshold: 0.8
      max_entries: 1000
      ttl_seconds: 3600
      eviction_policy: "fifo"

    # Tools auto-selection
    tools:
      enabled: true
      top_k: 3
      similarity_threshold: 0.2
      tools_db_path: "config/tools_db.json"
      fallback_to_empty: true

    # Prompt guard (jailbreak detection)
    prompt_guard:
      enabled: true
      model_id: "models/mom-jailbreak-classifier"
      threshold: 0.7
      use_cpu: true

    # Classifiers
    classifier:
      category_model:
        model_id: "models/lora_intent_classifier_bert-base-uncased_model"
        threshold: 0.6
        use_cpu: true
      pii_model:
        model_id: "models/pii_classifier_modernbert-base_presidio_token_model"
        threshold: 0.7
        use_cpu: true

    # Reasoning configuration per model family
    reasoning_families:
      deepseek:
        type: "chat_template_kwargs"
        parameter: "thinking"
      qwen3:
        type: "chat_template_kwargs"
        parameter: "enable_thinking"
      gpt:
        type: "reasoning_effort"
        parameter: "reasoning_effort"

    # API batch classification
    api:
      batch_classification:
        max_batch_size: 100
        concurrency_threshold: 5
        max_concurrency: 8
        metrics:
          enabled: true
          detailed_goroutine_tracking: true
          sample_rate: 1.0

    # Observability
    observability:
      tracing:
        enabled: false
        provider: "opentelemetry"
        exporter:
          type: "otlp"
          endpoint: "jaeger:4317"
```

### Tools Database

Define available tools for auto-selection:

```yaml
spec:
  toolsDb:
    - tool:
        type: "function"
        function:
          name: "search_web"
          description: "Search the web for information"
          parameters:
            type: "object"
            properties:
              query:
                type: "string"
                description: "Search query"
            required: ["query"]
      description: "Search the internet, web search, find information online"
      category: "search"
      tags: ["search", "web", "internet"]

    - tool:
        type: "function"
        function:
          name: "calculate"
          description: "Perform mathematical calculations"
          parameters:
            type: "object"
            properties:
              expression:
                type: "string"
            required: ["expression"]
      description: "Calculate mathematical expressions"
      category: "math"
      tags: ["math", "calculation"]
```

### Autoscaling (HPA)

```yaml
spec:
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 10
    targetCPUUtilizationPercentage: 70
    targetMemoryUtilizationPercentage: 80
```

### Ingress Configuration

```yaml
spec:
  ingress:
    enabled: true
    className: "nginx"  # or "haproxy", "traefik", etc.
    annotations:
      cert-manager.io/cluster-issuer: "letsencrypt-prod"
    hosts:
      - host: router.example.com
        paths:
          - path: /
            pathType: Prefix
            servicePort: 8080
    tls:
      - secretName: router-tls
        hosts:
          - router.example.com
```

## Production Deployment

### High Availability Setup

```yaml
apiVersion: vllm.ai/v1alpha1
kind: SemanticRouter
metadata:
  name: prod-router
spec:
  replicas: 3

  # Anti-affinity for spreading across nodes
  affinity:
    podAntiAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        - labelSelector:
            matchLabels:
              app.kubernetes.io/instance: prod-router
          topologyKey: kubernetes.io/hostname

  # Autoscaling
  autoscaling:
    enabled: true
    minReplicas: 3
    maxReplicas: 20
    targetCPUUtilizationPercentage: 70

  # Production resources
  resources:
    limits:
      memory: "10Gi"
      cpu: "4"
    requests:
      memory: "5Gi"
      cpu: "2"

  # Strict probes
  livenessProbe:
    enabled: true
    initialDelaySeconds: 60
    periodSeconds: 30
    failureThreshold: 3

  readinessProbe:
    enabled: true
    initialDelaySeconds: 30
    periodSeconds: 10
    failureThreshold: 3
```

### Pod Disruption Budget

Create a PDB to ensure availability during updates:

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: prod-router-pdb
spec:
  maxUnavailable: 1
  selector:
    matchLabels:
      app.kubernetes.io/instance: prod-router
```

### Resource Allocation Guidelines

| Workload Type | Memory Request | CPU Request | Memory Limit | CPU Limit |
|---------------|----------------|-------------|--------------|-----------|
| Development   | 1Gi            | 500m        | 2Gi          | 1         |
| Staging       | 3Gi            | 1           | 7Gi          | 2         |
| Production    | 5Gi            | 2           | 10Gi         | 4         |

## Monitoring and Observability

### Metrics

Prometheus metrics are exposed on port 9190:

```bash
# Port-forward to access metrics locally
kubectl port-forward svc/my-router 9190:9190

# View metrics
curl http://localhost:9190/metrics
```

**Key Metrics:**

- `semantic_router_request_duration_seconds` - Request latency
- `semantic_router_cache_hit_total` - Cache hit rate
- `semantic_router_classification_duration_seconds` - Classification latency
- `semantic_router_tokens_total` - Token usage
- `semantic_router_reasoning_requests_total` - Reasoning mode usage

### ServiceMonitor (Prometheus Operator)

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: semantic-router-metrics
spec:
  selector:
    matchLabels:
      app.kubernetes.io/instance: my-router
  endpoints:
    - port: metrics
      interval: 30s
      path: /metrics
```

### Distributed Tracing

Enable OpenTelemetry tracing:

```yaml
spec:
  config:
    observability:
      tracing:
        enabled: true
        provider: "opentelemetry"
        exporter:
          type: "otlp"
          endpoint: "jaeger-collector:4317"
          insecure: true
        sampling:
          type: "always_on"
          rate: 1.0
```

## Troubleshooting

### Common Issues

#### Backend Discovery Failures

**Symptom:** "No backends found" or "Failed to discover backend" in logs

**For KServe backends:**

```bash
# Check InferenceService exists and is ready
kubectl get inferenceservice llama-3-8b

# Check predictor service was created by KServe
kubectl get service llama-3-8b-predictor

# Verify InferenceService status
kubectl describe inferenceservice llama-3-8b
```

**For Llama Stack backends:**

```bash
# Verify services exist with correct labels
kubectl get services -l app=llama-stack,model=llama-3.3-70b

# Check service labels match discoveryLabels in CR
kubectl get service <service-name> -o jsonpath='{.metadata.labels}'
```

**For direct service backends:**

```bash
# Verify service exists in specified namespace
kubectl get service vllm-deepseek -n vllm-serving

# Check service has ports defined
kubectl get service vllm-deepseek -n vllm-serving -o jsonpath='{.spec.ports[0]}'
```

#### Gateway Integration Issues

**Symptom:** HTTPRoute not created or traffic not reaching semantic router

```bash
# Verify Gateway exists
kubectl get gateway istio-ingressgateway -n istio-system

# Check HTTPRoute was created
kubectl get httproute -l app.kubernetes.io/instance=my-router

# Verify Gateway supports HTTPRoute (Gateway API v1)
kubectl get gateway istio-ingressgateway -n istio-system -o yaml | grep -A5 listeners

# Check operator status
kubectl get semanticrouter my-router -o jsonpath='{.status.gatewayMode}'
# Should show: "gateway-integration"
```

#### OpenShift Route Issues

**Symptom:** Route not created on OpenShift

```bash
# Verify running on OpenShift cluster
kubectl api-resources | grep route.openshift.io

# Check if Route was created
kubectl get route -l app.kubernetes.io/instance=my-router

# Check operator detected OpenShift
kubectl logs -n semantic-router-operator-system \
  deployment/semantic-router-operator-controller-manager \
  | grep -i "openshift\|route"

# Verify Route status
kubectl get semanticrouter my-router -o jsonpath='{.status.openshiftFeatures}'
```

#### Pod stuck in `ImagePullBackOff`

```bash
# Check image pull secrets
kubectl describe pod <pod-name>

# Create image pull secret
kubectl create secret docker-registry ghcr-secret \
  --docker-server=ghcr.io \
  --docker-username=<username> \
  --docker-password=<personal-access-token>

# Add to SemanticRouter
spec:
  imagePullSecrets:
    - name: ghcr-secret
```

#### PVC stuck in `Pending`

```bash
# Check storage class exists
kubectl get storageclass

# Check PVC events
kubectl describe pvc my-router-models

# Update storage class in CR
spec:
  persistence:
    storageClassName: "your-available-storage-class"
```

#### Models not downloading

```bash
# Check if HF token secret exists
kubectl get secret hf-token-secret

# Create HF token secret
kubectl create secret generic hf-token-secret \
  --from-literal=token=hf_xxxxxxxxxxxxx

# Add to SemanticRouter CR
spec:
  env:
    - name: HF_TOKEN
      valueFrom:
        secretKeyRef:
          name: hf-token-secret
          key: token
```

#### Operator not detecting platform correctly

```bash
# Check operator logs for platform detection
kubectl logs -n semantic-router-operator-system \
  deployment/semantic-router-operator-controller-manager \
  | grep -i "platform\|openshift"

# Should see one of:
# "Detected OpenShift platform - will use OpenShift-compatible security contexts"
# "Detected standard Kubernetes platform - will use standard security contexts"
```

## Development

### Local Development

```bash
cd deploy/operator

# Run tests
make test

# Generate CRDs and code
make generate
make manifests

# Build operator binary
make build

# Run locally against your kubeconfig
make run
```

### Testing with kind

```bash
# Create kind cluster
kind create cluster --name operator-test

# Build and load image
make docker-build IMG=semantic-router-operator:dev
kind load docker-image semantic-router-operator:dev --name operator-test

# Deploy
make install
make deploy IMG=semantic-router-operator:dev

# Create test instance
kubectl apply -f config/samples/vllm_v1alpha1_semanticrouter.yaml
```

## Next Steps

- [Configure semantic router features](../configuration)
- [Set up monitoring and observability](../../tutorials/observability/dashboard)
- [Explore other deployment options](../installation.md)
- [Join the community](../../community/overview)

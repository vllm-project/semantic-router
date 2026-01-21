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
```

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

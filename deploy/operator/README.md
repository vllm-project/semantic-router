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

Create a `semantic-router.yaml` file:

```yaml
apiVersion: vllm.ai/v1alpha1
kind: SemanticRouter
metadata:
  name: test-router
  namespace: default
spec:
  replicas: 1
  image:
    repository: ghcr.io/vllm-project/semantic-router/extproc
    tag: latest
    pullPolicy: IfNotPresent
  service:
    type: ClusterIP
    grpc:
      port: 50051
      targetPort: 50051
    api:
      port: 8080
      targetPort: 8080
  resources:
    limits:
      memory: "2Gi"
      cpu: "1"
    requests:
      memory: "1Gi"
      cpu: "500m"
  persistence:
    enabled: false
  startupProbe:
    enabled: true
    periodSeconds: 10
    timeoutSeconds: 5
    failureThreshold: 60
  # Minimal config for testing
  config:
    bert_model:
      model_id: "models/mom-embedding-light"
      threshold: "0.6"
      use_cpu: true
    semantic_cache:
      enabled: true
      backend_type: "memory"
      max_entries: 100
    tools:
      enabled: false
    prompt_guard:
      enabled: false
    classifier:
      category_model:
        model_id: "models/lora_intent_classifier_bert-base-uncased_model"
        use_cpu: true
      pii_model:
        model_id: "models/pii_classifier_modernbert-base_presidio_token_model"
        use_cpu: true
    api:
      batch_classification:
        max_batch_size: 50
        metrics:
          enabled: true
    observability:
      tracing:
        enabled: false
  toolsDb:
  - tool:
      type: "function"
      function:
        name: "test_tool"
        description: "Test tool for CI"
        parameters:
          type: "object"
          properties:
            input:
              type: "string"
          required: ["input"]
    description: "Test tool"
    category: "test"
    tags: ["test"]
```

Apply it:

```bash
kubectl apply -f semantic-router.yaml
```

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

## Configuration

For production deployments, enable persistence:

```yaml
spec:
  persistence:
    enabled: true
    size: 10Gi
    storageClassName: "fast-ssd"  # Adjust for your cluster
```

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

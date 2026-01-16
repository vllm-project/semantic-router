# Semantic Router Operator

The Semantic Router Operator manages the lifecycle of [vLLM Semantic Router](https://github.com/vllm-project/semantic-router) instances on Kubernetes and OpenShift.

## Overview

vLLM Semantic Router is a Mixture-of-Models (MoM) router that intelligently directs OpenAI API requests to appropriate models based on semantic understanding of request intent. This operator automates the deployment and management of Semantic Router instances.

### Key Features

- **Declarative Configuration**: Define Semantic Router instances using Kubernetes Custom Resources
- **Automatic ML Model Management**: Downloads and manages ML models from HuggingFace
- **Full Helm Chart Parity**: Supports all features available in the Helm chart
- **OpenShift Native**: Built using Red Hat Universal Base Images (UBI 10)
- **OLM Integration**: Installable via Operator Lifecycle Manager on OpenShift
- **Auto-scaling**: Optional HorizontalPodAutoscaler support
- **Ingress Support**: Configurable ingress for external access

## Prerequisites

- **Kubernetes**: 1.25 or later (tested on 1.25-1.31)
- **OpenShift**: 4.12 or later (optional, for OLM-based installation)
- **Storage**: Persistent storage provider (10Gi recommended for ML models)
- **HuggingFace Token** (optional): For downloading gated models
- **OLM** (optional): For operator lifecycle management on vanilla Kubernetes

## Installation

### Option 1: Install via kubectl (Recommended for Kubernetes)

```bash
# Clone the repository
git clone https://github.com/vllm-project/semantic-router
cd semantic-router/operator

# Install CRDs
make install

# Deploy the operator
make deploy IMG=ghcr.io/vllm-project/semantic-router-operator:latest
```

### Option 2: Install via OpenShift OperatorHub

1. Navigate to the OpenShift Console
2. Go to **Operators** → **OperatorHub**
3. Search for "Semantic Router Operator"
4. Click **Install** and follow the prompts

### Option 3: Build and Install from Source

```bash
# Clone the repository
git clone https://github.com/vllm-project/semantic-router
cd semantic-router/operator

# Build and push the operator image
make docker-build docker-push IMG=<your-registry>/semantic-router-operator:latest

# Deploy the operator
make deploy IMG=<your-registry>/semantic-router-operator:latest
```

### Option 4: Install via OLM Bundle (Kubernetes + OLM or OpenShift)

```bash
# Build and push bundle
make bundle-build bundle-push BUNDLE_IMG=<your-registry>/semantic-router-operator-bundle:latest

# Create catalog
make catalog-build catalog-push CATALOG_IMG=<your-registry>/semantic-router-operator-catalog:latest

# Install on OpenShift
oc apply -f - <<EOF
apiVersion: operators.coreos.com/v1alpha1
kind: CatalogSource
metadata:
  name: semantic-router-catalog
  namespace: openshift-marketplace
spec:
  sourceType: grpc
  image: <your-registry>/semantic-router-operator-catalog:latest
  displayName: Semantic Router Operator
  publisher: vLLM Project
  updateStrategy:
    registryPoll:
      interval: 10m
EOF

# Create subscription
make openshift-deploy
```

## Quick Start

### 1. Create a SemanticRouter Instance

Create a basic SemanticRouter instance:

```yaml
apiVersion: vllm.ai/v1alpha1
kind: SemanticRouter
metadata:
  name: my-semantic-router
  namespace: default
spec:
  replicas: 1

  image:
    repository: ghcr.io/vllm-project/semantic-router/extproc
    tag: latest

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
```

Apply the configuration:

```bash
kubectl apply -f semantic-router.yaml
```

### 2. Check Status

```bash
# Check the SemanticRouter status
kubectl get semanticrouter my-semantic-router

# Check the deployment
kubectl get deployment my-semantic-router

# Check logs
kubectl logs -l app.kubernetes.io/instance=my-semantic-router
```

### 3. Access the Service

```bash
# Port forward to access locally
kubectl port-forward svc/my-semantic-router 50051:50051 8080:8080

# The gRPC service is available at localhost:50051
# The HTTP API is available at http://localhost:8080
```

## Configuration

### Image Configuration

```yaml
spec:
  image:
    repository: ghcr.io/vllm-project/semantic-router/extproc
    tag: latest
    pullPolicy: IfNotPresent
    imageRegistry: "my-registry.com"  # Optional registry prefix
```

### Resources

```yaml
spec:
  resources:
    limits:
      memory: "7Gi"
      cpu: "2"
    requests:
      memory: "3Gi"
      cpu: "1"
```

### Persistence

```yaml
spec:
  persistence:
    enabled: true
    storageClassName: "fast-ssd"
    accessMode: ReadWriteOnce
    size: 20Gi
    # OR use existing PVC
    existingClaim: "my-models-pvc"
```

### Autoscaling

```yaml
spec:
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 10
    targetCPUUtilizationPercentage: 70
    targetMemoryUtilizationPercentage: 80
```

### Ingress

```yaml
spec:
  ingress:
    enabled: true
    className: "nginx"
    annotations:
      cert-manager.io/cluster-issuer: "letsencrypt-prod"
    hosts:
    - host: semantic-router.example.com
      paths:
      - path: /
        pathType: Prefix
        servicePort: 8080
    tls:
    - secretName: semantic-router-tls
      hosts:
      - semantic-router.example.com
```

### Semantic Router Configuration

The operator supports all configuration options from the Helm chart:

```yaml
spec:
  config:
    # BERT model for embeddings
    bert_model:
      model_id: "models/mom-embedding-light"
      threshold: 0.6
      use_cpu: true

    # Semantic caching
    semantic_cache:
      enabled: true
      backend_type: "memory"
      similarity_threshold: 0.8
      max_entries: 1000
      ttl_seconds: 3600
      eviction_policy: "fifo"

    # Tools auto-selection
    tools:
      enabled: true
      top_k: 3
      similarity_threshold: 0.2

    # Prompt guard (jailbreak detection)
    prompt_guard:
      enabled: true
      model_id: "models/mom-jailbreak-classifier"
      threshold: 0.7

    # Classifiers
    classifier:
      category_model:
        model_id: "models/lora_intent_classifier_bert-base-uncased_model"
        threshold: 0.6
      pii_model:
        model_id: "models/pii_classifier_modernbert-base_presidio_token_model"
        threshold: 0.7

    # Reasoning configuration
    reasoning_families:
      qwen3:
        type: "chat_template_kwargs"
        parameter: "enable_thinking"
      deepseek:
        type: "chat_template_kwargs"
        parameter: "thinking"

    # Observability
    observability:
      tracing:
        enabled: true
        provider: "opentelemetry"
        exporter:
          endpoint: "jaeger:4317"
```

### Security Context

```yaml
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    allowPrivilegeEscalation: false
    capabilities:
      drop:
      - ALL

  podSecurityContext:
    fsGroup: 2000
    seccompProfile:
      type: RuntimeDefault
```

### Probes

```yaml
spec:
  # Startup probe for model loading (can take 10-60 minutes)
  startupProbe:
    enabled: true
    periodSeconds: 10
    timeoutSeconds: 5
    failureThreshold: 360  # 60 minutes total

  livenessProbe:
    enabled: true
    initialDelaySeconds: 30
    periodSeconds: 30
    timeoutSeconds: 10
    failureThreshold: 5

  readinessProbe:
    enabled: true
    initialDelaySeconds: 30
    periodSeconds: 30
    timeoutSeconds: 10
    failureThreshold: 5
```

## HuggingFace Token Configuration

For downloading gated models (like embeddinggemma-300m):

```bash
# Create secret
kubectl create secret generic hf-token-secret \
  --from-literal=token=YOUR_HUGGINGFACE_TOKEN

# Reference in SemanticRouter
spec:
  env:
  - name: HF_TOKEN
    valueFrom:
      secretKeyRef:
        name: hf-token-secret
        key: token
        optional: true
```

## Examples

### Basic Development Instance

```yaml
apiVersion: vllm.ai/v1alpha1
kind: SemanticRouter
metadata:
  name: dev-router
spec:
  replicas: 1
  config:
    semantic_cache:
      enabled: true
      max_entries: 100
```

### Production Instance with HA

```yaml
apiVersion: vllm.ai/v1alpha1
kind: SemanticRouter
metadata:
  name: prod-router
spec:
  replicas: 3

  autoscaling:
    enabled: true
    minReplicas: 3
    maxReplicas: 20
    targetCPUUtilizationPercentage: 70

  resources:
    limits:
      memory: "10Gi"
      cpu: "4"
    requests:
      memory: "5Gi"
      cpu: "2"

  persistence:
    enabled: true
    storageClassName: "fast-ssd"
    size: 20Gi

  ingress:
    enabled: true
    className: "nginx"
    hosts:
    - host: router.prod.example.com
      paths:
      - path: /
        pathType: Prefix
        servicePort: 8080
    tls:
    - secretName: router-tls
      hosts:
      - router.prod.example.com

  affinity:
    podAntiAffinity:
      preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        podAffinityTerm:
          labelSelector:
            matchLabels:
              app.kubernetes.io/instance: prod-router
          topologyKey: kubernetes.io/hostname
```

## Monitoring

### Prometheus Metrics

The operator automatically configures Prometheus metrics on port 9190:

```yaml
spec:
  service:
    metrics:
      enabled: true
      port: 9190
```

Access metrics:

```bash
kubectl port-forward svc/my-semantic-router 9190:9190
curl http://localhost:9190/metrics
```

### OpenTelemetry Tracing

Enable distributed tracing:

```yaml
spec:
  config:
    observability:
      tracing:
        enabled: true
        provider: "opentelemetry"
        exporter:
          type: "otlp"
          endpoint: "tempo:4317"
          insecure: true
        sampling:
          type: "always_on"
          rate: 1.0
```

## Troubleshooting

### Check Operator Logs

```bash
kubectl logs -n semantic-router-operator-system \
  -l app.kubernetes.io/name=semantic-router-operator
```

### Check SemanticRouter Status

```bash
kubectl describe semanticrouter my-semantic-router
```

### Model Download Issues

Models are downloaded at startup. Check the pod logs:

```bash
kubectl logs my-semantic-router-<pod-id>
```

If models fail to download:

1. Check HuggingFace token is configured correctly
2. Verify network connectivity to huggingface.co
3. Ensure sufficient storage space (10Gi minimum)
4. Check startup probe timeout is sufficient (default: 60 minutes)

### Common Issues

**Issue**: Pod stuck in Pending state

- **Solution**: Check PVC is bound: `kubectl get pvc`

**Issue**: Pod crashes with OOM

- **Solution**: Increase memory limits (minimum 3Gi)

**Issue**: Models not loading

- **Solution**: Check HF_TOKEN secret exists and is valid

## Development

### Building the Operator

```bash
# Build operator binary
make build

# Run tests
make test

# Build Docker image
make docker-build IMG=<your-registry>/semantic-router-operator:latest

# Push image
make docker-push IMG=<your-registry>/semantic-router-operator:latest
```

### Local Development

```bash
# Install CRDs
make install

# Run operator locally (outside cluster)
make run

# In another terminal, create a test instance
kubectl apply -f config/samples/vllm_v1alpha1_semanticrouter.yaml
```

### Generate Code

After modifying API types:

```bash
# Generate CRDs and code
make manifests generate
```

## Architecture

The operator manages the following resources:

- **Deployment**: Runs the semantic router pods
- **Service**: Exposes gRPC (50051), HTTP API (8080), and metrics (9190)
- **ConfigMap**: Stores configuration (config.yaml, tools_db.json)
- **PersistentVolumeClaim**: Stores downloaded ML models
- **ServiceAccount**: For RBAC
- **HorizontalPodAutoscaler** (optional): For autoscaling
- **Ingress** (optional): For external access

## Contributing

Contributions are welcome! Please see the main project [CONTRIBUTING.md](../CONTRIBUTING.md).

## License

Apache License 2.0. See [LICENSE](../LICENSE) for details.

## Support

- **Documentation**: https://vllm-semantic-router.com
- **GitHub Issues**: https://github.com/vllm-project/semantic-router/issues
- **Discussions**: https://github.com/vllm-project/semantic-router/discussions

## Related Documentation

- [Helm Chart Documentation](../deploy/helm/semantic-router/README.md)
- [Main Project README](../README.md)
- [Operator SDK Documentation](https://sdk.operatorframework.io/)

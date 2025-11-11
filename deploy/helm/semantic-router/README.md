# Semantic Router Helm Chart

A Helm chart for deploying Semantic Router - an intelligent routing system for LLM applications with built-in classification, caching, and security features.

## TL;DR

```bash
# Install with default values
helm install semantic-router ./deploy/helm/semantic-router

# Install with custom values
helm install semantic-router ./deploy/helm/semantic-router -f ./deploy/helm/semantic-router/values-dev.yaml
```

## Introduction

This chart bootstraps a Semantic Router deployment on a Kubernetes cluster using the Helm package manager. It includes:

- Intelligent routing and classification for LLM requests
- Built-in semantic caching (memory or Milvus)
- PII detection and jailbreak protection
- Tools database for function calling
- Multi-model support with automatic selection
- Prometheus metrics and observability
- Persistent storage for ML models

## Prerequisites

- Kubernetes 1.19+
- Helm 3.2.0+
- PV provisioner support in the underlying infrastructure (for persistent storage)
- (Optional) Ingress controller for external access
- (Optional) cert-manager for TLS certificates

## Installing the Chart

### Basic Installation

To install the chart with the release name `semantic-router`:

```bash
helm install semantic-router ./deploy/helm/semantic-router
```

### Install with Development Configuration

For local development with reduced resource requirements:

```bash
helm install semantic-router ./deploy/helm/semantic-router \
  -f ./deploy/helm/semantic-router/values-dev.yaml \
  --namespace vllm-semantic-router-system \
  --create-namespace
```

### Install with Production Configuration

For production deployment with high availability:

```bash
helm install semantic-router ./deploy/helm/semantic-router \
  -f ./deploy/helm/semantic-router/values-prod.yaml \
  --namespace vllm-semantic-router-system \
  --create-namespace
```

### Install with Custom Values

Create your own values file and install:

```bash
helm install semantic-router ./deploy/helm/semantic-router \
  -f my-values.yaml \
  --namespace my-namespace \
  --create-namespace
```

## Uninstalling the Chart

To uninstall/delete the `semantic-router` deployment:

```bash
helm uninstall semantic-router --namespace vllm-semantic-router-system
```

This command removes all the Kubernetes components associated with the chart and deletes the release.

## Configuration

### Key Configuration Parameters

The following table lists the key configurable parameters of the Semantic Router chart and their default values.

#### Global Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `global.namespace` | Override namespace for all resources | `""` (uses Release.Namespace) |
| `replicaCount` | Number of replicas | `1` |
| `nameOverride` | Override the name of the chart | `""` |
| `fullnameOverride` | Override the full name of the chart | `""` |

#### Image Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `image.repository` | Image repository | `ghcr.io/vllm-project/semantic-router/extproc` |
| `image.tag` | Image tag | `latest` |
| `image.pullPolicy` | Image pull policy | `IfNotPresent` |
| `imagePullSecrets` | Image pull secrets | `[]` |

#### Service Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `service.type` | Service type | `ClusterIP` |
| `service.grpc.port` | gRPC service port | `50051` |
| `service.api.port` | HTTP API service port | `8080` |
| `service.metrics.enabled` | Enable metrics service | `true` |
| `service.metrics.port` | Metrics service port | `9190` |

#### Resources

| Parameter | Description | Default |
|-----------|-------------|---------|
| `resources.limits.memory` | Memory limit | `6Gi` |
| `resources.limits.cpu` | CPU limit | `2` |
| `resources.requests.memory` | Memory request | `3Gi` |
| `resources.requests.cpu` | CPU request | `1` |

#### Persistence

| Parameter | Description | Default |
|-----------|-------------|---------|
| `persistence.enabled` | Enable persistent volume | `true` |
| `persistence.storageClassName` | Storage class name | `standard` |
| `persistence.size` | Storage size | `10Gi` |
| `persistence.accessMode` | Access mode | `ReadWriteOnce` |
| `persistence.existingClaim` | Use existing PVC | `""` |

#### Init Container

| Parameter | Description | Default |
|-----------|-------------|---------|
| `initContainer.enabled` | Enable init container for model downloading | `true` |
| `initContainer.image` | Init container image | `python:3.11-slim` |
| `initContainer.resources.limits.memory` | Init container memory limit | `1Gi` |
| `initContainer.resources.limits.cpu` | Init container CPU limit | `500m` |

#### Autoscaling

| Parameter | Description | Default |
|-----------|-------------|---------|
| `autoscaling.enabled` | Enable horizontal pod autoscaling | `false` |
| `autoscaling.minReplicas` | Minimum number of replicas | `1` |
| `autoscaling.maxReplicas` | Maximum number of replicas | `10` |
| `autoscaling.targetCPUUtilizationPercentage` | Target CPU utilization | `80` |

#### Ingress

| Parameter | Description | Default |
|-----------|-------------|---------|
| `ingress.enabled` | Enable ingress | `false` |
| `ingress.className` | Ingress class name | `""` |
| `ingress.annotations` | Ingress annotations | `{}` |
| `ingress.hosts` | Ingress hosts configuration | See values.yaml |
| `ingress.tls` | Ingress TLS configuration | `[]` |

#### Application Configuration

The `config` section contains the application-specific configuration:

- `bert_model`: BERT model configuration for embeddings
- `semantic_cache`: Semantic cache settings (memory/Milvus)
- `tools`: Tools database configuration
- `prompt_guard`: Jailbreak detection settings
- `vllm_endpoints`: vLLM endpoint configuration
- `classifier`: Category and PII classifier settings
- `categories`: Category-specific model scores and prompts
- `api`: API configuration including batch processing

See [values.yaml](values.yaml) for complete configuration options.

## Usage Examples

### Basic Deployment

```bash
# Install semantic router
helm install semantic-router ./deploy/helm/semantic-router

# Wait for deployment to be ready
kubectl wait --for=condition=Available deployment/semantic-router \
  -n vllm-semantic-router-system --timeout=600s

# Port forward to access the API
kubectl port-forward -n vllm-semantic-router-system \
  svc/semantic-router 8080:8080
```

### Test the API

```bash
# Health check
curl http://localhost:8080/health

# Intent classification
curl -X POST http://localhost:8080/api/v1/classify/intent \
  -H "Content-Type: application/json" \
  -d '{"text": "What is machine learning?"}'

# Category classification
curl -X POST http://localhost:8080/api/v1/classify/category \
  -H "Content-Type: application/json" \
  -d '{"text": "Explain photosynthesis"}'
```

### Access Metrics

```bash
kubectl port-forward -n vllm-semantic-router-system \
  svc/semantic-router-metrics 9190:9190

curl http://localhost:9190/metrics
```

### Upgrade Deployment

```bash
# Upgrade with new values
helm upgrade semantic-router ./deploy/helm/semantic-router \
  -f my-updated-values.yaml

# Upgrade to a new version
helm upgrade semantic-router ./deploy/helm/semantic-router \
  --set image.tag=v0.2.0

# Rollback to previous version
helm rollback semantic-router
```

### Custom Configuration Example

Create a `custom-values.yaml`:

```yaml
replicaCount: 2

resources:
  limits:
    memory: "8Gi"
    cpu: "2"
  requests:
    memory: "4Gi"
    cpu: "1"

config:
  vllm_endpoints:
    - name: "my-endpoint"
      address: "10.0.1.100"
      port: 8000
      weight: 1

  semantic_cache:
    enabled: true
    backend_type: "milvus"
    max_entries: 5000

ingress:
  enabled: true
  className: "nginx"
  hosts:
    - host: semantic-router.mydomain.com
      paths:
        - path: /
          pathType: Prefix
          servicePort: 8080
```

Then install:

```bash
helm install semantic-router ./deploy/helm/semantic-router \
  -f custom-values.yaml \
  --namespace production \
  --create-namespace
```

## Migration from Kustomize

If you're currently using the Kustomize deployment, here's how to migrate:

1. **Export your current configuration:**

   ```bash
   kubectl get configmap semantic-router-config \
     -n vllm-semantic-router-system \
     -o yaml > current-config.yaml
   ```

2. **Create a values file with your configuration:**

   ```bash
   # Extract config.yaml and tools_db.json from the configmap
   # and merge into your values file
   ```

3. **Uninstall Kustomize deployment:**

   ```bash
   kubectl delete -k deploy/kubernetes/
   ```

4. **Install with Helm:**

   ```bash
   helm install semantic-router ./deploy/helm/semantic-router \
     -f your-values.yaml \
     --namespace vllm-semantic-router-system \
     --create-namespace
   ```

## Development

### Validating the Chart

```bash
# Lint the chart
helm lint ./deploy/helm/semantic-router

# Dry-run installation
helm install semantic-router ./deploy/helm/semantic-router \
  --dry-run --debug

# Template rendering
helm template semantic-router ./deploy/helm/semantic-router \
  -f ./deploy/helm/semantic-router/values-dev.yaml
```

### Package the Chart

```bash
helm package ./deploy/helm/semantic-router

# Output: semantic-router-0.1.0.tgz
```

## Troubleshooting

### Pods not starting

Check pod status:

```bash
kubectl get pods -n vllm-semantic-router-system
kubectl describe pod <pod-name> -n vllm-semantic-router-system
kubectl logs <pod-name> -n vllm-semantic-router-system
```

### Model download failures

Check init container logs:

```bash
kubectl logs <pod-name> -n vllm-semantic-router-system -c model-downloader
```

### Insufficient resources

If pods are pending due to insufficient resources, reduce resource requests:

```bash
helm upgrade semantic-router ./deploy/helm/semantic-router \
  -f ./deploy/helm/semantic-router/values-dev.yaml
```

### PVC issues

Check PVC status:

```bash
kubectl get pvc -n vllm-semantic-router-system
kubectl describe pvc semantic-router-models -n vllm-semantic-router-system
```

## Support

For issues and feature requests, please visit:

- GitHub: https://github.com/vllm-project/semantic-router
- Documentation: https://semantic-router.io

## License

This Helm chart is licensed under the same license as the Semantic Router project.

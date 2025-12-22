# Helm Chart Deployment Guide

This directory contains the Helm chart for deploying Semantic Router on Kubernetes.

## Directory Structure

```
deploy/helm/
├── MIGRATION.md                    # Migration guide from Kustomize to Helm
├── validate-chart.sh              # Chart validation script
└── semantic-router/               # Helm chart
    ├── Chart.yaml                 # Chart metadata
    ├── values.yaml                # Default configuration values
    ├── values-dev.yaml            # Development environment values
    ├── values-prod.yaml           # Production environment values
    ├── README.md                  # Comprehensive chart documentation
    ├── .helmignore               # Helm ignore patterns
    └── templates/                 # Kubernetes resource templates
        ├── _helpers.tpl          # Template helpers
        ├── namespace.yaml        # Namespace resource
        ├── serviceaccount.yaml   # Service account
        ├── configmap.yaml        # Configuration
        ├── pvc.yaml              # Persistent volume claim
        ├── deployment.yaml       # Main deployment
        ├── service.yaml          # Services (gRPC, API, metrics)
        ├── ingress.yaml          # Ingress (optional)
        ├── hpa.yaml              # Horizontal Pod Autoscaler (optional)
        └── NOTES.txt             # Post-installation notes
```

## Quick Start

### Prerequisites

- Kubernetes 1.19+
- Helm 3.2.0+
- kubectl configured to access your cluster

### Install

```bash
# Using Make (recommended)
make helm-install

# Or with Helm directly
helm install semantic-router ./deploy/helm/semantic-router \
  --namespace vllm-semantic-router-system \
  --create-namespace
```

> Need a registry mirror/proxy (e.g., in China)? Append `--set global.imageRegistry=<your-registry>` to any Helm install/upgrade command.

### Verify Installation

```bash
# Check Helm release status
make helm-status

# Check pods
kubectl get pods -n vllm-semantic-router-system

# View logs
make helm-logs
```

### Access the Application

```bash
# Port forward API
make helm-port-forward-api

# Test the API
curl http://localhost:8080/health
```

## Deployment Scenarios

### Development Environment

For local development with reduced resources:

```bash
make helm-dev

# Or manually:
helm install semantic-router ./deploy/helm/semantic-router \
  -f ./deploy/helm/semantic-router/values-dev.yaml \
  --namespace vllm-semantic-router-system \
  --create-namespace
```

**Features:**

- Reduced resource requests (2Gi RAM, 500m CPU)
- Smaller storage (5Gi)
- Faster probes
- Debug-friendly configuration

### Production Environment

For production deployment with high availability:

```bash
make helm-prod

# Or manually:
helm install semantic-router ./deploy/helm/semantic-router \
  -f ./deploy/helm/semantic-router/values-prod.yaml \
  --namespace production \
  --create-namespace
```

**Features:**

- Multiple replicas (3)
- High resource allocation (8Gi RAM, 4 CPU)
- Auto-scaling enabled
- Security hardening
- Ingress with TLS
- Production-grade storage

### Custom Configuration

Create your own values file:

```yaml
# my-values.yaml
replicaCount: 2

resources:
  limits:
    memory: "8Gi"
    cpu: "2"

config:
  vllm_endpoints:
    - name: "my-endpoint"
      address: "10.0.1.100"
      port: 8000
      weight: 1

ingress:
  enabled: true
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
  -f my-values.yaml \
  --namespace my-namespace \
  --create-namespace
```

## Make Targets

The project includes convenient Make targets for Helm operations:

### Installation & Management

```bash
make helm-install              # Install the chart
make helm-upgrade              # Upgrade the release
make helm-uninstall            # Uninstall the release
make helm-status               # Show release status
make helm-list                 # List all releases
```

### Development

```bash
make helm-lint                 # Lint the chart
make helm-template             # Template the chart
make helm-dev                  # Deploy with dev config
make helm-prod                 # Deploy with prod config
make helm-package              # Package the chart
```

### Testing & Debugging

```bash
make helm-test                 # Test the deployment
make helm-logs                 # Show logs
make helm-values               # Show computed values
make helm-manifest             # Show deployed manifest
```

### Port Forwarding

```bash
make helm-port-forward-api     # Port forward API (8080)
make helm-port-forward-grpc    # Port forward gRPC (50051)
make helm-port-forward-metrics # Port forward metrics (9190)
```

### Rollback & Cleanup

```bash
make helm-rollback             # Rollback to previous version
make helm-history              # Show release history
make helm-clean                # Complete cleanup
```

### Help

```bash
make help-helm                 # Show Helm help
```

## Validation

Before deploying, validate the Helm chart:

```bash
# Run validation script
./deploy/helm/validate-chart.sh

# Or manually:
make helm-lint
make helm-template
```

## Upgrading

### In-Place Upgrade

```bash
# Upgrade with new values
helm upgrade semantic-router ./deploy/helm/semantic-router \
  -f my-updated-values.yaml \
  --namespace vllm-semantic-router-system

# Or using Make:
make helm-upgrade HELM_VALUES_FILE=my-updated-values.yaml
```

### Rollback

If an upgrade fails:

```bash
# Rollback to previous version
make helm-rollback

# Or rollback to specific revision
helm rollback semantic-router 1 --namespace vllm-semantic-router-system
```

## Configuration Examples

### Example 1: Custom Endpoints

```yaml
config:
  vllm_endpoints:
    - name: "endpoint-1"
      address: "10.0.1.10"
      port: 8000
      weight: 2
    - name: "endpoint-2"
      address: "10.0.1.11"
      port: 8000
      weight: 1
```

### Example 2: Enable Ingress

```yaml
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

### Example 3: Enable Auto-scaling

```yaml
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80
```

### Example 4: Custom Security Context

```yaml
podSecurityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 1000

securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  allowPrivilegeEscalation: false
  capabilities:
    drop:
      - ALL
```

## Migrating from Kustomize

If you're currently using the Kustomize deployment, see [MIGRATION.md](MIGRATION.md) for detailed migration instructions.

## Troubleshooting

### Pods Stuck in Pending

```bash
# Check events
kubectl describe pod -n vllm-semantic-router-system

# Common causes:
# - Insufficient resources
# - PVC not binding
# - Image pull errors

# Solution: Reduce resources
helm upgrade semantic-router ./deploy/helm/semantic-router \
  -f values-dev.yaml \
  --namespace vllm-semantic-router-system
```

### Model Download Issues

```bash
# Models are now downloaded automatically by the router at startup
# Check router logs for model download progress
kubectl logs <pod-name> -n vllm-semantic-router-system

# Common causes:
# - Network issues
# - HuggingFace rate limits
# - Insufficient storage
# - Missing huggingface-cli in container

# Solution: Check PVC and network
kubectl get pvc -n vllm-semantic-router-system
```

### Service Not Accessible

```bash
# Check service
kubectl get svc -n vllm-semantic-router-system

# Check endpoints
kubectl get endpoints -n vllm-semantic-router-system

# Test internally
kubectl run -it --rm debug --image=curlimages/curl --restart=Never -- \
  curl http://semantic-router.vllm-semantic-router-system:8080/health
```

## Best Practices

1. **Use Version Control**: Keep your `values.yaml` files in version control
2. **Environment Separation**: Use different namespaces and values files for different environments
3. **Resource Limits**: Always set appropriate resource limits based on your workload
4. **Monitoring**: Enable metrics and set up monitoring
5. **Security**: Use security contexts and network policies
6. **Backups**: Regularly backup your PVC data
7. **Testing**: Test upgrades in dev/staging before production

## CI/CD Integration

### GitHub Actions Example

```yaml
- name: Deploy with Helm
  run: |
    helm upgrade --install semantic-router ./deploy/helm/semantic-router \
      -f values-prod.yaml \
      --namespace production \
      --create-namespace \
      --wait \
      --timeout 10m
```

### GitLab CI Example

```yaml
deploy:
  script:
    - helm upgrade --install semantic-router ./deploy/helm/semantic-router
        -f values-prod.yaml
        --namespace production
        --create-namespace
        --wait
```

### ArgoCD Example

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: semantic-router
spec:
  project: default
  source:
    repoURL: https://github.com/vllm-project/semantic-router
    targetRevision: main
    path: deploy/helm/semantic-router
    helm:
      valueFiles:
        - values-prod.yaml
  destination:
    server: https://kubernetes.default.svc
    namespace: production
```

## Additional Resources

- [Chart README](semantic-router/README.md) - Detailed chart documentation
- [Migration Guide](MIGRATION.md) - Kustomize to Helm migration
- [Project Documentation](../../README.md) - Main project documentation
- [Helm Documentation](https://helm.sh/docs/) - Official Helm docs

## Support

For issues and questions:

- GitHub Issues: https://github.com/vllm-project/semantic-router/issues
- Documentation: https://semantic-router.io
- Chart Issues: Tag with `helm` label

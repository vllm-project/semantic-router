# Kind Cluster Setup for Semantic Router

This directory contains configuration and scripts for setting up a local Kubernetes cluster using [kind](https://kind.sigs.k8s.io/) for development and testing of semantic-router.

## Quick Start

### 1. Generate Kind Configuration

The `kind-config.yaml` file is auto-generated from the template to adapt to your local environment:

```bash
# From project root
./tools/kind/generate-kind-config.sh
```

This script will:

- Auto-detect your project root directory
- Replace `${PROJECT_ROOT}` with the absolute path
- Generate `kind-config.yaml` with correct host paths
- Create the `models/` directory if it doesn't exist

### 2. Create Kind Cluster

```bash
kind create cluster --config tools/kind/kind-config.yaml
```

This will create a cluster with:

- 1 control-plane node
- 1 worker node (with models directory mounted at `/mnt/models`)
- Resource limits configured for semantic-router workloads
- Port 30080 exposed for external access

### 3. Load Docker Images (for offline/local images)

If you have local images or need to work offline:

```bash
# Load init container image
kind load docker-image python:3.11-slim -n semantic-router-cluster

# Load semantic-router image
kind load docker-image ghcr.io/vllm-project/semantic-router/extproc:latest -n semantic-router-cluster
```

### 4. Deploy Semantic Router

```bash
kubectl apply -k deploy/kubernetes/
```

### 5. Verify Deployment

```bash
# Check pods
kubectl get pods -n vllm-semantic-router-system -o wide

# Check logs (models are downloaded automatically at startup)
kubectl logs -n vllm-semantic-router-system deploy/semantic-router -c semantic-router
```

## File Structure

- `kind-config.yaml.template` - Template with `${PROJECT_ROOT}` placeholder
- `generate-kind-config.sh` - Script to generate `kind-config.yaml` from template
- `kind-config.yaml` - Auto-generated, **DO NOT COMMIT** (in .gitignore)

## How It Works

### Path Auto-Detection

The `generate-kind-config.sh` script:

1. Detects the project root (two levels up from `tools/kind/`)
2. Exports `PROJECT_ROOT` environment variable
3. Uses `envsubst` to replace `${PROJECT_ROOT}` in the template
4. Outputs to `kind-config.yaml`

### Model Mounting

- **Worker Node**: The local `${PROJECT_ROOT}/models` directory is mounted to `/mnt/models` inside the worker node container
- **PersistentVolume**: Kubernetes PV uses `hostPath: /mnt/models` to access the models
- **Init Container**: Checks if models exist; if not, downloads them (requires internet connection)

### Resource Configuration

The cluster is configured with:

**Control Plane Node:**

- System reserved: 1Gi memory, 500m CPU
- Kube reserved: 1Gi memory, 500m CPU
- API server max concurrent requests: 400
- etcd quota: 8GB

**Worker Node:**

- System reserved: 500Mi memory, 250m CPU
- Kube reserved: 500Mi memory, 250m CPU
- Models directory mounted from host

## Troubleshooting

### Models Not Found in Pod

```bash
# Check if worker node has models mounted
docker exec semantic-router-cluster-worker ls -la /mnt/models

# Verify PV/PVC binding
kubectl get pv,pvc -n vllm-semantic-router-system

# Check pod is scheduled on worker (not control-plane)
kubectl get pods -n vllm-semantic-router-system -o wide
```

### Regenerate Configuration

If you move the project or need to update paths:

```bash
./tools/kind/generate-kind-config.sh
kind delete cluster --name semantic-router-cluster
kind create cluster --config tools/kind/kind-config.yaml
```

### ImagePullBackOff

For offline development or registry issues:

```bash
kind load docker-image <image-name> -n semantic-router-cluster
```

## Cleanup

```bash
# Delete just the deployment
kubectl delete -k deploy/kubernetes/

# Delete the entire cluster
kind delete cluster --name semantic-router-cluster
```

## Advanced Usage

### Using a Different Models Directory

Edit `kind-config.yaml.template` and change:

```yaml
- hostPath: ${PROJECT_ROOT}/models
  containerPath: /mnt/models
```

to:

```yaml
- hostPath: ${PROJECT_ROOT}/path/to/your/models
  containerPath: /mnt/models
```

Then regenerate:

```bash
./tools/kind/generate-kind-config.sh
```

### Multiple Worker Nodes

Add more worker nodes in the template:

```yaml
- role: worker
  extraMounts:
    - hostPath: ${PROJECT_ROOT}/models
      containerPath: /mnt/models
- role: worker
  extraMounts:
    - hostPath: ${PROJECT_ROOT}/models
      containerPath: /mnt/models
```

## References

- [kind Documentation](https://kind.sigs.k8s.io/)
- [kind Configuration](https://kind.sigs.k8s.io/docs/user/configuration/)
- [Kubernetes Persistent Volumes](https://kubernetes.io/docs/concepts/storage/persistent-volumes/)

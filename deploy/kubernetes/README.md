# Semantic Router Kubernetes Deployment

This directory contains Kubernetes manifests for deploying the Semantic Router using Kustomize. It provides two modes similar to docker-compose profiles:

- core: only the semantic-router (no llm-katan)
- llm-katan: semantic-router plus an llm-katan sidecar listening on 8002 (served model name `qwen3`)

## Architecture

The deployment consists of:

- **ConfigMap**: Contains `config.yaml` and `tools_db.json` configuration files
- **PersistentVolumeClaim**: 30Gi storage for model files (adjust based on models you enable)
- **Deployment**:
  - **Init Container**: Downloads/copies model files to persistent volume
  - **Main Container**: Runs the semantic router service
- **Services**:
  - Main service exposing gRPC port (50051), Classification API (8080), and metrics port (9190)
  - Separate metrics service for monitoring

## Ports

- **50051**: gRPC API (vLLM Semantic Router ExtProc)
- **8080**: Classification API (HTTP REST API)
- **9190**: Prometheus metrics

## Quick Start

### Standard Kubernetes Deployment

First-time apply (creates PVC via storage overlay):

```bash
kubectl apply -k deploy/kubernetes/overlays/storage
kubectl apply -k deploy/kubernetes/overlays/core   # or overlays/llm-katan

# Check deployment status
kubectl get pods -l app=semantic-router -n vllm-semantic-router-system
kubectl get services -l app=semantic-router -n vllm-semantic-router-system

# View logs
kubectl logs -l app=semantic-router -n vllm-semantic-router-system -f
```

Day-2 updates (do not touch PVC):

```bash
kubectl apply -k deploy/kubernetes/overlays/core   # or overlays/llm-katan
```

### Kind (Kubernetes in Docker) Deployment

For local development and testing, you can deploy to a kind cluster with optimized resource settings.

#### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed and running
- [kind](https://kind.sigs.k8s.io/docs/user/quick-start/#installation) installed
- [kubectl](https://kubernetes.io/docs/tasks/tools/) installed

#### Automated Deployment

Use the provided make targets for a complete automated setup:

```bash
# Complete setup: create cluster and deploy
make setup

# Or step by step:
make create-cluster
make deploy
```

The setup process will:

1. Create a kind cluster with optimized configuration
2. Deploy the semantic router with appropriate resource limits
3. Wait for the deployment to be ready
4. Show deployment status and access instructions

#### Manual Kind Deployment

If you prefer manual deployment:

**Step 1: Create kind cluster with custom configuration**

```bash
# Create cluster with optimized resource settings
kind create cluster --name semantic-router-cluster --config tools/kind/kind-config.yaml

# Verify cluster is ready
kubectl wait --for=condition=Ready nodes --all --timeout=300s
```

**Step 2: Deploy the application**

```bash
# First-time storage (PVC)
kubectl apply -k deploy/kubernetes/overlays/storage

# Deploy app
kubectl apply -k deploy/kubernetes/

# Wait for deployment to be ready
kubectl wait --for=condition=Available deployment/semantic-router -n vllm-semantic-router-system --timeout=600s
```

**Step 3: Check deployment status**

```bash
# Check pods
kubectl get pods -n vllm-semantic-router-system -o wide

# Check services
kubectl get services -n vllm-semantic-router-system

# View logs
kubectl logs -l app=semantic-router -n vllm-semantic-router-system -f
```

#### Resource Requirements for Kind

The deployment is optimized for kind clusters with the following resource allocation:

- **Init Container**: 512Mi memory, 250m CPU (limits: 1Gi memory, 500m CPU)
- **Main Container**: 3Gi memory, 1 CPU (limits: 6Gi memory, 2 CPU)
- **Total Cluster**: Recommended minimum 8GB RAM, 4 CPU cores

#### Kind Cluster Configuration

The `tools/kind/kind-config.yaml` provides:

- Control plane node with system resource reservations
- Worker node for application workloads
- Optimized kubelet settings for resource management

#### Accessing Services in Kind

Using make commands (recommended):

```bash
# Access Classification API (HTTP REST)
make port-forward-api

# Access gRPC API
make port-forward-grpc

# Access metrics
make port-forward-metrics
```

Or using kubectl directly:

```bash
# Access Classification API (HTTP REST)
kubectl port-forward -n vllm-semantic-router-system svc/semantic-router 8080:8080

# Access gRPC API
kubectl port-forward -n vllm-semantic-router-system svc/semantic-router 50051:50051

# Access metrics
kubectl port-forward -n vllm-semantic-router-system svc/semantic-router-metrics 9190:9190
```

#### Testing the Deployment

Use the provided make targets:

```bash
# Test overall deployment
make test-deployment

# Test Classification API specifically
make test-api

# Check deployment status
make status

# View logs
make logs
```

The make targets provide comprehensive testing including:

- Pod readiness checks
- Service availability verification
- PVC status validation
- API health checks
- Basic functionality testing

#### Cleanup

Using make commands (recommended):

```bash
# Complete cleanup: undeploy and delete cluster
make cleanup

# Or step by step:
make undeploy
make delete-cluster
```

Or using kubectl/kind directly:

```bash
# Remove deployment
kubectl delete -k deploy/kubernetes/

# Delete the kind cluster
kind delete cluster --name semantic-router-cluster
```

## Notes on dependencies

- Gateway API Inference Extension CRDs are required only when using the Envoy AI Gateway integration in `deploy/kubernetes/ai-gateway/`. Follow the installation steps in `website/docs/installation/kubernetes.md` if you plan to use the gateway path.
- The core kustomize deployment in this folder does not install Envoy Gateway or AI Gateway; those are optional components documented separately.

## Make Commands Reference

The project provides comprehensive make targets for managing kind clusters and deployments:

### Cluster Management

```bash
make create-cluster     # Create kind cluster with optimized configuration
make delete-cluster     # Delete kind cluster
make cluster-info       # Show cluster information and resource usage
```

### Deployment Management

```bash
make deploy             # Deploy semantic-router to the cluster
make undeploy           # Remove semantic-router from the cluster
make load-image         # Load Docker image into kind cluster
make status             # Show deployment status
```

### Testing and Monitoring

```bash
make test-deployment    # Test the deployment
make test-api           # Test the Classification API
make logs               # Show application logs
```

### Port Forwarding

```bash
make port-forward-api      # Port forward Classification API (8080)
make port-forward-grpc     # Port forward gRPC API (50051)
make port-forward-metrics  # Port forward metrics (9190)
```

### Combined Operations

```bash
make setup              # Complete setup (create-cluster + deploy)
make cleanup            # Complete cleanup (undeploy + delete-cluster)
```

### Configuration Variables

You can customize the deployment using environment variables:

```bash
# Custom cluster name
KIND_CLUSTER_NAME=my-cluster make create-cluster

# Custom kind config file
KIND_CONFIG_FILE=my-config.yaml make create-cluster

# Custom namespace
KUBE_NAMESPACE=my-namespace make deploy

# Custom Docker image
DOCKER_IMAGE=my-registry/semantic-router:latest make load-image
```

### Help

```bash
make help-kube          # Show all available Kubernetes targets
```

## Troubleshooting

### Common Issues

**Pod stuck in Pending state:**

```bash
# Check node resources
kubectl describe nodes

# Check pod events
kubectl describe pod -n semantic-router -l app=semantic-router
```

**Init container fails:**

```bash
# Check init container logs
kubectl logs -n semantic-router -l app=semantic-router -c model-downloader
```

**Out of memory errors:**

```bash
# Check resource usage
kubectl top pods -n semantic-router

# Adjust resource limits in base/deployment.yaml if needed
```

### Storage sizing

- The default PVC is 30Gi. If the enabled models are small, you can reduce it; otherwise reserve at least 2–3x the total model size.
- If your cluster's default StorageClass isn't named `standard`, change `storageClassName` in `pvc.yaml` accordingly or remove the field to use the default class.

### Resource Optimization

For different environments, you can adjust resource requirements:

- **Development**: 2Gi memory, 0.5 CPU
- **Testing**: 4Gi memory, 1 CPU
- **Production**: 8Gi+ memory, 2+ CPU

Edit the `resources` section in `base/deployment.yaml` accordingly.

## Files Overview

### Kubernetes Manifests (`deploy/kubernetes/`)

- `base/` - Shared resources (Namespace, Service, ConfigMap, Deployment)
  - `namespace.yaml` - Dedicated namespace for the application
  - `service.yaml` - gRPC, HTTP API, and metrics services
  - `deployment.yaml` - App deployment (init downloads by default; imagePullPolicy IfNotPresent)
  - `config.yaml` - Application configuration (defaults to qwen3 @ 127.0.0.1:8002)
  - `tools_db.json` - Tools database for semantic routing
  - `pv.yaml` - OPTIONAL hostPath PV for local models (edit path as needed)
- `overlays/core/` - Core deployment (no llm-katan), references `base/`
- `overlays/llm-katan/` - Adds llm-katan sidecar via local patch (no parent file references)
- `overlays/storage/` - PVC only (self-contained `namespace.yaml` + `pvc.yaml`), run once to create storage
- `kustomization.yaml` - Root entry (defaults to `overlays/core`)

### Development Tools

## Choose a mode: core or llm-katan

- Core mode (default root points here):

  ```bash
  kubectl apply -k deploy/kubernetes
  # or explicitly
  kubectl apply -k deploy/kubernetes/overlays/core
  ```

- llm-katan mode:

  ```bash
  kubectl apply -k deploy/kubernetes/overlays/llm-katan
  ```

Notes for llm-katan:

Notes for llm-katan:

- The init container will attempt to download `Qwen/Qwen3-0.6B` into `/app/models/Qwen/Qwen3-0.6B` and the embedding model `sentence-transformers/all-MiniLM-L12-v2` into `/app/models/all-MiniLM-L12-v2`. In restricted networks, these downloads may fail—pre-populate the PV or point the init script to your internal artifact store as needed.
- The default Kubernetes `config.yaml` has been aligned to use `qwen3` and endpoint `127.0.0.1:8002`.

- `tools/kind/kind-config.yaml` - Kind cluster configuration for local development
- `tools/make/kube.mk` - Make targets for Kubernetes operations
- `Makefile` - Root makefile including all make targets

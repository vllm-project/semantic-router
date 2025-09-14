# Semantic Router Kubernetes Deployment

This directory contains Kubernetes manifests for deploying the Semantic Router using Kustomize.

## Architecture

The deployment consists of:

- **ConfigMap**: Contains `config.yaml` and `tools_db.json` configuration files
- **PersistentVolumeClaim**: 10Gi storage for model files  
- **Deployment**: 
  - **Init Container**: Downloads/copies model files to persistent volume
  - **Main Container**: Runs the semantic router service
- **Services**: 
  - Main service exposing gRPC port (50051) and metrics port (9190)
  - Separate metrics service for monitoring

## Ports

- **50051**: gRPC API (vLLM Semantic Router ExtProc)
- **9190**: Prometheus metrics

## Deployment

```bash
kubectl apply -k deploy/kubernetes/

# Check deployment status
kubectl get pods -l app=semantic-router -n semantic-router
kubectl get services -l app=semantic-router -n semantic-router

# View logs
kubectl logs -l app=semantic-router -n semantic-router -f
```

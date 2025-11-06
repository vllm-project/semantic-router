# Install in Kubernetes

Deploy the vLLM Semantic Router on Kubernetes using the provided manifests.

## Quick Start

```bash
# Deploy semantic router
kubectl apply -k deploy/kubernetes/

# Wait for deployment
kubectl wait --for=condition=Available deployment/semantic-router -n vllm-semantic-router-system --timeout=600s
```

## Configuration

Edit `deploy/kubernetes/config.yaml` to configure your endpoints and policies before deployment.

## Integration Options

For advanced features, see the integration guides:

- [Install with Envoy AI Gateway](k8s/ai-gateway.md) - Envoy AI Gateway for traffic management and load balancing

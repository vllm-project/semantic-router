# LLM Katan Kubernetes fixture

These Kustomize manifests run
[LLM Katan](../../../e2e/testing/llm-katan/README.md), a small
OpenAI-compatible server used for development and integration tests. It lets a
cluster exercise model discovery and routing without deploying a production
inference engine.

LLM Katan is a test fixture. Do not use these manifests as a production model
serving reference or as evidence of model quality, throughput, or isolation.

## What is included

```text
base/                   shared Namespace, Deployment, Service, and PVC
components/common/      common Kustomize component
overlays/gpt35/         served-model alias gpt-3.5-turbo
overlays/claude/        served-model alias claude-3-haiku
verify-deployment.sh    cluster and endpoint checks
```

Both overlays use the maintained LLM Katan image and change the model identity
presented by the API. Review the Kustomize output before applying it:

```bash
kubectl kustomize deploy/kubernetes/llm-katan/overlays/gpt35
```

## Deploy one fixture

```bash
kubectl apply -k deploy/kubernetes/llm-katan/overlays/gpt35
kubectl wait --for=condition=Available \
  deployment/llm-katan-gpt35 \
  --namespace llm-katan-system \
  --timeout=10m
kubectl get pods,services,pvc --namespace llm-katan-system
```

The first start may download a model into the PVC. Inspect the init-container
and server logs if readiness does not complete:

```bash
kubectl logs --namespace llm-katan-system \
  deployment/llm-katan-gpt35 --all-containers
```

## Exercise the API

Forward the overlay's Service and send a request from another terminal:

```bash
kubectl port-forward --namespace llm-katan-system \
  service/llm-katan-gpt35 8000:8000
```

```bash
curl --fail-with-body http://127.0.0.1:8000/health
curl --fail-with-body http://127.0.0.1:8000/v1/models
curl --fail-with-body http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Reply with OK"}],
    "max_tokens": 8
  }'
```

Deploy the second alias in the same namespace when a routing test needs two
distinct model names:

```bash
kubectl apply -k deploy/kubernetes/llm-katan/overlays/claude
kubectl get services --namespace llm-katan-system
```

## Validate or remove

The verification script checks resources and probes a selected Service:

```bash
deploy/kubernetes/llm-katan/verify-deployment.sh \
  llm-katan-system llm-katan-gpt35
```

Remove only the overlays that you applied:

```bash
kubectl delete -k deploy/kubernetes/llm-katan/overlays/claude
kubectl delete -k deploy/kubernetes/llm-katan/overlays/gpt35
```

For local, non-Kubernetes usage and all server flags, see the
[LLM Katan test-server README](../../../e2e/testing/llm-katan/README.md).

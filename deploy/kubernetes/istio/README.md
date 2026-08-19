# Istio Gateway example

This example connects Semantic Router to an Istio-managed Envoy Gateway through
Envoy External Processing. The Router chooses between two model aliases;
`HTTPRoute` resources send each selected alias to its backend Service.

```text
client -> Istio Gateway -> HTTPRoute -> model Service
                    |
                    +-> ext_proc -> Semantic Router
```

For the supported gateway architecture and version guidance, start with the
[Istio deployment guide](../../../website/docs/installation/k8s/istio.md). This
README documents only the manifests in this directory.

## Requirements

- a Kubernetes cluster with `kubectl` access;
- Istio with Kubernetes Gateway API support;
- the Gateway API CRDs required by the manifests;
- two GPU nodes, or equivalent edits to use backends your cluster can run;
- `HF_TOKEN` access for the gated model, if required.

The included vLLM manifests request `nvidia.com/gpu: 1` for each backend and
create 40 GiB and 20 GiB PVCs. Review the image tags, storage class, resources,
and model access before applying them.

## 1. Deploy the model backends

```bash
kubectl create secret generic hf-token-secret \
  --from-literal=token="$HF_TOKEN"
kubectl apply -f deploy/kubernetes/istio/vLlama3.yaml
kubectl apply -f deploy/kubernetes/istio/vPhi4.yaml
kubectl wait --for=condition=Available deployment/llama-8b --timeout=20m
kubectl wait --for=condition=Available deployment/phi4-mini --timeout=20m
```

If you replace either backend, update its Service name and port in
[`config.yaml`](config.yaml) and the matching `HTTPRoute`.

## 2. Deploy the Router and gateway wiring

Install Istio and the required Gateway API CRDs using their upstream
instructions, then apply the repository assets:

```bash
kubectl apply -f deploy/kubernetes/istio/gateway.yaml
kubectl apply -k deploy/kubernetes/istio/
kubectl wait --for=condition=Available deployment/semantic-router \
  --namespace vllm-semantic-router-system \
  --timeout=10m

kubectl apply -f deploy/kubernetes/istio/destinationrule.yaml
kubectl apply -f deploy/kubernetes/istio/envoyfilter.yaml
kubectl apply -f deploy/kubernetes/istio/httproute-llama3-8b.yaml
kubectl apply -f deploy/kubernetes/istio/httproute-phi4-mini.yaml
```

The `EnvoyFilter` selects the workload labeled for `inference-gateway` and
calls `semantic-router.vllm-semantic-router-system.svc.cluster.local:50051`.
Changing either name requires updating the filter.

## 3. Verify routing

Check resource readiness before sending traffic:

```bash
kubectl get gateway,httproute
kubectl get pods,services
kubectl get pods,services --namespace vllm-semantic-router-system
```

For minikube, obtain the gateway URL dynamically:

```bash
GATEWAY_URL="$(minikube service inference-gateway-istio --url)"
```

Send an automatic route request:

```bash
curl --fail-with-body "$GATEWAY_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "What is 2 + 2?"}],
    "max_tokens": 32,
    "temperature": 0
  }'
```

Use `model: llama3-8b` or `model: phi4-mini` to verify explicit model routing.
The selected alias must match the `x-selected-model` header match in the
corresponding `HTTPRoute`.

## Diagnose failures

```bash
kubectl describe gateway inference-gateway
kubectl describe httproute vsr-llama8b
kubectl logs deployment/inference-gateway-istio -c istio-proxy
kubectl logs deployment/semantic-router \
  --namespace vllm-semantic-router-system
```

- No accepted route: inspect the `Gateway` and `HTTPRoute` status conditions.
- ext_proc errors: compare the gateway label and Router gRPC Service with
  `envoyfilter.yaml`.
- backend timeout: probe the backend Service from inside the cluster and check
  model readiness.
- Router startup delay: inspect the model-downloader init container and PVC.

## Cleanup

```bash
kubectl delete -f deploy/kubernetes/istio/httproute-phi4-mini.yaml
kubectl delete -f deploy/kubernetes/istio/httproute-llama3-8b.yaml
kubectl delete -f deploy/kubernetes/istio/envoyfilter.yaml
kubectl delete -f deploy/kubernetes/istio/destinationrule.yaml
kubectl delete -k deploy/kubernetes/istio/
kubectl delete -f deploy/kubernetes/istio/vPhi4.yaml
kubectl delete -f deploy/kubernetes/istio/vLlama3.yaml
kubectl delete -f deploy/kubernetes/istio/gateway.yaml
```

Uninstall Istio only if this example created the installation and no other
workload uses it.

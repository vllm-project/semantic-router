# Semantic Router with llm-d scheduling

This example separates two routing decisions:

- Semantic Router chooses a model pool from request signals.
- The llm-d endpoint picker chooses a replica inside that pool.

```text
client -> Istio Gateway -> Semantic Router model choice
                              |
                              +-> HTTPRoute -> InferencePool -> llm-d EPP -> replica
```

The manifests create one `InferencePool` and one endpoint-picker deployment for
each of the `llama3-8b` and `phi4-mini` aliases. The included backend example
has one replica per pool, so it demonstrates the control-plane integration but
does not measure replica-scheduling gains.

For the maintained architecture and prerequisites, see the
[llm-d integration guide](../../../website/docs/installation/k8s/llm-d.md).
Complete the [Istio example](../istio/README.md) through model, Gateway, and
Router deployment before applying the llm-d-specific resources below.

## Apply the llm-d resources

Run from the repository root:

```bash
kubectl apply -f deploy/kubernetes/llmd-base/inferencepool-llama.yaml
kubectl apply -f deploy/kubernetes/llmd-base/inferencepool-phi4.yaml

kubectl apply -f deploy/kubernetes/llmd-base/dest-rule-epp-llama.yaml
kubectl apply -f deploy/kubernetes/llmd-base/dest-rule-epp-phi4.yaml

kubectl apply -f deploy/kubernetes/llmd-base/httproute-llama-pool.yaml
kubectl apply -f deploy/kubernetes/llmd-base/httproute-phi4-pool.yaml
```

Do not also apply the direct-Service routes from the Istio example. Both sets
match the same `x-selected-model` values and would create ambiguous ownership.

The endpoint-picker manifests currently use the model Services and labels from
the Istio example:

| Router alias | InferencePool | Backend selector |
| --- | --- | --- |
| `llama3-8b` | `vllm-llama3-8b-instruct` | `app: vllm-llama3-8b-instruct` |
| `phi4-mini` | `vllm-phi4-mini` | `app: phi4-mini` |

If you change a model alias, update the canonical Router Model, `HTTPRoute` header
match, `InferencePool`, EPP arguments, and backend labels as one contract.

## Verify the two layers

First check Kubernetes status:

```bash
kubectl get inferencepools
kubectl get httproutes
kubectl get deployments,services
kubectl get pods --namespace vllm-semantic-router-system
```

Inspect route and pool conditions rather than comparing pod names with a saved
example:

```bash
kubectl describe httproute vsr-llama8b
kubectl describe inferencepool vllm-llama3-8b-instruct
```

For minikube, obtain the current gateway URL and send an automatic-selection
request:

```bash
GATEWAY_URL="$(minikube service inference-gateway-istio --url)"
curl --fail-with-body "$GATEWAY_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Explain a Linux process."}],
    "max_tokens": 64,
    "temperature": 0
  }'
```

Use `model: llama3-8b` and `model: phi4-mini` to separate explicit route
resolution from semantic model choice. Add multiple backend replicas only when
you need to validate endpoint selection inside a pool.

## Diagnose failures

- `HTTPRoute` has unresolved references: verify the installed Inference
  Extension API version and the `InferencePool` name.
- EPP is not ready: inspect the EPP deployment logs and its RBAC resources.
- Requests bypass a pool: remove the direct-Service route for the same selected
  model.
- Wrong model pool: compare the Router alias and emitted `x-selected-model`
  value with the `HTTPRoute` header match.

```bash
kubectl logs deployment/llm-d-inference-scheduler-llama3-8b
kubectl logs deployment/inference-gateway-istio -c istio-proxy
kubectl logs deployment/semantic-router \
  --namespace vllm-semantic-router-system
```

## Cleanup

```bash
kubectl delete -f deploy/kubernetes/llmd-base/httproute-phi4-pool.yaml
kubectl delete -f deploy/kubernetes/llmd-base/httproute-llama-pool.yaml
kubectl delete -f deploy/kubernetes/llmd-base/dest-rule-epp-phi4.yaml
kubectl delete -f deploy/kubernetes/llmd-base/dest-rule-epp-llama.yaml
kubectl delete -f deploy/kubernetes/llmd-base/inferencepool-phi4.yaml
kubectl delete -f deploy/kubernetes/llmd-base/inferencepool-llama.yaml
```

Clean up the shared Router, Gateway, and model resources with the Istio guide.

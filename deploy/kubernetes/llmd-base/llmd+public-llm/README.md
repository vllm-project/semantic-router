# Local and hosted model routing example

This example extends the llm-d topology with a hosted OpenAI-compatible
backend. Semantic Router can choose either:

- `llama3-8b`, backed by a local `InferencePool` and llm-d endpoint picker; or
- `gpt-4o-mini`, represented by an Istio egress Service and `HTTPRoute`.

It is an integration example, not a recommended secret-management pattern. The
included `httproute-openai.template` contains an authorization placeholder;
substituting a credential into that manifest stores the credential in the
Kubernetes API and may expose it through Git, shell history, logs, or users who
can read `HTTPRoute` objects. In a shared cluster, inject provider credentials
through a gateway extension or egress proxy that reads a Kubernetes Secret.

## Before you apply it

Complete these two examples first:

1. [Istio Gateway](../../istio/README.md) for the Gateway, Router, and local
   model.
2. [llm-d scheduling](../README.md) for the local `InferencePool` and endpoint
   picker.

Review these files together:

| File | Purpose |
| --- | --- |
| `config.yaml.openai` | Canonical Models, Recipe decisions, and Entrypoint assignments. |
| `svc-openai.yaml` | Kubernetes `ExternalName` Service for `api.openai.com`. |
| `svc-entry-openai.yaml` | Istio egress registration. |
| `dest-rule-openai.yaml` | TLS origination for the hosted endpoint. |
| `httproute-openai.template` | Selected-model match and demo auth header. |

Hosted models and credentials may incur cost and transfer request content to a
third party. Confirm data-governance requirements before enabling the route.

## Configure the Router

Do not overwrite a working config without saving it. Copy the example, review
it, and update model/provider IDs to match the account you intend to use:

```bash
cp deploy/kubernetes/istio/config.yaml /tmp/vsr-istio-config.yaml
cp deploy/kubernetes/llmd-base/llmd+public-llm/config.yaml.openai \
  deploy/kubernetes/istio/config.yaml
kubectl apply -k deploy/kubernetes/istio/
```

Restore `/tmp/vsr-istio-config.yaml` after the example if the Istio config is a
tracked local customization.

## Apply egress resources

```bash
kubectl apply -f deploy/kubernetes/llmd-base/llmd+public-llm/svc-openai.yaml
kubectl apply -f deploy/kubernetes/llmd-base/llmd+public-llm/svc-entry-openai.yaml
kubectl apply -f deploy/kubernetes/llmd-base/llmd+public-llm/dest-rule-openai.yaml
```

Then create `vsr-openai-g4` using your cluster's approved credential-injection
mechanism. The checked-in template shows the required route shape:

- parent Gateway: `inference-gateway`;
- selected-model match: `gpt-4o-mini`;
- backend: `openai-external:443`;
- upstream Host header: `api.openai.com`.

Do not commit a rendered route or print its authorization header. If you use
the template in an isolated throwaway cluster, treat the rendered file and the
Kubernetes object as secrets, remove them immediately after the test, and
rotate the credential.

The local route remains the pool-backed route from the parent example:

```bash
kubectl apply -f deploy/kubernetes/llmd-base/httproute-llama-pool.yaml
```

## Verify behavior

```bash
kubectl get httproutes
kubectl describe httproute vsr-openai-g4
kubectl describe httproute vsr-llama8b
```

Send one explicit request to each alias before testing `model: auto`. Obtain
the current gateway URL instead of copying an address from another cluster:

```bash
GATEWAY_URL="$(minikube service inference-gateway-istio --url)"
curl --fail-with-body "$GATEWAY_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Reply with OK"}],
    "max_tokens": 8
  }'
```

An HTTP success proves connectivity, not that the request used the intended
backend. Check the Router routing headers or trace and the gateway access log;
do not infer provider identity from response wording.

## Cleanup

Delete the hosted route before the supporting egress resources:

```bash
kubectl delete httproute vsr-openai-g4 --ignore-not-found
kubectl delete -f deploy/kubernetes/llmd-base/llmd+public-llm/dest-rule-openai.yaml
kubectl delete -f deploy/kubernetes/llmd-base/llmd+public-llm/svc-entry-openai.yaml
kubectl delete -f deploy/kubernetes/llmd-base/llmd+public-llm/svc-openai.yaml
```

Remove any rendered credential-bearing file, rotate any credential exposed to
the Kubernetes object, and restore the previous Router config.

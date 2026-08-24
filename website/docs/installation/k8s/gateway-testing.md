---
title: Test a Kubernetes Gateway Deployment
sidebar_label: Test a Gateway Deployment
description: Verify gateway reachability, direct model requests, semantic routing, headers, and backend selection.
---

# Test a Kubernetes Gateway Deployment

Use this checklist after installing Semantic Router behind Istio, Envoy
Gateway, Gateway API Inference Extension, or another supported gateway.

## 1. Resolve the real gateway address

Do not copy an IP or NodePort from another cluster. Inspect the Service created
for your Gateway:

```bash
kubectl get gateway -A
kubectl get service -A | grep -i gateway
```

For Minikube, the helper can print a reachable URL:

```bash
export GATEWAY_URL="$(minikube service inference-gateway-istio --url | head -n 1)"
```

For a LoadBalancer Service, use its assigned hostname or IP. For a local-only
cluster, port-forward the Gateway Service and set `GATEWAY_URL` to the forwarded
address.

```bash
test -n "$GATEWAY_URL" && printf 'Gateway: %s\n' "$GATEWAY_URL"
```

## 2. Check Gateway API status

```bash
kubectl get gateway,httproute -A
kubectl describe httproute <route-name> -n <namespace>
```

The route should be accepted and its backend references resolved. In an LLM-D
deployment, also inspect the `InferencePool` and EPP scheduler selected by each
route.

## 3. List exposed models

```bash
curl -fsS "$GATEWAY_URL/v1/models"
```

Confirm that the response contains the physical or virtual model names expected
by the active Router configuration.

## 4. Send a direct-model request

Replace `physical-model` with a provider model exposed by the config:

```bash
curl -fsS -D /tmp/direct-headers.txt \
  "$GATEWAY_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "physical-model",
    "messages": [{"role": "user", "content": "Reply with one short sentence."}],
    "max_tokens": 64,
    "temperature": 0
  }'
```

A direct provider name should reach that provider without running recipe
signals, decisions, route plugins, cache, learning, or session routing.

## 5. Send a routed request

Replace `virtual-model` with an entrypoint or automatic alias configured for the
deployment:

```bash
curl -fsS -D /tmp/routed-headers.txt \
  "$GATEWAY_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "virtual-model",
    "messages": [{"role": "user", "content": "Explain why 2 + 2 equals 4."}],
    "max_tokens": 128,
    "temperature": 0
  }'
```

Inspect the response and routing headers supported by your deployment. Confirm
that the selected provider is valid for the matched decision; do not assume a
particular category or backend solely from the prompt wording.

## 6. Verify the backend path

Correlate the request with Gateway, Router, and provider logs. A successful HTTP
response alone does not prove that the intended route or scheduler handled it.

```bash
kubectl logs -n <router-namespace> deployment/<router-deployment> --since=5m
kubectl logs -n <gateway-namespace> deployment/<gateway-deployment> --since=5m
```

Keep request and response bodies out of shared logs when they may contain
sensitive data.

## Failure guide

| Symptom | Check first |
|---------|-------------|
| No external connection | Gateway Service address, LoadBalancer/NodePort, firewall |
| HTTPRoute not accepted | Parent reference, listener hostname, allowed routes |
| Backend reference unresolved | Service/InferencePool name, namespace, port |
| `/v1/models` works but completions fail | provider readiness, served model name, credentials |
| Direct request works but virtual model fails | entrypoint, recipe, signals/decisions, default model |
| Response succeeds through wrong pool | route match, selected-model headers, EPP/Gateway logs |

Delete temporary header files after inspection if they may contain sensitive
metadata.

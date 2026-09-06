---
title: Integrate with llm-d
description: Let Semantic Router choose a model pool while llm-d schedules a replica inside that pool.
---

# Integrate with llm-d

Use this topology when a request must make two different choices:

1. **Semantic Router** chooses the logical model or model pool from request
   intent, policy, and recipe state.
2. **llm-d** chooses a healthy replica inside that pool using load, prefix
   cache, or other endpoint information.

Do not configure both systems to make the same decision. Semantic Router should
not choose a Pod, and llm-d should not decide which business policy or model
family applies to the request.

```text
client
  -> inference gateway
  -> Semantic Router ExtProc
  -> HTTPRoute selected by x-selected-model
  -> InferencePool
  -> llm-d endpoint picker
  -> model replica
```

## Before you start

Deploy and verify llm-d independently before adding Semantic Router. llm-d's
release artifacts, CRDs, charts, and plugin configuration evolve together, so
use one supported llm-d release rather than mixing copied manifests from
different versions.

- Follow the current [llm-d quickstart](https://llm-d.ai/docs/getting-started/quickstart)
  or a suitable well-lit path.
- Choose a supported [gateway integration](https://llm-d.ai/docs/infrastructure/gateway).
- Use the [llm-d artifacts reference](https://llm-d.ai/docs/api-reference/artifacts)
  for the matching Gateway API Inference Extension resources and charts.

At this boundary you should already be able to send a request through the
gateway to each `InferencePool` without Semantic Router.

## Integration contract

Keep these names aligned across the two systems:

| Name | Owner | Requirement |
| --- | --- | --- |
| Provider model | Semantic Router | `providers.models[].name` is the logical pool name selected by a recipe. |
| Request header | Semantic Router | The Router writes the selected provider model to `x-selected-model`. |
| Route match | Gateway API | An `HTTPRoute` matches that exact header value. |
| Backend reference | Gateway API / llm-d | The route targets the intended `InferencePool`. |
| Served model | Model server | The pool's replicas accept the model identity forwarded by the gateway. |

For example, this route maps the Router's `local/code` selection to an existing
llm-d pool. Adjust names and namespaces to your deployment:

```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: code-pool
  namespace: inference
spec:
  parentRefs:
    - name: llm-d-inference-gateway
  rules:
    - matches:
        - headers:
            - type: Exact
              name: x-selected-model
              value: local/code
      backendRefs:
        - group: inference.networking.k8s.io
          kind: InferencePool
          name: code-pool
          port: 8000
```

The route does not configure the pool or endpoint picker. Those remain part of
the llm-d deployment and must use the API version supported by that release.

## Add Semantic Router

1. Create a complete canonical Router config whose provider model names match
   the route values. Validate it before deployment:

   ```bash
   vllm-sr validate --config config.yaml
   ```

2. Deploy Semantic Router with the Helm or Operator workflow described in
   [Configuration workflows](../configuration-workflows). For direct Helm,
   use `configOverride` so the chart sample config is replaced atomically.

3. Attach Semantic Router to the gateway as an ExtProc service. The exact
   resource is gateway-specific; use
   [Gateway API Inference Extension](gateway-api-inference-extension) for the
   supported attachment patterns.

4. Keep `global.router.clear_route_cache: true` in the canonical Router config.
   The gateway must call ExtProc in the downstream request path, preserve its
   route-cache clearing response, and then re-evaluate the `HTTPRoute` after
   Semantic Router writes `x-selected-model`.

## Verify one layer at a time

First check resource status without relying on generated Pod names:

```bash
kubectl get gateway,httproute -A
kubectl get inferencepools -A
kubectl get httproute code-pool -n inference \
  -o jsonpath='{.status.parents[*].conditions[?(@.type=="ResolvedRefs")].status}{"\n"}'
```

Then send a request using a virtual model exposed by your active Router config:

```bash
curl -i "$GATEWAY_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "vllm-sr/auto",
    "messages": [{"role": "user", "content": "Review this function for a race condition."}]
  }'
```

Verify all three decisions rather than stopping at HTTP 200:

- the response contains the expected `x-vsr-selected-model` value;
- the matching `HTTPRoute` reports `ResolvedRefs=True`; and
- llm-d selected a ready endpoint from the intended `InferencePool`.

## Common failures

| Symptom | Check |
| --- | --- |
| Route never matches | Compare `x-selected-model` with the `HTTPRoute` value, including case and namespace. |
| `ResolvedRefs=False` | Check the `InferencePool` name, group, port, and cross-namespace permissions. |
| Correct pool, wrong served model | Align the provider's model identity with the model name accepted by the replicas. |
| Semantic Router is bypassed | Confirm the gateway invokes ExtProc before route matching. |
| EPP has no endpoints | Diagnose the llm-d pool selector, Pod readiness, and release-matched plugin config. |

## Production boundaries

- Pin Semantic Router, llm-d, the gateway, CRDs, and model-server images.
- Keep provider credentials out of Router YAML; use Secret-backed bindings at
  the component that owns the credential.
- Define failure behavior explicitly. A fail-open ExtProc policy can bypass
  semantic policy; a fail-closed policy can stop all traffic when the Router is
  unavailable.
- Test direct pool access, semantic selection, and endpoint scheduling
  separately before a combined rollout.

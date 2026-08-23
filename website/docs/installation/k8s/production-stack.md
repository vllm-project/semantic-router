---
title: Integrate with vLLM Production Stack
description: Connect Semantic Router model-pool selection to model services managed by vLLM Production Stack.
---

# Integrate with vLLM Production Stack

Use this topology when vLLM Production Stack already owns model deployment,
service discovery, and replica scheduling, while Semantic Router should choose
the model pool from request meaning and policy.

This page describes the integration contract. Production Stack releases and
Helm values change independently, so install it with the current
[Production Stack documentation](https://github.com/vllm-project/production-stack)
instead of copying a frozen chart configuration from this guide.

## Responsibility split

| Component | Owns |
| --- | --- |
| Semantic Router | Signals, decisions, model-pool selection, and recipe-scoped plugins. |
| vLLM Production Stack | Model servers, service discovery, replica scheduling, and inference lifecycle. |
| Gateway | Client traffic and the ExtProc connection to Semantic Router. |

Semantic Router selects an eligible model. Production Stack then chooses a
replica serving that model. PII, jailbreak, or other signals affect traffic
only when your decisions and plugins enforce a policy; detection alone does
not block a request.

## Before you begin

You need:

- a working Production Stack deployment with at least one OpenAI-compatible
  model endpoint;
- stable Kubernetes service names for those endpoints;
- a Gateway that can call Semantic Router through ExtProc; and
- `kubectl`, Helm, model credentials, and enough inference capacity.

Do not bind a Router provider to a Service `ClusterIP`. Use Kubernetes DNS so
the configuration survives Service recreation.

## 1. Verify the model service

Follow the upstream installation guide, then record the model name, namespace,
Service name, and port:

```bash
kubectl get services -A
kubectl get pods -A
```

Send a direct Chat Completions request to the Production Stack endpoint before
adding Semantic Router. This separates backend or scheduler failures from
semantic-routing failures.

## 2. Bind the model in canonical configuration

Create one Semantic Router Model for each model pool you want policy to select.
A Kubernetes connection uses this shape:

```yaml
models:
  - name: production/qwen3
    card: {capabilities: [chat]}
    connections:
      - provider: vllm
        endpoint: http://vllm-router-service.default.svc.cluster.local:80/v1
        model: Qwen/Qwen3-8B
```

Replace the endpoint and model identifier with values from your deployment.
If Production Stack exposes different services per model, create a binding for
each service. If it exposes one multi-model service, keep distinct provider
Models and use the backend's served model identifiers.

Add model cards, decisions, and entrypoints that reference these provider
names, then validate the complete document:

```bash
vllm-sr validate --config config.yaml
```

## 3. Deploy Semantic Router

Use [Configuration Workflows](../configuration-workflows#helm) to deploy the
validated config with `configOverride`, then attach one of the supported
[Kubernetes gateways](ai-gateway). Pin chart and image versions for production;
the development `0.0.0-latest` chart is for testing current main.

The upstream
[Semantic Router integration tutorial](https://github.com/vllm-project/production-stack/blob/main/tutorials/24-semantic-router-integration.md)
can provide additional context, but review its image tags and values against
the current Production Stack release before applying them.

## 4. Verify both routing layers

1. Send a direct request to each model service.
2. Send the same request through the Gateway using a configured virtual model.
3. Inspect the Semantic Router selection headers.
4. Confirm Production Stack sent the request to a replica for the selected
   model pool.

Use [Test a Kubernetes Gateway Deployment](gateway-testing) for the common
Gateway checks. A successful semantic decision does not prove that the selected
model is ready, so retain both direct and routed generation tests.

## Cleanup

Remove Semantic Router and Gateway resources with the commands from their
respective guides. Remove Production Stack with the release name and namespace
you selected during its installation; do not copy a cleanup command that was
written for a different release.

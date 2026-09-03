---
title: SemanticRouter CRD Reference
sidebar_label: SemanticRouter CRD
description: Top-level field guide for the vllm.ai/v1alpha1 SemanticRouter custom resource.
---

# SemanticRouter CRD Reference

`SemanticRouter` is the Operator-owned resource for deploying a Router and
binding it to Kubernetes model services.

```yaml
apiVersion: vllm.ai/v1alpha1
kind: SemanticRouter
metadata:
  name: my-router
spec: {}
```

This page summarizes the top-level contract. The installed CRD is authoritative
for nested OpenAPI validation and defaults:

```bash
kubectl explain semanticrouter.spec --recursive
```

The source schema is available in
[`semanticrouter_types.go`](https://github.com/vllm-project/semantic-router/blob/main/deploy/operator/api/v1alpha1/semanticrouter_types.go)
and the generated CRD in
[`vllm.ai_semanticrouters.yaml`](https://github.com/vllm-project/semantic-router/blob/main/deploy/operator/config/crd/bases/vllm.ai_semanticrouters.yaml).

## Top-level `spec` fields

| Field | Purpose |
|-------|---------|
| `image` | Router image repository, tag, registry prefix, and pull policy. |
| `replicas` | Fixed replica count when autoscaling is not controlling replicas. |
| `imagePullSecrets` | Registry credentials referenced by name. |
| `serviceAccount` | Create or select the workload ServiceAccount. |
| `service` | Service type and API, gRPC, and metrics ports. |
| `resources` | Container CPU, memory, and other resource requests/limits. |
| `persistence` | Model-storage PVC settings or an existing claim. |
| `config` | Operator configuration adapters and canonical `routing` object. |
| `toolsDb` | Function-tool records materialized for Router tool selection. |
| `vllmEndpoints` | Model services discovered as canonical providers and model cards. |
| `autoscaling` | HPA enablement and CPU/memory targets. |
| `startupProbe`, `livenessProbe`, `readinessProbe` | Workload probe tuning. |
| `securityContext`, `podSecurityContext` | Container and pod security settings. |
| `podAnnotations` | Additional annotations, including scraper integration. |
| `nodeSelector`, `tolerations`, `affinity` | Pod scheduling constraints. |
| `env`, `args` | Additional Router environment variables and arguments. |
| `gateway` | Reference to an existing Kubernetes Gateway. |
| `openshift` | OpenShift Route behavior. |
| `ingress` | Kubernetes Ingress configuration. |

## `vllmEndpoints`

Each entry creates one logical provider model and one backend reference:

```yaml
spec:
  vllmEndpoints:
    - name: qwen-backend
      model: qwen/assistant
      reasoningFamily: qwen3
      backend:
        type: service
        service:
          name: qwen-vllm
          namespace: model-serving
          port: 8000
      weight: 1
```

Supported backend types are `service`, `kserve`, and `llamastack`. Optional
`loras` declare adapters exposed by the generated routing model card. The first
resolved model becomes the default unless configuration overrides it.

## `config`

The Operator always renders a canonical v0.3 Router document. Its configuration
surface has two parts:

- `config.routing` passes through the canonical routing object, including model
  cards, signals, projections, decisions, algorithms, and route plugins;
- typed adapter fields such as `response_cache`, `tools`, `prompt_guard`,
  `classifier`, `complexity_rules`, `reasoning_families`, `api`, and
  `observability` are translated into their canonical provider or `global`
  locations.

Do not assume an arbitrary local `config.yaml` key is valid directly under
`spec.config`. Use the CRD schema and
[Configuration Workflows](../installation/configuration-workflows) when moving
between local YAML and the Operator.

The deprecated `semantic_cache` adapter is retained for compatibility;
`response_cache` is the canonical field. Do not set both.

## Status

`status` reports the observed generation, replica counts, conditions, phase,
gateway mode, and detected OpenShift features. Controllers and automation
should use conditions and `status.observedGeneration`, not only the phase.

```bash
kubectl get semanticrouter <name> -o jsonpath='{.status.conditions}'
kubectl get semanticrouter <name> -o jsonpath='{.status.observedGeneration}'
```

## Related guides

- [Deploy with the Kubernetes Operator](../installation/k8s/operator)
- [Operate an Operator deployment](../installation/k8s/operator-operations)
- [Configuration](../installation/configuration)

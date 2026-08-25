---
title: SemanticRouter CRD Reference
sidebar_label: SemanticRouter CRD
description: Top-level field guide for the vllm.ai/v1alpha1 SemanticRouter deployment resource.
---

# SemanticRouter CRD Reference

`SemanticRouter` is the Operator-owned Kubernetes deployment resource. It
selects an immutable Router bootstrap and configures workload concerns; it is
not a second Router configuration language.

```yaml
apiVersion: vllm.ai/v1alpha1
kind: SemanticRouter
metadata:
  name: my-router
spec:
  bootstrap:
    configMapRef:
      name: my-router-bootstrap-v1
      key: config.yaml
```

The installed CRD is authoritative for nested OpenAPI validation and defaults:

```bash
kubectl explain semanticrouter.spec --recursive
```

The source schema is available in
[`semanticrouter_types.go`](https://github.com/vllm-project/semantic-router/blob/main/deploy/operator/api/v1alpha1/semanticrouter_types.go)
and the generated CRD in
[`vllm.ai_semanticrouters.yaml`](https://github.com/vllm-project/semantic-router/blob/main/deploy/operator/config/crd/bases/vllm.ai_semanticrouters.yaml).

## Top-level `spec` fields

| Field | Purpose |
| --- | --- |
| `bootstrap` | Required immutable ConfigMap name and key containing the sole v0.3 Router startup manifest. |
| `image` | Router image repository, tag, registry prefix, and pull policy. |
| `replicas` | Fixed replica count when autoscaling is not controlling replicas. |
| `imagePullSecrets` | Registry credentials referenced by name. |
| `serviceAccount` | Create or select the workload ServiceAccount. |
| `service` | Inference Service type plus API, gRPC, private Management, and metrics ports. |
| `resources` | Container CPU, memory, and other resource requests and limits. |
| `persistence` | Model-storage PVC settings or an existing claim. |
| `autoscaling` | HPA enablement and CPU or memory targets. |
| `startupProbe`, `livenessProbe`, `readinessProbe` | Workload probe tuning. |
| `securityContext`, `podSecurityContext` | Container and Pod security settings. |
| `podAnnotations` | Additional Pod annotations. |
| `nodeSelector`, `tolerations`, `affinity` | Pod scheduling constraints. |
| `env`, `envFrom`, `args` | Additional environment sources and Router process arguments. |
| `volumes`, `volumeMounts` | Deployment-owned ConfigMap, Secret, CSI, or other volumes shared with the Router and migration Job. |
| `gateway` | Reference to an existing Kubernetes Gateway. |
| `openshift` | OpenShift Route behavior. |
| `ingress` | Kubernetes Ingress configuration. |
| `podDisruptionBudget` | Disruption protection; defaults on when a Management store is configured. |
| `topologySpread` | One portable failure-domain spread constraint with the same capability-derived default. |
| `networkPolicy` | Listener-specific ingress peers; defaults on and fails closed with a Management store. |

## `bootstrap.configMapRef`

The reference has two required fields:

```yaml
spec:
  bootstrap:
    configMapRef:
      name: my-router-bootstrap-v1
      key: config.yaml
```

The ConfigMap must:

- exist in the same namespace as the `SemanticRouter`;
- set `immutable: true`;
- contain the selected `data` key; and
- contain one v0.3 manifest.

The selected key is projected read-only to `/app/config.yaml`. `spec.args`
cannot override `--config`, and `spec.env` cannot override `CONFIG_FILE`.
Operator-owned volume names and the bootstrap mount path cannot be overridden.

The manifest may contain Models, Recipes, and Entrypoints. Without a Management
store the file remains authoritative. With a Management store it seeds an empty
database once and the Management API owns subsequent changes. Updating the
ConfigMap reference changes the Pod template and causes a rollout. The Operator
does not perform in-place file reloads.

Configuring a Management store requires exactly one
`global.stores.management.postgres.dsn_env` or `.dsn_file`. The Operator passes
that reference—not the DSN value—to an explicit schema migration Job and gates
the Router rollout on Job completion.

## Durable Services and isolation

`spec.service.type` applies only to the inference Service named after the
`SemanticRouter`. Enabling the Management API creates `<name>-management`;
configuring a Management store creates `<name>-backend-dispatch`; metrics use
`<name>-metrics`. These additional Services are private ClusterIP Services.

`networkPolicy.inferencePeers`, `.managementPeers`, and `.metricsPeers` map to
only their corresponding listeners. Omitted peer families stay denied. The
backend-dispatch listener permits only Pods belonging to the same
`SemanticRouter`.

A Management store defaults `podDisruptionBudget.enabled`,
`topologySpread.enabled`, and `networkPolicy.enabled` to true. File-only
deployments default each to false. An explicit `enabled` value overrides the
derived default.

## Status

`status` reports the observed generation, replica counts, conditions, phase,
gateway topology, immutable bootstrap digest, public and
Management Service names, migration state, and detected OpenShift features.
Automation should use conditions and `status.observedGeneration`, not only
phase.

```bash
kubectl get semanticrouter <name> -o jsonpath='{.status.conditions}'
kubectl get semanticrouter <name> -o jsonpath='{.status.observedGeneration}'
kubectl get semanticrouter <name> -o jsonpath='{.status.migration}'
```

Durable rollout conditions are:

- `BootstrapReady`: the selected immutable v0.3 bootstrap passed the
  deployment-boundary checks;
- `MigrationReady`: the content-addressed schema Job succeeded; and
- `Available`: the workload is ready and no migration or reconciliation gate
  is active.

## Related guides

- [Deploy with the Kubernetes Operator](../installation/k8s/operator)
- [Operate an Operator deployment](../installation/k8s/operator-operations)
- [Configuration](../installation/configuration)

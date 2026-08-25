---
sidebar_position: 3
sidebar_label: Kubernetes Operator
title: Deploy with the Kubernetes Operator
description: Install the Semantic Router Operator and deploy a v0.3 Router bootstrap with optional durable management and access.
---

# Deploy with the Kubernetes Operator

The Semantic Router Operator owns the Kubernetes lifecycle of Router
workloads. It creates deployment, service, availability, isolation, storage,
autoscaling, ingress, and optional platform resources declared by a
`SemanticRouter` resource.

Router configuration stays separate. Every deployment selects exactly one
immutable v0.3 ConfigMap key with `spec.bootstrap.configMapRef`. The file may
contain Models, Recipes, and Entrypoints. Without a Management store it is the
routing authority. With `global.stores.management.postgres`, an empty database
is seeded once from the file and subsequent desired-state changes use the
Management API.

The Operator never discovers model servers or synthesizes Router resources.

## Prerequisites

- a supported Kubernetes or OpenShift cluster;
- `kubectl` or `oc` configured for the target cluster;
- Git, GNU Make, and Go 1.25 or newer for a source install;
- permission to install CRDs and cluster-scoped RBAC; and
- model endpoints reachable under the egress policy in the Router manifest.

Durable management needs reachable PostgreSQL. Router-native API keys and
global quotas additionally need Valkey and the required security material. See
[Router-Native Access Control Deployment](../../proposals/router-native-access-control-deployment#kubernetes-deployment).

## Install the Operator

```bash
git clone https://github.com/vllm-project/semantic-router.git
cd semantic-router/deploy/operator

make install
make deploy IMG=ghcr.io/vllm-project/semantic-router/operator:latest
```

Verify the controller:

```bash
kubectl get pods -n semantic-router-operator-system
kubectl logs -n semantic-router-operator-system \
  deployment/semantic-router-operator-controller-manager
kubectl explain semanticrouter.spec.bootstrap
```

Pin a released image tag or digest in controlled environments.

## Create a file-backed Router

Create the immutable manifest and the workload declaration as separate
objects. The ConfigMap comes first so reconciliation can validate it
immediately:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: my-router-bootstrap-v1
  namespace: default
immutable: true
data:
  config.yaml: |
    version: v0.3
    providers:
      models:
        - name: local-model
          provider_model_id: local-model
          api_format: openai
          backend_refs:
            - provider: openai-compatible
              endpoint: http://model-server.default.svc.cluster.local:8000
    routing:
      modelCards:
        - name: local-model
          capabilities: [chat]
          modality: ar
    recipes:
      - name: default
        routing:
          decisions:
            - name: Answer
              rules: {}
    entrypoints:
      - model_names: [assistant]
        recipe: default
        assignments:
          Answer:
            models:
              - model: local-model
---
apiVersion: vllm.ai/v1alpha1
kind: SemanticRouter
metadata:
  name: my-router
  namespace: default
spec:
  bootstrap:
    configMapRef:
      name: my-router-bootstrap-v1
      key: config.yaml
  replicas: 1
  resources:
    requests:
      cpu: 500m
      memory: 1Gi
    limits:
      cpu: "2"
      memory: 4Gi
```

Apply and wait for readiness:

```bash
kubectl apply -f my-router.yaml
kubectl get semanticrouter my-router -w
kubectl get deployment,service \
  -l app.kubernetes.io/instance=my-router
```

The selected key is mounted read-only as `/app/config.yaml`. The Operator
checks that the ConfigMap is immutable, the key exists, and the manifest is
v0.3. Router startup
performs full schema, provider, Recipe, and Entrypoint validation.

## Update the bootstrap

Kubernetes does not mutate an immutable ConfigMap. Create a new object and move
the reference:

```yaml
spec:
  bootstrap:
    configMapRef:
      name: my-router-bootstrap-v2
      key: config.yaml
```

The reference is part of the Pod template, so the Operator performs a normal
rollout. It does not watch files or custom routing resources for in-process
reloads. File-authoritative routing changes with that rollout. When a
Management store is configured, use the Management API and published snapshots
for routine routing changes instead.

## Add durable management and access

Add `global.stores.management.postgres` to enable durable routing resources.
Set `global.services.management_api.enabled: true` when the private Management
listener is required. Add `global.stores.runtime.redis` plus
`global.services.access.enabled: true` for API keys, grants, global quotas,
settlement, usage, and audit. Users, Teams, keys, grants, and quota policies are
always dynamic resources; they never enter the ConfigMap. Models, Recipes, and
Entrypoints may remain as the initial seed.

Mount credentials from Kubernetes Secrets through `spec.env`, `spec.envFrom`,
or additional Secret volumes. The public ConfigMap contains only environment
or file references. The maintained durable sample shows the object boundary; the
[deployment proposal](../../proposals/router-native-access-control-deployment#router-bootstrap-configuration)
defines every required durable bootstrap value.

```yaml
spec:
  env:
    - name: VLLM_SR_POSTGRES_DSN
      valueFrom:
        secretKeyRef:
          name: router-control-plane
          key: postgres-dsn
```

The bootstrap must select exactly one PostgreSQL migration source with
`global.stores.management.postgres.dsn_env` or `.dsn_file`. Reconciliation creates a
content-addressed schema Job with the Router image and waits for it to succeed.
Only then does the Operator create or roll the Router Deployment. Observe the
gate through `status.migration` and the `MigrationReady` condition.

A new durable installation has two readiness phases. When enabled, the private
Management Service becomes reachable after the Router process and stores are
healthy, while the inference Deployment remains unready. Router replicas first converge the unique
application-installed Provider Catalog through the configured rollout-group gate;
they never replace an existing desired or active catalog revision during startup.
Use the private Service from an authorized console or automation client to complete
identity bootstrap and publish the first Model, Recipe, Entrypoint, access policy,
and routing revision. `/ready` succeeds only after the catalog and coupled routing
revision are active on every required replica. Existing installations with active
revisions follow ordinary rollout waiting.

Durable deployments have distinct listener Services:

| Service | Exposure | Purpose |
| --- | --- | --- |
| `<name>` | `spec.service.type` | Inference and ExtProc traffic only. |
| `<name>-management` | Private `ClusterIP`, only when `management_api.enabled` | TLS Management API only. |
| `<name>-backend-dispatch` | Private `ClusterIP` | Router-to-Router dispatch only. |
| `<name>-metrics` | Private `ClusterIP` | Metrics when enabled. |

`spec.service.management.port` changes only the private Service port; the
container target remains the Management port in the immutable bootstrap.

PodDisruptionBudget, topology spread, and NetworkPolicy default on when a
Management store is configured and off for file-only deployments. An explicit
`enabled` value overrides that capability-derived default. NetworkPolicy is
fail closed: leaving a peer list empty does not allow that listener. Define
separate `inferencePeers`, `managementPeers`, and `metricsPeers` rather than a
single broad allow-list.

```yaml
spec:
  podDisruptionBudget:
    minAvailable: 2
  topologySpread:
    topologyKey: topology.kubernetes.io/zone
    whenUnsatisfiable: ScheduleAnyway
  networkPolicy:
    inferencePeers:
      - namespaceSelector:
          matchLabels:
            kubernetes.io/metadata.name: gateway-system
    managementPeers:
      - podSelector:
          matchLabels:
            app.kubernetes.io/component: console
```

## Gateway and OpenShift modes

Without `spec.gateway.existingRef`, the Operator deploys a local Envoy sidecar.
Client traffic enters the Service, Envoy invokes ExtProc, and the Router's
backend invoker dispatches the selected Model.

To use an existing Gateway:

```yaml
spec:
  gateway:
    existingRef:
      name: shared-gateway
      namespace: gateway-system
```

The Operator switches to gateway-integration mode but does not create an
`HTTPRoute`. Create that route separately and target the Router Service.

On OpenShift, `spec.openshift.routes.enabled` may create a Route. Choose its TLS
termination to match the Router listener contract.

## Verify the deployment

```bash
kubectl get semanticrouter my-router -o wide
kubectl describe semanticrouter my-router
kubectl get pods,service,pvc,job,pdb,networkpolicy \
  -l app.kubernetes.io/instance=my-router
kubectl logs deployment/my-router -c semantic-router
```

Verify that:

1. the referenced ConfigMap is immutable and contains the selected key;
2. `status.observedGeneration` matches `metadata.generation`;
3. deployments with a Management store report `MigrationReady=True`;
4. expected replicas are ready;
5. only intended peers can reach Management and metrics; and
6. a real request succeeds through each important Entrypoint.

## Next

- [Operate an Operator deployment](operator-operations)
- [SemanticRouter CRD reference](../../api/semantic-router-crd)
- [Configuration Workflows](../configuration-workflows)
- [Kubernetes Gateway API](gateway-api-inference-extension)
- [Security Hardening](../security-hardening)

# Semantic Router Operator

The Operator reconciles a `SemanticRouter` custom resource into a Router
workload and its Kubernetes deployment boundary. Router configuration has one
explicit source: the immutable ConfigMap selected by
`spec.bootstrap.configMapRef`.

The Operator does not author Models, Recipes, Entrypoints, provider
connections, access policy, or quota policy. Standalone deployments put their
immutable routing resources in the selected v0.4 manifest. Managed deployments
put only infrastructure bootstrap in that manifest and change desired state
through the Router Management API.

Managed reconciliation is deliberately ordered. A content-addressed migration
Job must complete before a new Router Deployment is created or rolled. The
Operator then creates separate inference, Management, backend-dispatch, and
metrics Services, plus mode-aware PodDisruptionBudget, topology-spread, and
NetworkPolicy resources. Management and backend-dispatch Services are always
private `ClusterIP` Services; `spec.service.type` applies only to inference.

User-facing installation and day-two operations are documented in the
[Operator guide](../../website/docs/installation/k8s/operator.md) and
[operations guide](../../website/docs/installation/k8s/operator-operations.md).
The generated field reference is
[`website/docs/api/semantic-router-crd.md`](../../website/docs/api/semantic-router-crd.md).

## Install from this checkout

Requirements:

- a Kubernetes cluster and matching `kubectl` context;
- Go and the container toolchain required by the Makefile;
- permission to install CRDs and cluster-scoped RBAC.

```bash
cd deploy/operator
make install
make docker-build docker-push \
  IMG=registry.example.com/semantic-router-operator:dev
make deploy IMG=registry.example.com/semantic-router-operator:dev
```

Inspect the controller and CRD before creating a Router:

```bash
kubectl get deployments --all-namespaces \
  -l app.kubernetes.io/name=semantic-router-operator
kubectl explain semanticrouter.spec.bootstrap
```

## Bootstrap a Router

Create an immutable ConfigMap first. The selected key is mounted read-only as
`/app/config.yaml` in every Router replica:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: router-bootstrap-v1
immutable: true
data:
  config.yaml: |
    version: v0.4
    global:
      control_plane:
        mode: standalone
    # Standalone Models, Recipes, and Entrypoints belong in this same file.
---
apiVersion: vllm.ai/v1alpha1
kind: SemanticRouter
metadata:
  name: router
spec:
  bootstrap:
    configMapRef:
      name: router-bootstrap-v1
      key: config.yaml
  replicas: 1
```

Apply one of the maintained examples:

| Sample | Contract |
| --- | --- |
| `vllm.ai_v1alpha1_semanticrouter_standalone.yaml` | Immutable v0.4 manifest with routing resources. |
| `vllm.ai_v1alpha1_semanticrouter_managed.yaml` | Infrastructure-only managed bootstrap. |

```bash
kubectl apply -f \
  config/samples/vllm.ai_v1alpha1_semanticrouter_standalone.yaml
kubectl get semanticrouters
```

The controller requires the referenced ConfigMap to set `immutable: true` and
to contain the selected key. It checks the v0.4 mode boundary before creating
the workload; full Router validation remains a Router startup responsibility.

Managed mode additionally requires exactly one PostgreSQL migration source at
`global.stores.access.postgres.dsn_env` or `.dsn_file`. The referenced
environment variable or mounted file is passed to the explicit migration Job;
the DSN value never enters the custom resource or ConfigMap.

To change an immutable bootstrap, create a new ConfigMap and update the
reference. In managed mode the new migration Job gates the rollout. In
standalone mode the resulting Pod-template change performs a normal rollout.
There is no in-place config reload or Operator-side routing synthesis.

## Deployment modes

Without `spec.gateway.existingRef`, the reconciled pod includes the local Envoy
sidecar that sends requests through the Router ExtProc service. With an
existing Gateway reference, the controller omits the local sidecar and connects
the Router to that Gateway. It does not create an `HTTPRoute`; apply and manage
that route separately.

On OpenShift, `spec.openshift.routes.enabled` may create a Route. Its TLS
termination must match the backend contract.

## Managed deployment controls

Managed mode enables disruption protection, topology spread, and ingress
isolation by default. NetworkPolicy is fail closed: an omitted peer family
remains denied. Supply only the peers that should reach each listener:

```yaml
spec:
  service:
    management:
      port: 8443
  podDisruptionBudget:
    minAvailable: 2
  topologySpread:
    topologyKey: kubernetes.io/hostname
    whenUnsatisfiable: DoNotSchedule
  networkPolicy:
    inferencePeers:
      - namespaceSelector:
          matchLabels:
            kubernetes.io/metadata.name: gateway-system
    managementPeers:
      - podSelector:
          matchLabels:
            app.kubernetes.io/component: console
    metricsPeers:
      - podSelector:
          matchLabels:
            app.kubernetes.io/component: monitoring
```

Use `spec.envFrom`, `spec.volumes`, and `spec.volumeMounts` to project Secret
material needed by both the Router and migration Job. Operator-owned volume
names and `/app/config.yaml` cannot be overridden.

## Develop and validate

```bash
cd deploy/operator

# Run against the current kubeconfig.
make run

# Unit/envtest checks.
make test

# Regenerate deepcopy, CRD, RBAC, and webhook artifacts after API changes.
make manifests generate
```

Keep these surfaces aligned after an API change:

1. `api/v1alpha1/semanticrouter_types.go`;
2. admission validation and tests;
3. controller reconciliation and tests;
4. CRD bases and bundle manifests;
5. samples; and
6. the generated website CRD reference.

Do not hand-edit a generated CRD as the source of a field change.

## Observe and remove

```bash
kubectl describe semanticrouter <name>
kubectl get job,pdb,networkpolicy \
  -l app.kubernetes.io/instance=<name>
kubectl get events --sort-by=.lastTimestamp
kubectl logs --all-containers \
  -l app.kubernetes.io/name=semantic-router-operator \
  --all-namespaces
```

Delete test `SemanticRouter` resources before removing the controller. Inspect
PVC and external-store retention before cleanup:

```bash
make undeploy
make uninstall
```

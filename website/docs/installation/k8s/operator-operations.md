---
title: Operate an Operator Deployment
sidebar_label: Kubernetes Operator
description: Monitor, update, scale, and troubleshoot a SemanticRouter managed by the Kubernetes Operator.
---

# Operate an Operator Deployment

This guide covers day-two work for a `SemanticRouter`. For installation and a
first deployment, see [Deploy with the Kubernetes Operator](operator).

## Read reconciliation status

```bash
kubectl get semanticrouter <name> -o wide
kubectl describe semanticrouter <name>
kubectl get semanticrouter <name> -o jsonpath='{.status.conditions}'
```

Use `metadata.generation`, `status.observedGeneration`, status conditions, and
ready replicas together. Inspect the workload when reconciliation stalls:

```bash
kubectl get deployment,pod,service,configmap,pvc,job,pdb,networkpolicy \
  -l app.kubernetes.io/instance=<name>
kubectl logs -n semantic-router-operator-system \
  deployment/semantic-router-operator-controller-manager
```

## Change configuration safely

`spec.bootstrap.configMapRef` points to the only Router startup manifest. The
referenced ConfigMap must be immutable, so configuration changes are explicit
releases:

1. copy the current manifest into a newly named ConfigMap;
2. validate the v0.3 manifest before applying it;
3. create the new immutable ConfigMap;
4. update `spec.bootstrap.configMapRef.name` or `.key`;
5. when a Management store is configured, wait for the content-addressed migration Job and
   `MigrationReady=True`;
6. wait for the Deployment rollout and Router readiness; and
7. send real requests through each important Entrypoint.

Keep the old immutable ConfigMap until rollback is no longer needed. Rolling
back means restoring the previous reference. The Router and Operator do not
reload a changed file in place.

With a Management store, the bootstrap may seed Models, Recipes, and
Entrypoints once. Use the Router Management API for later routing changes and
for identity, key, policy, and quota resources. Those changes publish immutable
generations without changing the Pod bootstrap reference.

## Scale and availability

Set fixed replicas through `spec.replicas`, or enable the HPA adapter:

```yaml
spec:
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 10
    targetCPUUtilizationPercentage: 70
```

Replicas with durable management share PostgreSQL desired state and ledger
data. Router-native access additionally shares Valkey projections and counters.
Do not scale either capability unless its configured stores are healthy and
reachable from every replica.

A Management store enables a PodDisruptionBudget and topology-spread constraint
by default. Tune or disable them explicitly when the availability target
requires different behavior:

```yaml
spec:
  podDisruptionBudget:
    enabled: true
    minAvailable: 2
  topologySpread:
    enabled: true
    maxSkew: 1
    topologyKey: topology.kubernetes.io/zone
    whenUnsatisfiable: ScheduleAnyway
```

The disruption budget protects voluntary evictions; it does not replace
multiple replicas or failure-domain placement.

## Listener isolation

The inference Service follows `spec.service.type`. Management,
backend-dispatch, and metrics Services remain private `ClusterIP` Services.
A Management store also enables an ingress NetworkPolicy by default.

Peer lists are listener-specific and fail closed. An empty `managementPeers`
list keeps the Management listener unreachable through the Pod network. The
backend-dispatch listener always accepts only same-`SemanticRouter` Pods. Test
both intended access and denied cross-namespace access after policy changes.

## Metrics and tracing

Router metrics are available on port 9190 by default:

```bash
kubectl port-forward service/<name> 9190:9190
curl -sS http://localhost:9190/metrics | head
```

Prometheus scraping and OpenTelemetry export are optional. Configure them in
the selected Router manifest; the Operator does not install a monitoring stack.
Keep trace sampling and captured attributes appropriate for request-data
sensitivity.

## Common failures

### Bootstrap ConfigMap is rejected

Confirm the reference, immutability bit, and selected key:

```bash
kubectl get semanticrouter <name> -o \
  jsonpath='{.spec.bootstrap.configMapRef}'
kubectl get configmap <bootstrap-name> -o yaml
```

The Operator accepts the same v0.3 routing seed in file-only and durable
deployments. Full manifest errors appear in Router startup logs after the
Operator's lightweight deployment-boundary checks pass.

### Pod did not roll after a change

An immutable ConfigMap cannot be edited. Verify that the
`spec.bootstrap.configMapRef` value itself changed and that the Deployment Pod
template references the new object:

```bash
kubectl get deployment <name> -o \
  jsonpath='{.spec.template.spec.volumes[?(@.name=="config-volume")].configMap}'
kubectl rollout status deployment/<name>
```

### Durable Router is not ready

Inspect the explicit migration gate first:

```bash
kubectl get semanticrouter <name> -o jsonpath='{.status.migration}'
kubectl get job -l app.kubernetes.io/instance=<name>
kubectl logs job/<migration-job-name>
```

Then check PostgreSQL, Valkey, referenced Secrets, schema migration completion,
Management listener TLS, publication acknowledgements, and backend egress
policy. On a fresh installation, also confirm that an authorized Management client
completed bootstrap and published the first coupled routing and policy revision. The
private Management Service remains reachable for this step while `/ready` is false;
the inference Service does not. Durable startup fails closed when shared authority is
unavailable.

### Gateway mode has no route

The Operator does not create an `HTTPRoute`. Verify the referenced Gateway and
apply a route targeting the Router Service:

```bash
kubectl get gateway -A
kubectl get httproute -A
```

### Pod is in `ImagePullBackOff`

Inspect pod events, verify the image reference, and provide an image pull
Secret when required:

```bash
kubectl describe pod <pod-name>
```

### PVC remains pending

Check the StorageClass, access mode, size, and provisioner:

```bash
kubectl get storageclass
kubectl describe pvc <pvc-name>
```

## Deletion and data

Deleting a `SemanticRouter` removes Operator-owned Kubernetes resources. It
does not delete external PostgreSQL, Valkey, model-serving, or observability
systems. Back up durable state and inspect PVC reclaim policies before deleting
claims or namespaces.

## References

- [Deploy with the Kubernetes Operator](operator)
- [SemanticRouter CRD reference](../../api/semantic-router-crd)
- [API and Observability](../../tutorials/global/api-and-observability)
- [Upgrade and Rollback](../upgrade-rollback)

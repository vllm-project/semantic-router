---
title: Operate an Operator Deployment
sidebar_label: Kubernetes Operator
description: Monitor, update, scale, and troubleshoot a SemanticRouter managed by the Kubernetes Operator.
---

# Operate an Operator Deployment

This guide covers the day-two tasks for a `SemanticRouter` resource. For
installation and a first deployment, see
[Deploy with the Kubernetes Operator](operator).

## Read reconciliation status

```bash
kubectl get semanticrouter <name> -o wide
kubectl describe semanticrouter <name>
kubectl get semanticrouter <name> -o jsonpath='{.status.conditions}'
```

Use `metadata.generation`, `status.observedGeneration`, status conditions, and
ready replicas together. A running controller does not mean the latest custom
resource generation has been applied successfully.

Inspect the owned workload when reconciliation stalls:

```bash
kubectl get deployment,pod,service,configmap,pvc \
  -l app.kubernetes.io/instance=<name>
kubectl logs -n semantic-router-operator-system \
  deployment/semantic-router-operator-controller-manager
```

## Update safely

1. Export the current custom resource and record the deployed image reference.
2. Review CRD and release notes for schema changes.
3. Apply the new custom resource or Operator version in a non-production
   environment.
4. Wait for the observed generation and ready replicas to converge.
5. Send real requests through every important entrypoint.
6. Roll back the custom resource or image reference if readiness or routing
   regresses.

Pin image tags or digests. Avoid changing the Operator, Router image, routing
policy, model pool, and storage backend in one rollout unless those changes are
intentionally coupled and tested together.

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

Use node affinity, tolerations, topology spread or anti-affinity, and a
separately managed PodDisruptionBudget to match your availability target.
Learning or other replica-local mutable state may require additional design;
do not assume that adding replicas makes every stateful routing feature
consistent.

## Metrics and tracing

The Router exposes Prometheus metrics on the configured metrics Service port
(9190 by default):

```bash
kubectl port-forward service/<name> 9190:9190
curl -sS http://localhost:9190/metrics | head
```

Configure a `ServiceMonitor` or equivalent scraper with the labels used by your
deployment. Enable OpenTelemetry through `spec.config.observability` and send
traces to a collector reachable from the Router namespace. Keep trace sampling
and captured attributes appropriate for the sensitivity of request data.

## Common failures

### Backend discovery fails

For `service` backends, verify the Service name, namespace, port, endpoints, and
network policy. For KServe, verify that the `InferenceService` is ready and its
predictor Service exists. For Llama Stack, inspect the labels on candidate
Services.

```bash
kubectl get service,endpoints -n <backend-namespace>
kubectl describe inferenceservice <name> -n <backend-namespace>
```

### Gateway mode has no route

The Operator currently does not create an `HTTPRoute`. Verify the referenced
Gateway, then apply a route that targets the Router Service on its API port.

```bash
kubectl get gateway -A
kubectl get httproute -A
```

### Pod is in `ImagePullBackOff`

Inspect pod events, confirm that the image exists, and provide an
`imagePullSecret` when the registry requires authentication.

```bash
kubectl describe pod <pod-name>
```

See [Restricted Network Environments](../../troubleshooting/network-tips) for
registry-mirror and cluster-egress guidance.

### PVC remains pending

Check that the requested StorageClass and access mode exist in the cluster and
that the provisioner can satisfy the requested size:

```bash
kubectl get storageclass
kubectl describe pvc <pvc-name>
```

Changing storage settings can affect existing data. Back up durable state and
follow the storage provider's migration procedure before replacing a claim.

### Model artifact download fails

Reference the download token from a Secret, check egress and certificate
configuration, and inspect Router logs. Do not put the token directly in
`spec.env` or a ConfigMap.

## Deletion and data

Before deleting a `SemanticRouter`, identify which state is ephemeral, which is
stored in a PVC, and which is held by external Redis, Valkey, Milvus, Qdrant, or
Postgres services. Deleting the custom resource removes managed Kubernetes
objects according to their ownership and retention policy; it does not
necessarily remove external data stores.

Back up durable data first, and inspect PVC reclaim policies before deleting
claims or namespaces.

## References

- [Deploy with the Kubernetes Operator](operator)
- [SemanticRouter CRD reference](../../api/semantic-router-crd)
- [API and Observability](../../tutorials/global/api-and-observability)
- [Upgrade and Rollback](../upgrade-rollback)

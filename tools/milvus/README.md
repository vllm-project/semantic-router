# Milvus installation validation

`test-milvus-deployment.sh` exercises the Kubernetes commands documented in
the public Milvus installation guide. It creates or reuses a Kind cluster,
installs the Milvus Helm chart, checks the service, applies test-only client
and network resources, and optionally removes the Helm release.

This is a deployment smoke test, not a production Milvus topology or load
test.

## Prerequisites

- `kubectl`, Kind, Helm, and Make
- enough local container capacity for the selected chart topology
- network access to the Milvus Helm repository and container images

## Run

Interactive mode prompts for deployment topology, cluster reuse, and cleanup:

```bash
./tools/milvus/test-milvus-deployment.sh
```

For a non-interactive standalone smoke that leaves resources available for
inspection:

```bash
MILVUS_MODE=standalone \
RECREATE_CLUSTER=false \
CLEANUP=false \
./tools/milvus/test-milvus-deployment.sh
```

| Variable | Values | Effect |
| --- | --- | --- |
| `MILVUS_MODE` | `standalone`, `cluster` | Selects the Milvus chart topology. |
| `RECREATE_CLUSTER` | `true`, `false` | Recreates the existing `semantic-router-cluster` Kind cluster when true. |
| `CLEANUP` | `true`, `false` | Uninstalls the Milvus release and removes test resources when true. |

Cluster mode deploys additional etcd, object-storage, and messaging
components and needs substantially more local capacity. `CLEANUP=true` does
not delete the Kind cluster itself.

## Diagnose a failure

The script prints pod, service, PVC, storage-class, NetworkPolicy, and recent
Milvus log state before cleanup. If the Helm install fails earlier, inspect the
same namespace directly:

```bash
kubectl get pods,svc,pvc -n vllm-semantic-router-system
kubectl get events -n vllm-semantic-router-system --sort-by=.lastTimestamp
```

The script disables `ServiceMonitor` creation so it does not require the
Prometheus Operator, and cluster mode selects Pulsar v3 explicitly.

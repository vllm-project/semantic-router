---
sidebar_position: 5
---

# Milvus Semantic Cache

This guide covers deploying Milvus as the semantic cache backend for the Semantic Router in Kubernetes. Milvus provides persistent, scalable vector storage compared to the default in-memory cache.

:::note
Milvus is optional. The router works with the default memory backend out of the box. Use Milvus when you need persistence, horizontal scaling, or cache sharing across router replicas.
:::

## Deployment Options

Two approaches are available:

- **Helm**: Quick start and parameterized deployments
- **Milvus Operator**: Production-grade lifecycle management, rolling upgrades, health checks, and dependency orchestration

## Prerequisites

- Kubernetes cluster with `kubectl` configured
- Default `StorageClass` available
- Helm 3.x installed

:::note[ServiceMonitor Requirement]
The default Helm values enable ServiceMonitor for Prometheus metrics collection, which requires [Prometheus Operator](https://github.com/prometheus-operator/prometheus-operator) to be installed first.

**For testing without Prometheus Operator**, disable ServiceMonitor using `--set metrics.serviceMonitor.enabled=false` (see deployment commands below).
:::

## Deploy with Helm

### Standalone Mode

Suitable for development and small-scale deployments:

```bash
helm repo add milvus https://zilliztech.github.io/milvus-helm/
helm repo update
```

**Without Prometheus Operator** (for testing/development):

```bash
helm install milvus-semantic-cache milvus/milvus \
  --set cluster.enabled=false \
  --set etcd.replicaCount=1 \
  --set minio.mode=standalone \
  --set pulsar.enabled=false \
  --set metrics.serviceMonitor.enabled=false \
  --namespace vllm-semantic-router-system --create-namespace
```

**With Prometheus Operator** (production with monitoring):

```bash
helm install milvus-semantic-cache milvus/milvus \
  --set cluster.enabled=false \
  --set etcd.replicaCount=1 \
  --set minio.mode=standalone \
  --set pulsar.enabled=false \
  --namespace vllm-semantic-router-system --create-namespace
```

### Cluster Mode

Recommended for production with high availability:

```bash
helm repo add milvus https://zilliztech.github.io/milvus-helm/
helm repo update
```

:::note[Pulsar Version]
Milvus 2.4+ uses Pulsar v3 by default. The values below disable the old Pulsar to avoid conflicts.
:::

**Without Prometheus Operator** (for testing):

```bash
helm install milvus-semantic-cache milvus/milvus \
  --set cluster.enabled=true \
  --set etcd.replicaCount=3 \
  --set minio.mode=distributed \
  --set pulsar.enabled=false \
  --set pulsarv3.enabled=true \
  --set metrics.serviceMonitor.enabled=false \
  --namespace vllm-semantic-router-system --create-namespace
```

**With Prometheus Operator** (production with monitoring):

```bash
helm install milvus-semantic-cache milvus/milvus \
  --set cluster.enabled=true \
  --set etcd.replicaCount=3 \
  --set minio.mode=distributed \
  --set pulsar.enabled=false \
  --set pulsarv3.enabled=true \
  --namespace vllm-semantic-router-system --create-namespace
```

## Deploy with Milvus Operator

1. Install Milvus Operator following the [official instructions](https://github.com/zilliztech/milvus-operator)

2. Apply the Custom Resource:

**Standalone:**

```bash
kubectl apply -n vllm-semantic-router-system -f - <<EOF
apiVersion: milvus.io/v1beta1
kind: Milvus
metadata:
  name: milvus-standalone
spec:
  mode: standalone
  components:
    disableMetrics: false
  dependencies:
    storage:
      inCluster:
        values:
          mode: standalone
        deletionPolicy: Delete
        pvcDeletion: true
    etcd:
      inCluster:
        values:
          replicaCount: 1
  config: {}
EOF
```

**Cluster:**

```bash
kubectl apply -n vllm-semantic-router-system -f - <<EOF
apiVersion: milvus.io/v1beta1
kind: Milvus
metadata:
  name: milvus-cluster
spec:
  mode: cluster
  components:
    disableMetrics: false
  dependencies:
    storage:
      inCluster:
        values:
          mode: distributed
        deletionPolicy: Retain
        pvcDeletion: false
    etcd:
      inCluster:
        values:
          replicaCount: 3
    pulsar:
      inCluster:
        values:
          broker:
            replicaCount: 1
  config: {}
EOF
```

## Configure Semantic Router

### Apply Milvus Client Config

```bash
kubectl apply -n vllm-semantic-router-system -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: milvus-client-config
data:
  milvus.yaml: |
    connection:
      host: "milvus-semantic-cache.vllm-semantic-router-system.svc.cluster.local"
      port: 19530
      timeout: 60
      auth:
        enabled: false
      tls:
        enabled: false
    collection:
      name: "semantic_cache"
      description: "Semantic cache"
      vector_field:
        name: "embedding"
        dimension: 384
        metric_type: "IP"
      index:
        type: "HNSW"
        params:
          M: 16
          efConstruction: 64
    search:
      params:
        ef: 64
      topk: 10
      consistency_level: "Session"
    development:
      auto_create_collection: true
      verbose_errors: true
EOF
```

### Update Router Config

Ensure these settings in your router configuration:

```yaml
semantic_cache:
  backend_type: "milvus"
  backend_config_path: "config/semantic-cache/milvus.yaml"
```

## Networking and Security

### Network Policy

Restrict access to Milvus:

```bash
kubectl apply -n vllm-semantic-router-system -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-router-to-milvus
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/name: milvus
  policyTypes:
    - Ingress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: vllm-semantic-router-system
          podSelector:
            matchLabels:
              app.kubernetes.io/name: semantic-router
      ports:
        - protocol: TCP
          port: 19530
EOF
```

### TLS and Authentication

1. Create secrets for credentials and certificates:

```bash
# Auth credentials
kubectl create secret generic milvus-auth -n vllm-semantic-router-system \
  --from-literal=username="YOUR_USERNAME" \
  --from-literal=password="YOUR_PASSWORD"

# TLS certificates
kubectl create secret generic milvus-tls -n vllm-semantic-router-system \
  --from-file=ca.crt=/path/to/ca.crt \
  --from-file=client.crt=/path/to/client.crt \
  --from-file=client.key=/path/to/client.key
```

2. Update Milvus client configuration:

```yaml
connection:
  host: "milvus-cluster.vllm-semantic-router-system.svc.cluster.local"
  port: 19530
  timeout: 60
  auth:
    enabled: true
    username: "${MILVUS_USERNAME}"
    password: "${MILVUS_PASSWORD}"
  tls:
    enabled: true
```

:::tip
Wire environment variables or projected Secret volumes to the router deployment and reference them in the config.
:::

## Storage

Ensure a default `StorageClass` exists. Milvus Helm chart and Operator automatically create necessary PVCs for etcd and MinIO.

## Monitoring

:::note[Requires Prometheus Operator]
ServiceMonitor requires [Prometheus Operator](https://github.com/prometheus-operator/prometheus-operator) to be installed in your cluster. The default Helm values enable ServiceMonitor.
:::

### Install Prometheus Operator

If not already installed:

```bash
kubectl apply -f https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/main/bundle.yaml

# Wait for CRDs to be ready
kubectl wait --for condition=established --timeout=60s \
  crd/servicemonitors.monitoring.coreos.com
```

### Deploy Milvus with Monitoring

ServiceMonitor is enabled by default. Just omit the `--set metrics.serviceMonitor.enabled=false` flag:

```bash
helm install milvus-semantic-cache milvus/milvus \
  --set cluster.enabled=false \
  --set etcd.replicaCount=1 \
  --set minio.mode=standalone \
  --set pulsar.enabled=false \
  --namespace vllm-semantic-router-system --create-namespace
```

### Verify ServiceMonitor

```bash
kubectl get servicemonitor -n vllm-semantic-router-system
```

### Disable Monitoring (Optional)

For testing environments without Prometheus, add `--set metrics.serviceMonitor.enabled=false`:

```bash
helm install milvus-semantic-cache milvus/milvus \
  --set cluster.enabled=false \
  --set etcd.replicaCount=1 \
  --set minio.mode=standalone \
  --set pulsar.enabled=false \
  --set metrics.serviceMonitor.enabled=false \
  --namespace vllm-semantic-router-system --create-namespace
```

## Migration from Memory Cache

### Pre-migration Checklist

- Milvus deployed and healthy: `kubectl get pods -l app.kubernetes.io/name=milvus`
- Network connectivity verified between router and Milvus
- Sufficient storage provisioned for expected cache size

### Staged Rollout

```bash
# Step 1: Deploy Milvus (using Helm for simplicity)
helm install milvus-semantic-cache milvus/milvus \
  --set cluster.enabled=false \
  --set etcd.replicaCount=1 \
  --set minio.mode=standalone \
  --set pulsar.enabled=false \
  --set metrics.serviceMonitor.enabled=false \
  --namespace vllm-semantic-router-system --create-namespace

# Step 2: Wait for ready
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=milvus \
  -n vllm-semantic-router-system --timeout=300s

# Step 3: Update router config (set backend_type: "milvus")
kubectl edit configmap semantic-router-config -n vllm-semantic-router-system

# Step 4: Restart router
kubectl rollout restart deployment/semantic-router -n vllm-semantic-router-system
```

### Validation

```bash
# Check logs for Milvus connection
kubectl logs -l app=semantic-router -n vllm-semantic-router-system | grep -i milvus

# Test cache functionality
curl -X POST http://<router-endpoint>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "test", "messages": [{"role": "user", "content": "Hello"}]}'

# Repeat request to verify cache hit
curl -X POST http://<router-endpoint>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "test", "messages": [{"role": "user", "content": "Hello"}]}'
```

### Monitor Metrics

- Cache hit ratio should stabilize after warm-up
- Latency: Milvus adds ~1-5ms per lookup vs memory cache
- Error rate should remain at baseline

### Rollback

```bash
# Revert to memory backend
kubectl patch configmap semantic-router-config -n vllm-semantic-router-system \
  --type merge -p '{"data":{"config.yaml":"semantic_cache:\n  backend_type: \"memory\""}}'

# Restart router
kubectl rollout restart deployment/semantic-router -n vllm-semantic-router-system

# Verify
kubectl logs -l app=semantic-router -n vllm-semantic-router-system | grep -i "cache"
```

:::note
Data in Milvus is preserved and can be reused when switching back.
:::

## Backup and Recovery

### Backup Strategies

**1. Milvus Native Backup (Recommended)**

Use [milvus-backup](https://github.com/zilliztech/milvus-backup):

```bash
# Install
wget https://github.com/zilliztech/milvus-backup/releases/latest/download/milvus-backup_Linux_x86_64.tar.gz
tar -xzf milvus-backup_Linux_x86_64.tar.gz

# Create backup
./milvus-backup create -n semantic_cache_backup \
  --milvus.address milvus-cluster.vllm-semantic-router-system.svc.cluster.local:19530

# List / Restore
./milvus-backup list
./milvus-backup restore -n semantic_cache_backup
```

**2. Storage-Level Backup**

Use volume snapshots (requires CSI snapshot controller):

```yaml
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshot
metadata:
  name: milvus-data-snapshot
  namespace: vllm-semantic-router-system
spec:
  volumeSnapshotClassName: csi-snapclass
  source:
    persistentVolumeClaimName: milvus-data
```

**3. MinIO/S3 Backup (Cluster Mode)**

Configure bucket versioning and replication:

```bash
mc version enable myminio/milvus-bucket
mc replicate add myminio/milvus-bucket --remote-bucket milvus-bucket-dr \
  --arn "arn:minio:replication::..."
```

### Recovery Procedures

**From milvus-backup:**

```bash
# Stop router
kubectl scale deployment/semantic-router -n vllm-semantic-router-system --replicas=0

# Restore
./milvus-backup restore -n semantic_cache_backup --restore_index

# Restart router
kubectl scale deployment/semantic-router -n vllm-semantic-router-system --replicas=3
```

**From VolumeSnapshot:**

```bash
kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: milvus-data-restored
  namespace: vllm-semantic-router-system
spec:
  dataSource:
    name: milvus-data-snapshot
    kind: VolumeSnapshot
    apiGroup: snapshot.storage.k8s.io
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi
EOF
```

### Backup Schedule Recommendation

| Environment | Frequency | Retention | Method |
|-------------|-----------|-----------|--------|
| Development | Weekly | 2 backups | milvus-backup |
| Staging | Daily | 7 backups | milvus-backup + snapshots |
| Production | Every 6 hours | 14 backups | milvus-backup + S3 replication |

## Troubleshooting

### Both Pulsar and Pulsar v3 Running

**Symptom**: Both `pulsar` and `pulsarv3` pods are running in cluster mode

```bash
kubectl get pods -n vllm-semantic-router-system | grep pulsar
# Shows both milvus-semantic-cache-pulsar-* and milvus-semantic-cache-pulsarv3-* pods
```

**Cause**: Both old Pulsar and Pulsar v3 are enabled in Helm values

**Solution**: Use only Pulsar v3 (recommended for Milvus 2.4+)

```bash
# Uninstall existing release
helm uninstall milvus-semantic-cache -n vllm-semantic-router-system

# Reinstall with correct configuration
helm install milvus-semantic-cache milvus/milvus \
  --set cluster.enabled=true \
  --set etcd.replicaCount=3 \
  --set minio.mode=distributed \
  --set pulsar.enabled=false \
  --set pulsarv3.enabled=true \
  --set metrics.serviceMonitor.enabled=false \
  --namespace vllm-semantic-router-system --create-namespace
```

### ServiceMonitor CRD Not Found

**Symptom**: Helm installation fails with error:

```
Error: INSTALLATION FAILED: unable to build kubernetes objects from release manifest: 
resource mapping not found for name: "milvus-semantic-cache-milvus-standalone" namespace: "vllm-semantic-router-system" 
from "": no matches for kind "ServiceMonitor" in version "monitoring.coreos.com/v1"
ensure CRDs are installed first
```

**Solution**: Disable ServiceMonitor or install Prometheus Operator

```bash
# Option 1: Disable ServiceMonitor (recommended for testing)
helm install milvus-semantic-cache milvus/milvus \
  --set cluster.enabled=false \
  --set etcd.replicaCount=1 \
  --set minio.mode=standalone \
  --set pulsar.enabled=false \
  --set metrics.serviceMonitor.enabled=false \
  --namespace vllm-semantic-router-system --create-namespace

# Option 2: Install Prometheus Operator first
kubectl apply -f https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/main/bundle.yaml
```

### Connection Issues

**Symptom**: `failed to connect to Milvus: context deadline exceeded`

```bash
# Verify Milvus is running
kubectl get pods -l app.kubernetes.io/name=milvus -n vllm-semantic-router-system

# Check service endpoint
kubectl get svc -l app.kubernetes.io/name=milvus -n vllm-semantic-router-system

# Test connectivity from router pod
kubectl exec -it deploy/semantic-router -n vllm-semantic-router-system -- \
  nc -zv milvus-cluster.vllm-semantic-router-system.svc.cluster.local 19530

# Check NetworkPolicy
kubectl get networkpolicy -n vllm-semantic-router-system

# Verify DNS
kubectl exec -it deploy/semantic-router -n vllm-semantic-router-system -- \
  nslookup milvus-cluster.vllm-semantic-router-system.svc.cluster.local
```

### Authentication Failures

**Symptom**: `authentication failed` or `access denied`

```bash
# Verify credentials
kubectl get secret milvus-auth -n vllm-semantic-router-system -o jsonpath='{.data.username}' | base64 -d

# Check auth in Milvus logs
kubectl logs -l app.kubernetes.io/component=proxy -n vllm-semantic-router-system | grep -i auth

# Verify router config
kubectl get configmap milvus-client-config -n vllm-semantic-router-system -o yaml
```

### Performance Issues

**Symptom**: High latency or timeouts

```bash
# Check resource usage
kubectl top pods -l app.kubernetes.io/name=milvus -n vllm-semantic-router-system

# Review metrics
kubectl port-forward svc/milvus-cluster 9091:9091 -n vllm-semantic-router-system
# Visit http://localhost:9091/metrics
```

Check collection stats via pymilvus:

```python
from pymilvus import connections, Collection
connections.connect(host="localhost", port="19530")
col = Collection("semantic_cache")
print(col.num_entities)
print(col.index())
```

### Collection Issues

**Symptom**: `collection not found` or schema mismatch

```bash
# List collections
kubectl exec -it deploy/milvus-cluster-proxy -n vllm-semantic-router-system -- \
  curl -s localhost:9091/api/v1/collections

# Check auto_create setting
kubectl get configmap milvus-client-config -n vllm-semantic-router-system -o yaml | grep auto_create
```

Manual collection creation:

```python
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

connections.connect(host="localhost", port="19530")
fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
    FieldSchema(name="response", dtype=DataType.VARCHAR, max_length=65535),
]
schema = CollectionSchema(fields, description="Semantic cache")
collection = Collection("semantic_cache", schema)
collection.create_index("embedding", {
    "index_type": "HNSW",
    "metric_type": "IP",
    "params": {"M": 16, "efConstruction": 64}
})
```

### Storage Issues

**Symptom**: PVC pending or storage full

```bash
# Check PVC status
kubectl get pvc -n vllm-semantic-router-system

# Check StorageClass
kubectl get sc

# Check available storage
kubectl exec -it deploy/milvus-cluster-datanode -n vllm-semantic-router-system -- df -h

# Expand PVC
kubectl patch pvc milvus-data -n vllm-semantic-router-system \
  -p '{"spec":{"resources":{"requests":{"storage":"50Gi"}}}}'
```

### Pod Crash / OOM

**Symptom**: CrashLoopBackOff or OOMKilled

```bash
# Check events
kubectl describe pod -l app.kubernetes.io/name=milvus -n vllm-semantic-router-system

# Check previous logs
kubectl logs -l app.kubernetes.io/name=milvus -n vllm-semantic-router-system --previous

# Increase memory
kubectl patch deployment milvus-cluster-proxy -n vllm-semantic-router-system \
  --type='json' -p='[{"op": "replace", "path": "/spec/template/spec/containers/0/resources/limits/memory", "value":"4Gi"}]'
```

### Diagnostic Commands

```bash
# Overall health
kubectl get all -l app.kubernetes.io/name=milvus -n vllm-semantic-router-system

# Component logs
kubectl logs -l app.kubernetes.io/component=proxy -n vllm-semantic-router-system --tail=100
kubectl logs -l app.kubernetes.io/component=datanode -n vllm-semantic-router-system --tail=100
kubectl logs -l app.kubernetes.io/component=querynode -n vllm-semantic-router-system --tail=100

# etcd health (cluster mode)
kubectl exec -it milvus-cluster-etcd-0 -n vllm-semantic-router-system -- etcdctl endpoint health

# MinIO health (cluster mode)
kubectl exec -it milvus-cluster-minio-0 -n vllm-semantic-router-system -- mc admin info local
```

## Next Steps

- [Configuration Guide](configuration.md) - Semantic cache configuration options
- [Observability](../tutorials/observability/metrics.md) - Monitoring and metrics

## References

- [Configure Milvus with Helm](https://milvus.io/docs/configure-helm.md)
- [Milvus Operator](https://github.com/zilliztech/milvus-operator)
- [Milvus Monitoring](https://milvus.io/docs/monitor.md)

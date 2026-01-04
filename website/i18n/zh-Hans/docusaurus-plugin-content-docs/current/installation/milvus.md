---
translation:
  source_commit: "bac2743"
  source_file: "docs/installation/milvus.md"
  outdated: false
sidebar_position: 5
---

# Milvus Semantic Cache

本指南涵盖在 Kubernetes 中部署 Milvus 作为 Semantic Router 的 Semantic Cache 后端。与默认的 memory cache 相比，Milvus 提供了持久化、可扩展的向量存储。

:::note
Milvus 是可选的。Router 开箱即用，默认使用 memory backend。当您需要持久化、水平扩展或在 Router 副本之间共享缓存时，请使用 Milvus。
:::

## 部署选项

有两种方法可用：

- **Helm**: 快速启动和参数化部署
- **Milvus Operator**: 生产级生命周期管理、滚动升级、健康检查和依赖编排

## 前提条件

- 配置了 `kubectl` 的 Kubernetes 集群
- 可用的默认 `StorageClass`
- 安装了 Helm 3.x

:::note[ServiceMonitor 要求]
默认 Helm 值启用了 ServiceMonitor 以进行 Prometheus 指标收集，这需要先安装 [Prometheus Operator](https://github.com/prometheus-operator/prometheus-operator)。

**对于没有 Prometheus Operator 的测试**，请使用 `--set metrics.serviceMonitor.enabled=false` 禁用 ServiceMonitor（参见下面的部署命令）。
:::

## 使用 Helm 部署

### 单机模式 (Standalone Mode)

适用于开发和小规模部署：

```bash
helm repo add milvus https://zilliztech.github.io/milvus-helm/
helm repo update
```

**无 Prometheus Operator** (用于测试/开发)：

```bash
helm install milvus-semantic-cache milvus/milvus \
  --set cluster.enabled=false \
  --set etcd.replicaCount=1 \
  --set minio.mode=standalone \
  --set pulsar.enabled=false \
  --set metrics.serviceMonitor.enabled=false \
  --namespace vllm-semantic-router-system --create-namespace
```

**有 Prometheus Operator** (用于带监控的生产环境)：

```bash
helm install milvus-semantic-cache milvus/milvus \
  --set cluster.enabled=false \
  --set etcd.replicaCount=1 \
  --set minio.mode=standalone \
  --set pulsar.enabled=false \
  --namespace vllm-semantic-router-system --create-namespace
```

### 集群模式 (Cluster Mode)

推荐用于具有高可用性的生产环境：

```bash
helm repo add milvus https://zilliztech.github.io/milvus-helm/
helm repo update
```

:::note[Pulsar 版本]
Milvus 2.4+ 默认使用 Pulsar v3。下面的值禁用了旧版 Pulsar 以避免冲突。
:::

**无 Prometheus Operator** (用于测试)：

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

**有 Prometheus Operator** (用于带监控的生产环境)：

```bash
helm install milvus-semantic-cache milvus/milvus \
  --set cluster.enabled=true \
  --set etcd.replicaCount=3 \
  --set minio.mode=distributed \
  --set pulsar.enabled=false \
  --set pulsarv3.enabled=true \
  --namespace vllm-semantic-router-system --create-namespace
```

## 使用 Milvus Operator 部署

1. 按照 [官方说明](https://github.com/zilliztech/milvus-operator) 安装 Milvus Operator

2. 应用自定义资源 (Custom Resource)：

**单机 (Standalone):**

```bash
kubectl apply -n vllm-semantic-router-system -f - <<EOF
apiversion: milvus.io/v1beta1
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

**集群 (Cluster):**

```bash
kubectl apply -n vllm-semantic-router-system -f - <<EOF
apiversion: milvus.io/v1beta1
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

## 配置 Semantic Router

### 应用 Milvus 客户端配置

```bash
kubectl apply -n vllm-semantic-router-system -f - <<EOF
apiversion: v1
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

### 更新路由配置

确保您的路由配置中有这些设置：

```yaml
semantic_cache:
  backend_type: "milvus"
  backend_config_path: "config/semantic-cache/milvus.yaml"
```

## 网络和安全

### 网络策略

限制对 Milvus 的访问：

```bash
kubectl apply -n vllm-semantic-router-system -f - <<EOF
apiversion: networking.k8s.io/v1
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

### TLS 和身份验证

1. 为凭据和证书创建 secret：

```bash
# 身份验证凭据
kubectl create secret generic milvus-auth -n vllm-semantic-router-system \
  --from-literal=username="YOUR_USERNAME" \
  --from-literal=password="YOUR_PASSWORD"

# TLS 证书
kubectl create secret generic milvus-tls -n vllm-semantic-router-system \
  --from-file=ca.crt=/path/to/ca.crt \
  --from-file=client.crt=/path/to/client.crt \
  --from-file=client.key=/path/to/client.key
```

1. 更新 Milvus 客户端配置：

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
将环境变量或投影的 Secret 卷连接到路由部署，并在配置中引用它们。
:::

## 存储

确保存在默认的 `StorageClass`。Milvus Helm Chart 和 Operator 会自动为 etcd 和 MinIO 创建必要的 PVC。

## 监控

:::note[需要 Prometheus Operator]
ServiceMonitor 需要在您的集群中安装 [Prometheus Operator](https://github.com/prometheus-operator/prometheus-operator)。默认 Helm 值已启用 ServiceMonitor。
:::

### 安装 Prometheus Operator

如果尚未安装：

```bash
kubectl apply -f https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/main/bundle.yaml

# 等待 CRD 准备就绪
kubectl wait --for condition=established --timeout=60s \
  crd/servicemonitors.monitoring.coreos.com
```

### 部署带有监控的 Milvus

ServiceMonitor 默认启用。只需省略 `--set metrics.serviceMonitor.enabled=false` 标志：

```bash
helm install milvus-semantic-cache milvus/milvus \
  --set cluster.enabled=false \
  --set etcd.replicaCount=1 \
  --set minio.mode=standalone \
  --set pulsar.enabled=false \
  --namespace vllm-semantic-router-system --create-namespace
```

### 验证 ServiceMonitor

```bash
kubectl get servicemonitor -n vllm-semantic-router-system
```

### 禁用监控 (可选)

对于没有 Prometheus 的测试环境，添加 `--set metrics.serviceMonitor.enabled=false`：

```bash
helm install milvus-semantic-cache milvus/milvus \
  --set cluster.enabled=false \
  --set etcd.replicaCount=1 \
  --set minio.mode=standalone \
  --set pulsar.enabled=false \
  --set metrics.serviceMonitor.enabled=false \
  --namespace vllm-semantic-router-system --create-namespace
```

## 从内存缓存迁移

### 迁移前检查清单

- Milvus 已部署且健康：`kubectl get pods -l app.kubernetes.io/name=milvus`
- 验证了路由和 Milvus 之间的网络连接
- 为预期缓存大小预配置了足够的存储空间

### 分阶段部署

```bash
# 步骤 1: 部署 Milvus (为简单起见使用 Helm)
helm install milvus-semantic-cache milvus/milvus \
  --set cluster.enabled=false \
  --set etcd.replicaCount=1 \
  --set minio.mode=standalone \
  --set pulsar.enabled=false \
  --set metrics.serviceMonitor.enabled=false \
  --namespace vllm-semantic-router-system --create-namespace

# 步骤 2: 等待就绪
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=milvus \
  -n vllm-semantic-router-system --timeout=300s

# 步骤 3: 更新路由配置 (设置 backend_type: "milvus")
kubectl edit configmap semantic-router-config -n vllm-semantic-router-system

# 步骤 4: 重启路由
kubectl rollout restart deployment/semantic-router -n vllm-semantic-router-system
```

### 验证

```bash
# 检查日志以获取 Milvus 连接
kubectl logs -l app=semantic-router -n vllm-semantic-router-system | grep -i milvus

# 测试缓存功能
curl -X POST http://<router-endpoint>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "test", "messages": [{"role": "user", "content": "Hello"}]}'

# 重复请求以验证缓存命中
curl -X POST http://<router-endpoint>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "test", "messages": [{"role": "user", "content": "Hello"}]}'
```

### 监控指标

- 缓存命中率应在预热后稳定下来
- 延迟：Milvus 每次查找比内存缓存增加约 1-5ms
- 错误率应保持在基线水平

### 回滚

```bash
# 恢复到内存后端
kubectl patch configmap semantic-router-config -n vllm-semantic-router-system \
  --type merge -p '{"data":{"config.yaml":"semantic_cache:\n  backend_type: \"memory\""}}'

# 重启路由
kubectl rollout restart deployment/semantic-router -n vllm-semantic-router-system

# 验证
kubectl logs -l app=semantic-router -n vllm-semantic-router-system | grep -i "cache"
```

:::note
Milvus 中的数据会被保留，切换回来时可以重用。
:::

## 备份和恢复

### 备份策略

**1. Milvus 原生备份 (推荐)**

使用 [milvus-backup](https://github.com/zilliztech/milvus-backup)：

```bash
# 安装
wget https://github.com/zilliztech/milvus-backup/releases/latest/download/milvus-backup_Linux_x86_64.tar.gz
tar -xzf milvus-backup_Linux_x86_64.tar.gz

# 创建备份
./milvus-backup create -n semantic_cache_backup \
  --milvus.address milvus-cluster.vllm-semantic-router-system.svc.cluster.local:19530

# 列出 / 恢复
./milvus-backup list
./milvus-backup restore -n semantic_cache_backup
```

**2. 存储级备份**

使用卷快照（需要 CSI 快照控制器）：

```yaml
apiversion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshot
metadata:
  name: milvus-data-snapshot
  namespace: vllm-semantic-router-system
spec:
  volumeSnapshotClassName: csi-snapclass
  source:
    persistentVolumeClaimName: milvus-data
```

**3. MinIO/S3 备份 (集群模式)**

配置存储桶版本控制和复制：

```bash
mc version enable myminio/milvus-bucket
mc replicate add myminio/milvus-bucket --remote-bucket milvus-bucket-dr \
  --arn "arn:minio:replication::..."
```

### 恢复程序

**从 milvus-backup 恢复：**

```bash
# 停止路由
kubectl scale deployment/semantic-router -n vllm-semantic-router-system --replicas=0

# 恢复
./milvus-backup restore -n semantic_cache_backup --restore_index

# 重启路由
kubectl scale deployment/semantic-router -n vllm-semantic-router-system --replicas=3
```

**从卷快照恢复：**

```bash
kubectl apply -f - <<EOF
apiversion: v1
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

### 备份计划建议

| 环境 | 频率 | 保留 | 方法 |
|-------------|-----------|-----------|--------|
| 开发 | 每周 | 2 个备份 | milvus-backup |
| 预发布 (Staging) | 每天 | 7 个备份 | milvus-backup + 快照 |
| 生产 | 每 6 小时 | 14 个备份 | milvus-backup + S3 复制 |

## 故障排除

### Pulsar 和 Pulsar v3 同时运行

**症状**: 在集群模式下，`pulsar` 和 `pulsarv3` pod 都在运行

```bash
kubectl get pods -n vllm-semantic-router-system | grep pulsar
# 显示 milvus-semantic-cache-pulsar-* 和 milvus-semantic-cache-pulsarv3-* pod
```

**原因**: Helm 值中同时启用了旧版 Pulsar 和 Pulsar v3

**解决方案**: 仅使用 Pulsar v3（推荐用于 Milvus 2.4+）

```bash
# 卸载现有发布
helm uninstall milvus-semantic-cache -n vllm-semantic-router-system

# 使用正确配置重新安装
helm install milvus-semantic-cache milvus/milvus \
  --set cluster.enabled=true \
  --set etcd.replicaCount=3 \
  --set minio.mode=distributed \
  --set pulsar.enabled=false \
  --set pulsarv3.enabled=true \
  --set metrics.serviceMonitor.enabled=false \
  --namespace vllm-semantic-router-system --create-namespace
```

### ServiceMonitor CRD 未找到

**症状**: Helm 安装失败，并显示错误：

```
Error: INSTALLATION FAILED: unable to build kubernetes objects from release manifest: 
resource mapping not found for name: "milvus-semantic-cache-milvus-standalone" namespace: "vllm-semantic-router-system" 
from "": no matches for kind "ServiceMonitor" in version "monitoring.coreos.com/v1"
ensure CRDs are installed first
```

**解决方案**: 禁用 ServiceMonitor 或安装 Prometheus Operator

```bash
# 选项 1: 禁用 ServiceMonitor (推荐用于测试)
helm install milvus-semantic-cache milvus/milvus \
  --set cluster.enabled=false \
  --set etcd.replicaCount=1 \
  --set minio.mode=standalone \
  --set pulsar.enabled=false \
  --set metrics.serviceMonitor.enabled=false \
  --namespace vllm-semantic-router-system --create-namespace

# 选项 2: 先安装 Prometheus Operator
kubectl apply -f https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/main/bundle.yaml
```

### 连接问题

**症状**: `failed to connect to Milvus: context deadline exceeded`

```bash
# 验证 Milvus 正在运行
kubectl get pods -l app.kubernetes.io/name=milvus -n vllm-semantic-router-system

# 检查服务端点
kubectl get svc -l app.kubernetes.io/name=milvus -n vllm-semantic-router-system

# 从 router pod 测试连接
kubectl exec -it deploy/semantic-router -n vllm-semantic-router-system -- \
  nc -zv milvus-cluster.vllm-semantic-router-system.svc.cluster.local 19530

# 检查 NetworkPolicy
kubectl get networkpolicy -n vllm-semantic-router-system

# 验证 DNS
kubectl exec -it deploy/semantic-router -n vllm-semantic-router-system -- \
  nslookup milvus-cluster.vllm-semantic-router-system.svc.cluster.local
```

### 身份验证失败

**症状**: `authentication failed` 或 `access denied`

```bash
# 验证凭据
kubectl get secret milvus-auth -n vllm-semantic-router-system -o jsonpath='{.data.username}' | base64 -d

# 在 Milvus 日志中检查 auth
kubectl logs -l app.kubernetes.io/component=proxy -n vllm-semantic-router-system | grep -i auth

# 验证路由配置
kubectl get configmap milvus-client-config -n vllm-semantic-router-system -o yaml
```

### 性能问题

**症状**: 高延迟或超时

```bash
# 检查资源使用情况
kubectl top pods -l app.kubernetes.io/name=milvus -n vllm-semantic-router-system

# 查看指标
kubectl port-forward svc/milvus-cluster 9091:9091 -n vllm-semantic-router-system
# 访问 http://localhost:9091/metrics
```

通过 pymilvus 检查集合统计信息：

```python
from pymilvus import connections, Collection
connections.connect(host="localhost", port="19530")
col = Collection("semantic_cache")
print(col.num_entities)
print(col.index())
```

### 集合问题

**症状**: `collection not found` 或模式不匹配

```bash
# 列出集合
kubectl exec -it deploy/milvus-cluster-proxy -n vllm-semantic-router-system -- \
  curl -s localhost:9091/api/v1/collections

# 检查 auto_create 设置
kubectl get configmap milvus-client-config -n vllm-semantic-router-system -o yaml | grep auto_create
```

手动创建集合：

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

### 存储问题

**症状**: PVC pending 或存储已满

```bash
# 检查 PVC 状态
kubectl get pvc -n vllm-semantic-router-system

# 检查 StorageClass
kubectl get sc

# 检查可用存储
kubectl exec -it deploy/milvus-cluster-datanode -n vllm-semantic-router-system -- df -h

# 扩容 PVC
kubectl patch pvc milvus-data -n vllm-semantic-router-system \
  -p '{"spec":{"resources":{"requests":{"storage":"50Gi"}}}}'
```

### Pod 崩溃 / OOM

**症状**: CrashLoopBackOff 或 OOMKilled

```bash
# 检查事件
kubectl describe pod -l app.kubernetes.io/name=milvus -n vllm-semantic-router-system

# 检查之前的日志
kubectl logs -l app.kubernetes.io/name=milvus -n vllm-semantic-router-system --previous

# 增加内存
kubectl patch deployment milvus-cluster-proxy -n vllm-semantic-router-system \
  --type='json' -p='[{"op": "replace", "path": "/spec/template/spec/containers/0/resources/limits/memory", "value":"4Gi"}]'
```

### 诊断命令

```bash
# 整体健康状况
kubectl get all -l app.kubernetes.io/name=milvus -n vllm-semantic-router-system

# 组件日志
kubectl logs -l app.kubernetes.io/component=proxy -n vllm-semantic-router-system --tail=100
kubectl logs -l app.kubernetes.io/component=datanode -n vllm-semantic-router-system --tail=100
kubectl logs -l app.kubernetes.io/component=querynode -n vllm-semantic-router-system --tail=100

# etcd 健康状况 (集群模式)
kubectl exec -it milvus-cluster-etcd-0 -n vllm-semantic-router-system -- etcdctl endpoint health

# MinIO 健康状况 (集群模式)
kubectl exec -it milvus-cluster-minio-0 -n vllm-semantic-router-system -- mc admin info local
```

## 下一步

- [配置指南](configuration.md) - 语义缓存配置选项
- [可观测性](../tutorials/observability/metrics.md) - 监控和指标

## 参考

- [使用 Helm 配置 Milvus](https://milvus.io/docs/configure-helm.md)
- [Milvus Operator](https://github.com/zilliztech/milvus-operator)
- [Milvus 监控](https://milvus.io/docs/monitor.md)

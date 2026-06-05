---
sidebar_position: 6
sidebar_label: Valkey 代理记忆
---

# Valkey Agentic Memory

本指南介绍如何将 Valkey 部署为 Semantic Router 的 agentic memory 后端。Valkey 是一个轻量、与 Redis 兼容的替代方案：它通过内置 Search 模块提供向量相似度存储能力，可用于替代 Milvus 的部分场景。

::::note
Valkey 是可选项。默认的 memory 后端是 Milvus。如果你希望单二进制部署、避免依赖 etcd/MinIO 之类外部组件，或你已在使用 Valkey 做缓存，那么可以选择 Valkey。
::::

## 何时用 Valkey，何时用 Milvus

| 关注点 | Valkey | Milvus |
|---------|--------|--------|
| 部署复杂度 | 单二进制 + Search 模块 | 依赖 etcd、MinIO/S3，可选 Pulsar |
| 水平扩展 | Cluster 模式（手工分片） | 原生分布式架构 |
| 存储模型 | 内存为主，可选持久化 | 磁盘为主，索引可 memory-mapped |
| 更适合 | 小到中等负载、dev/test、已有 Redis/Valkey 基础设施 | 大规模生产、十亿级向量 |
| 向量索引 | 通过 FT.CREATE 的 HNSW | HNSW、IVF_FLAT、IVF_SQ8 等 |

## 前置条件

- Valkey 8.0+，并且 **启用 Search 模块**
- Search 模块在 **1.2.0** 版本引入了用于向量搜索的 text 支持
- `valkey/valkey-bundle` 镜像默认包含 Search；Search 1.2.0 可在 valkey-bundle 的 `unstable` 与 `9.1.0-rc1` 版本中使用
- 若你的 Valkey 不包含 Search 模块，可参考 [手动添加](https://github.com/valkey-io/valkey-search/blob/main/README.md)
- Kubernetes 场景：Helm 3.x 与已配置的 `kubectl`

::::info Search 模块有问题？
如果你在加载或使用 Search 模块时遇到问题，请 [提交 issue](https://github.com/vllm-project/semantic-router/issues/new)，我们会协助排查。
::::

## 使用 Docker 部署

### 快速开始

```bash
docker run -d --name valkey-memory \
  -p 6379:6379 \
  valkey/valkey-bundle:latest
```

验证 Search 模块已加载：

```bash
docker exec valkey-memory valkey-cli MODULE LIST | grep search
```

### 启用持久化

```bash
docker run -d --name valkey-memory \
  -p 6379:6379 \
  -v valkey-data:/data \
  valkey/valkey-bundle:latest \
  valkey-server --appendonly yes
```

## 在 Kubernetes 中部署

### 使用 StatefulSet

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: valkey-memory
  namespace: vllm-semantic-router-system
spec:
  serviceName: valkey-memory
  replicas: 1
  selector:
    matchLabels:
      app: valkey-memory
  template:
    metadata:
      labels:
        app: valkey-memory
    spec:
      containers:
        - name: valkey
          image: valkey/valkey-bundle:latest
          ports:
            - containerPort: 6379
          args: ["valkey-server", "--appendonly", "yes"]
          # For production, add --requirepass or mount a Secret:
          # args: ["valkey-server", "--appendonly", "yes", "--requirepass", "$(VALKEY_PASSWORD)"]
          volumeMounts:
            - name: data
              mountPath: /data
          resources:
            requests:
              memory: "256Mi"
              cpu: "250m"
            limits:
              memory: "1Gi"
              cpu: "1000m"
  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 5Gi
---
apiVersion: v1
kind: Service
metadata:
  name: valkey-memory
  namespace: vllm-semantic-router-system
spec:
  selector:
    app: valkey-memory
  ports:
    - port: 6379
      targetPort: 6379
  clusterIP: None
```

## 配置 Router

在你的 `config.yaml` 中添加 Valkey memory 后端配置：

```yaml
global:
  stores:
    memory:
      enabled: true
      backend: valkey
      auto_store: true
      valkey:
        host: valkey-memory          # Service name or hostname
        port: 6379
        database: 0
        timeout: 10
        collection_prefix: "mem:"
        index_name: mem_idx
        dimension: 384               # Must match your embedding model
        metric_type: COSINE           # COSINE, L2, or IP
        index_m: 16
        index_ef_construction: 256
      embedding_model: bert
      default_retrieval_limit: 5
      default_similarity_threshold: 0.70
      hybrid_search: true
      hybrid_mode: rerank
      adaptive_threshold: true
```

### 配置参考

| 参数 | 默认值 | 说明 |
|-----------|---------|-------------|
| `host` | `localhost` | Valkey 服务地址 |
| `port` | `6379` | 端口 |
| `database` | `0` | 数据库编号（0-15） |
| `password` | _(empty)_ | 认证密码 |
| `timeout` | `10` | 连接超时（秒） |
| `collection_prefix` | `mem:` | HASH 文档 key 前缀 |
| `index_name` | `mem_idx` | FT.CREATE 索引名 |
| `dimension` | `384` | embedding 向量维度 |
| `metric_type` | `COSINE` | 距离度量：`COSINE`、`L2`、`IP` |
| `index_m` | `16` | HNSW 的 M 参数（每个节点链接数） |
| `index_ef_construction` | `256` | HNSW 构建时搜索宽度 |

### 可选：Redis Hot Cache

你可以在 Valkey memory store 前再叠一层 Redis/Valkey hot cache，用于加速高频访问的 memories：

```yaml
      redis_cache:
        enabled: true
        address: "valkey-memory:6379"
        ttl_seconds: 900
        db: 1                        # Use a different DB to avoid key collisions
        key_prefix: "memory_cache:"
```

## 按 decision 覆盖 memory 插件（Per-Decision Memory Plugin）

路由可以通过 `memory` plugin 覆盖全局 memory 设置：

```yaml
routing:
  decisions:
    - name: personalized_route
      plugins:
        - type: memory
          configuration:
            enabled: true
            retrieval_limit: 10
            similarity_threshold: 0.60
            auto_store: true
```

更多细节请参考 [Memory plugin tutorial](/docs/tutorials/plugin/memory)。

## 性能调优

### HNSW 索引参数

- **`index_m`**（默认 16）：增大可提升召回率，但会增加内存占用。对需要高精度的生产负载可使用 32-64。
- **`index_ef_construction`**（默认 256）：增大可提升索引质量，但会让构建更慢。生产建议 512+。

### 内存估算

每条 memory 大致占用：

- HASH 字段：约 500-2000 字节（内容、元数据、时间戳等）
- embedding 向量：`dimension * 4` 字节（例如 384 * 4 ≈ 1.5KB）
- HNSW 索引开销：约 `dimension * index_m * 4` 字节/条

以 10 万条 memories、384 维、M=16 为例：

- 数据：约 300MB
- 索引：约 240MB
- **合计：约 540MB**（不含 Valkey 本身基础开销）

### 持久化

启用 AOF（Append-Only File）提高持久性：

```bash
valkey-server --appendonly yes --appendfsync everysec
```

RDB 快照（时间点备份）：

```bash
valkey-server --save 900 1 --save 300 10
```

## 故障排查

### Search 模块未加载

```
FT.CREATE failed: unknown command 'FT.CREATE'
```

请确认你使用的是 `valkey/valkey-bundle`（包含 Search），而不是 `valkey/valkey`：

```bash
valkey-cli MODULE LIST
# Should show: name search ver ...
```

### 连接超时

```
valkey: connection timeout
```

- 确认 hostname 可解析：`nslookup valkey-memory`
- 检查端口连通性：`nc -zv valkey-memory 6379`
- 若网络较慢，可适当增大 `timeout`

### 索引已存在

Router 启动时会检查索引是否存在，若存在则跳过创建。如果你需要重建索引（例如修改了 `dimension` 或 `metric_type`）：

```bash
valkey-cli FT.DROPINDEX mem_idx
```

索引会在下一次请求时自动重建。

### 内存不足（Out of Memory）

Valkey 将全部数据存放在内存中。如果触达内存上限：

1. 在 Valkey 配置中设置 `maxmemory` 与 `maxmemory-policy`
2. 使用 `quality_scoring.max_memories_per_user` 限制每个用户存储上限
3. 启用 memory consolidation 合并相似 memories

## 从 Milvus 迁移

将已有部署从 Milvus 切换到 Valkey：

1. 更新 `config.yaml`：将 `backend: valkey` 并添加 `valkey:` 配置块
2. 删除或注释 `milvus:` 配置块
3. 重启 router —— 它会自动创建 Valkey 索引
4. Milvus 中已有 memories **不会** 自动迁移

::::warning
切换后端不会迁移数据。若需要保留已有 memories，请先从 Milvus 导出，再通过 memory API 导入后再切换。
::::


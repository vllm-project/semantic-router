---
sidebar_position: 6
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/installation/valkey-memory.md"
  outdated: false
---

# Valkey 智能体记忆

本指南介绍将 Valkey 部署为 Semantic Router 的智能体记忆后端。Valkey 提供轻量、兼容 Redis 的替代方案，通过内置 Search 模块进行向量相似度存储。

:::note
Valkey 是可选的。默认记忆后端是 Milvus。当你希望单二进制部署、没有 etcd 或 MinIO 等外部依赖，或已经为缓存运行 Valkey 时，使用 Valkey。
:::

## 何时使用 Valkey 与 Milvus

| 关注点 | Valkey | Milvus |
|---------|--------|--------|
| 部署复杂度 | 带 Search 模块的单二进制 | 需要 etcd、MinIO/S3，可选 Pulsar |
| 水平扩展 | 集群模式（手动分片） | 原生分布式架构 |
| 内存模型 | 内存存储，可选持久化 | 基于磁盘，带内存映射索引 |
| 最适合 | 中小型工作负载、开发/测试、现有 Redis/Valkey 基础设施 | 较大或分布式向量工作负载 |
| 向量索引 | 通过 FT.CREATE 的 HNSW | HNSW、IVF_FLAT、IVF_SQ8 等 |

## 前置条件

- 启用 Search 模块的 Valkey 发行版。`valkey/valkey-bundle` 镜像包含该模块。
- 你部署的发行版所支持的 Valkey 与 Search 模块版本对。在生产中固定该发行版，而不是跟随 `latest` 或 RC 标签。
- 如果你的 Valkey 发行版未捆绑 Search，请遵循上游 [Valkey Search 快速开始](https://github.com/valkey-io/valkey-search/blob/main/QUICK_START.md)，并在加载模块之前复核其发行说明。
- 对于 Kubernetes：Helm 3.x 和已配置的 `kubectl`

:::info Search 模块遇到问题？
如果在加载或使用 Search 模块时遇到问题，请[开一个 issue](https://github.com/vllm-project/semantic-router/issues/new)，以便我们提供帮助。
:::

## 使用 Docker 部署

### 快速开始

```bash
docker network inspect vllm-sr-network >/dev/null 2>&1 || \
  docker network create vllm-sr-network

docker run -d --name valkey-memory \
  --network vllm-sr-network \
  valkey/valkey-bundle:latest
```

验证 Search 模块已加载：

```bash
docker exec valkey-memory valkey-cli MODULE LIST | grep search
```

### 带持久化

```bash
docker run -d --name valkey-memory \
  --network vllm-sr-network \
  -v valkey-data:/data \
  valkey/valkey-bundle:latest \
  valkey-server --appendonly yes
```

下面配置中的主机名 `valkey-memory` 是 `vllm-sr-network` 上的 Docker DNS。如果用自定义栈名称启动 Router，例如 `VLLM_SR_STACK_NAME=team-a vllm-sr serve`，请将 Valkey 附加到 `team-a-vllm-sr-network`，并使用匹配的可到达主机名。

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

该清单是未认证的开发示例。对于生产，使用你的 Valkey operator 或 chart 的 Secret 集成和网络策略，而不是将密码放在 Pod 命令行中。通过密钥管理工作流，在 Router 的 `global.stores.memory.valkey.password` 中设置相同凭据。

## 配置 Router

将 Valkey 记忆后端添加到你的 `config.yaml`：

```yaml
global:
  stores:
    memory:
      enabled: true
      backend: valkey
      auto_store: true
      valkey:
        host: valkey-memory          # 服务名或主机名
        port: 6379
        database: 0
        timeout: 10
        collection_prefix: "mem:"
        index_name: mem_idx
        dimension: 384               # 必须与嵌入模型匹配
        metric_type: COSINE           # COSINE、L2 或 IP
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

| 参数 | 默认值 | 描述 |
|-----------|---------|-------------|
| `host` | `localhost` | Valkey 服务器主机名 |
| `port` | `6379` | Valkey 服务器端口 |
| `database` | `0` | 数据库编号（0-15） |
| `password` | _(空)_ | 身份验证密码 |
| `timeout` | `10` | 连接超时（秒） |
| `collection_prefix` | `mem:` | HASH 文档的键前缀 |
| `index_name` | `mem_idx` | FT.CREATE 索引名称 |
| `dimension` | 派生 | 嵌入向量维度；省略时，`mmbert` 使用 256，当前其他记忆嵌入模型使用 384 |
| `metric_type` | `COSINE` | 距离度量：`COSINE`、`L2` 或 `IP` |
| `index_m` | `16` | HNSW M 参数（每个节点的链接数） |
| `index_ef_construction` | `256` | HNSW 构建时搜索宽度 |
| `tls_enabled` | `false` | 使用 TLS 连接 Valkey |
| `tls_ca_path` | _(空)_ | 挂载到 Router 中的 PEM 编码 CA 文件；空值使用系统信任存储 |
| `tls_insecure_skip_verify` | `false` | 跳过证书验证；在隔离开发之外保持 `false` |

对于生产 TLS 端点，将 CA 证书挂载到 Router，并将密码保留在由环境提供的密钥中：

```yaml
global:
  stores:
    memory:
      enabled: true
      backend: valkey
      valkey:
        host: valkey.example.internal
        port: 6380
        password: ${VALKEY_PASSWORD}
        tls_enabled: true
        tls_ca_path: /etc/valkey/certs/ca.pem
        tls_insecure_skip_verify: false
```

### 可选的 Redis 热缓存

你可以在 Valkey 记忆存储前面叠加 Redis/Valkey 热缓存，用于频繁访问的记忆：

```yaml
      redis_cache:
        enabled: true
        address: "valkey-memory:6379"
        ttl_seconds: 900
        db: 1                        # 使用不同的 DB 以避免键冲突
        key_prefix: "memory_cache:"
```

## 按决策的记忆插件

路由可以使用 `memory` 插件覆盖全局记忆设置：

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

详情见 [记忆插件教程](/zh-Hans/docs/tutorials/plugin/memory)。

## 性能调优

### HNSW 索引参数

- **`index_m`**（默认 16）：更高的值可以提高召回，但代价是更多内存和索引工作。
- **`index_ef_construction`**（默认 256）：更高的值可以提高索引质量，但代价是更慢的构建。

用有代表性的数据调优这两个参数，并一起测量召回、延迟、构建时间和内存。没有适用于每个语料或容量目标的生产安全值。

### 内存容量

仅原始 float32 嵌入就使用每条目 `dimension * 4` 字节。实际内存更高，并取决于：

- 序列化内容、元数据和时间戳；
- Search 模块和 Valkey 版本；
- HNSW 图结构和索引参数；以及
- 分配器碎片和 Valkey 基础开销。

不要用固定的每条目乘数来确定生产容量。加载有代表性的数据集，检查 `INFO memory` 和 `FT.INFO <index>`，并为索引构建、复制和流量增长预留余量。

### 持久化

启用 AOF（Append-Only File）以获得持久性：

```bash
valkey-server --appendonly yes --appendfsync everysec
```

对于 RDB 快照（时间点备份）：

```bash
valkey-server --save 900 1 --save 300 10
```

## 故障排查

### Search 模块未加载

```
FT.CREATE failed: unknown command 'FT.CREATE'
```

确保你使用的是 `valkey/valkey-bundle`（包含 Search），而不是普通的 `valkey/valkey`：

```bash
valkey-cli MODULE LIST
# 应显示：name search ver ...
```

### 连接超时

```
valkey: connection timeout
```

- 验证主机名能解析：`nslookup valkey-memory`
- 检查端口连通性：`nc -zv valkey-memory 6379`
- 如果网络较慢，增加配置中的 `timeout`

### 索引已存在

Router 在启动时检查现有索引，如果已存在则跳过创建。如果需要重建索引（例如在更改 `dimension` 或 `metric_type` 之后）：

```bash
valkey-cli FT.DROPINDEX mem_idx
```

删除索引后重启 Router（或以其他方式重新初始化记忆存储）。索引创建在 Valkey 存储启动时运行，而不是在下一次请求时。在对实时存储使用此流程之前，先对照生产数据的副本测试重建。

### 内存不足

Valkey 将所有数据存储在内存中。如果达到内存上限：

1. 检查 `INFO memory` 和 `FT.INFO <index>`，以区分文档、索引和分配器增长。
2. 通过应用工作流删除过期或不需要的记忆，或缩短创建它们的保留策略。
3. 在更改 `maxmemory-policy` 之前增加容量或分片数据集；驱逐可能删除应用期望保留的记忆。

## 从 Milvus 迁移

要将现有部署从 Milvus 切换到 Valkey：

1. 更新 `config.yaml`，设置 `backend: valkey` 并添加 `valkey:` 块
2. 删除或注释掉 `milvus:` 块
3. 重启 Router——它会自动创建 Valkey 索引
4. Milvus 中的现有记忆**不会**自动迁移

:::warning
切换后端不会迁移数据，两个后端也不共享存储。管理 API 可以列出和删除记忆，但没有导入或批量导出端点，因此切换后 Valkey 存储从空开始，并随新流量重新积累记忆。
:::

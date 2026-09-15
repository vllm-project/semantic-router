---
sidebar_position: 7
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/installation/qdrant.md"
  outdated: false
---

# Qdrant 向量库

本指南介绍将 [Qdrant](https://qdrant.tech/) 部署为 Semantic Router 后端。Qdrant 可以用作语义缓存、智能体记忆存储、向量存储和路由回放存储。

## 前置条件

- Docker，或已配置 `kubectl` 的 Kubernetes 集群
- 对于 Kubernetes：已安装 Helm 3.x

## 使用 Docker 部署

### 快速开始

```bash
docker network inspect vllm-sr-network >/dev/null 2>&1 || \
  docker network create vllm-sr-network

docker run -d --name qdrant \
  --network vllm-sr-network \
  -p 127.0.0.1:6333:6333 \
  qdrant/qdrant:latest
```

验证 Qdrant 正在运行：

```bash
curl http://localhost:6333/healthz
```

### 带持久化

```bash
docker run -d --name qdrant \
  --network vllm-sr-network \
  -p 127.0.0.1:6333:6333 \
  -v qdrant-data:/qdrant/storage \
  qdrant/qdrant:latest
```

### 带 API 密钥认证

```bash
export QDRANT_API_KEY="$(openssl rand -hex 32)"

docker run -d --name qdrant \
  --network vllm-sr-network \
  -p 127.0.0.1:6333:6333 \
  -v qdrant-data:/qdrant/storage \
  -e QDRANT__SERVICE__API_KEY="$QDRANT_API_KEY" \
  qdrant/qdrant:latest
```

启用身份验证后，将相同的环境引用添加到 Router 使用的每个 Qdrant 块：

```yaml
api_key: ${QDRANT_API_KEY}
```

将值保留在进程环境或 Kubernetes Secret 中；不要把字面 API 密钥放入 Router 配置。当 Qdrant 服务器不需要身份验证时，省略 `api_key`。

主机映射仅在回环上暴露 HTTP 健康/API 端口。Router 通过共享 Docker 网络直接使用 Qdrant 的 gRPC 端口，因此不需要在主机上发布。下面配置中的主机名 `qdrant` 是 `vllm-sr-network` 上的 Docker DNS。如果用自定义栈名称启动 Router，例如 `VLLM_SR_STACK_NAME=team-a vllm-sr serve`，请将 Qdrant 附加到 `team-a-vllm-sr-network`，并使用匹配的可到达主机名。

Docker 示例对短期评估使用 `latest`。对于共享或生产部署，固定已发布的 Qdrant 版本或镜像 digest。

## 在 Kubernetes 中部署

### 使用 Helm

```bash
helm repo add qdrant https://qdrant.github.io/qdrant-helm
helm repo update

helm install qdrant qdrant/qdrant \
  --namespace vllm-semantic-router-system --create-namespace \
  --set persistence.size=10Gi
```

### 使用 StatefulSet

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: qdrant
  namespace: vllm-semantic-router-system
spec:
  serviceName: qdrant
  replicas: 1
  selector:
    matchLabels:
      app: qdrant
  template:
    metadata:
      labels:
        app: qdrant
    spec:
      containers:
        - name: qdrant
          image: qdrant/qdrant:latest
          ports:
            - containerPort: 6333
            - containerPort: 6334
          volumeMounts:
            - name: data
              mountPath: /qdrant/storage
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "2Gi"
              cpu: "1000m"
  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 10Gi
---
apiVersion: v1
kind: Service
metadata:
  name: qdrant
  namespace: vllm-semantic-router-system
spec:
  selector:
    app: qdrant
  ports:
    - name: rest
      port: 6333
      targetPort: 6333
    - name: grpc
      port: 6334
      targetPort: 6334
  clusterIP: None
```

该 StatefulSet 是未认证的评估示例。对于共享或生产集群，固定 chart 或镜像版本，通过 Kubernetes Secret 配置 API 密钥，在 Qdrant 和 Router 的 `api_key` / `use_tls` 绑定中启用 TLS，用 NetworkPolicy 限制访问，并为持久卷定义备份和恢复流程。

## 配置 Router

### 语义缓存

```yaml
global:
  stores:
    response_cache:
      enabled: true
      backend_type: qdrant
      similarity_threshold: 0.90
      ttl_seconds: 7200
      embedding_model: bert
      qdrant:
        host: qdrant                   # 服务名或主机名
        port: 6334
        use_tls: false
        collection_name: semantic_cache
        connect_timeout: 10
```

### 智能体记忆

```yaml
global:
  stores:
    memory:
      enabled: true
      backend: qdrant
      qdrant:
        host: qdrant
        port: 6334
        collection: agentic_memory
        dimension: 384               # 必须与嵌入模型匹配
      embedding_model: bert
      default_retrieval_limit: 5
      default_similarity_threshold: 0.70
```

### 已上传文档的向量存储

```yaml
global:
  stores:
    vector_store:
      enabled: true
      backend_type: qdrant
      file_storage_dir: /var/lib/vsr/data
      embedding_model: multimodal
      embedding_dimension: 384
      qdrant:
        host: qdrant
        port: 6334
        use_tls: false
        connect_timeout: 10
        collection_prefix: "vsr_vs_"
      metadata_store: memory
```

当多个 Router 副本必须看到同一份已上传文件注册表时，使用持久共享元数据，而不是 `memory`。嵌入维度必须与配置的嵌入模型匹配。

### 路由回放存储

```yaml
global:
  services:
    router_replay:
      enabled: true
      store_backend: qdrant
      qdrant:
        host: qdrant
        port: 6334
        collection_name: router_replay
```

这会启用 Router 范围的回放策略。仅当决策需要覆盖捕获或保留行为时，才添加路由局部 `router_replay` 插件；参见 [路由回放插件](../tutorials/plugin/router-replay)。

### 配置参考

全部四个 Qdrant 绑定都接受 `host`、`port`、可选的 `api_key` 和 `use_tls`。它们的 collection 字段有意不同：

| 能力 | Collection 字段 | 其他 Qdrant 专用字段 |
| --- | --- | --- |
| 响应缓存 | `collection_name` | `connect_timeout` |
| Agentic memory | `collection` | `dimension`、`connect_timeout` |
| 已上传文档的向量存储 | `collection_prefix` | `connect_timeout` |
| 路由回放 | `collection_name` | 回放 schema 中没有连接超时字段 |

启用身份验证时，对 `api_key` 使用 `${QDRANT_API_KEY}` 这样的环境引用。校验完整配置，而不是将一个能力的字段复制到另一个。

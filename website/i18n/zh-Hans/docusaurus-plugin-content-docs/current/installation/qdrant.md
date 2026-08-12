---
sidebar_position: 7
translation:
  source_commit: "39d7fa4b"
  source_file: "docs/installation/qdrant.md"
  outdated: false
---

# Qdrant

本指南介绍如何将 [Qdrant](https://qdrant.tech/) 部署为 Semantic Router 的后端。Qdrant 可用作语义缓存（semantic cache）、智能体记忆存储（agentic memory store）、向量存储（vector store）和路由器回放存储（router replay store）。

## 前置条件

- Docker，或者已配置 `kubectl` 的 Kubernetes 集群
- Kubernetes 场景：已安装 Helm 3.x

## 使用 Docker 部署

### 快速开始

```bash
docker run -d --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  qdrant/qdrant:latest
```

验证 Qdrant 是否正在运行：

```bash
curl http://localhost:6333/healthz
```

### 启用持久化

```bash
docker run -d --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v qdrant-data:/qdrant/storage \
  qdrant/qdrant:latest
```

### 启用 API Key 身份验证

```bash
docker run -d --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v qdrant-data:/qdrant/storage \
  -e QDRANT__SERVICE__API_KEY=your-secret-key \
  qdrant/qdrant:latest
```

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

## 配置路由器

### 语义缓存（Semantic Cache）

```yaml
global:
  stores:
    semantic_cache:
      enabled: true
      backend_type: qdrant
      similarity_threshold: 0.90
      ttl_seconds: 7200
      embedding_model: bert
      qdrant:
        host: qdrant                   # 服务名称或主机名
        port: 6334
        api_key: ""
        use_tls: false
        collection_name: semantic_cache
        connect_timeout: 10
```

### 智能体记忆（Agentic Memory）

```yaml
global:
  stores:
    memory:
      enabled: true
      backend: qdrant
      qdrant:
        host: qdrant
        port: 6334
        api_key: ""
        collection: agentic_memory
        dimension: 384               # 必须与你的嵌入模型匹配
      embedding_model: bert
      default_retrieval_limit: 5
      default_similarity_threshold: 0.70
```

### 路由器回放存储（Router Replay Store）

```yaml
global:
  services:
    router_replay:
      store_backend: qdrant
      qdrant:
        host: qdrant
        port: 6334
        api_key: ""
        collection_name: router_replay
```

### 配置参考

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `host` | `localhost` | Qdrant 服务器主机名 |
| `port` | `6334` | Qdrant gRPC 端口 |
| `api_key` | _（空）_ | 用于身份验证的 API 密钥 |
| `use_tls` | `false` | 为 gRPC 连接启用 TLS |
| `collection_name` | 视情况而定 | 要使用的 collection（若不存在则自动创建） |
| `connect_timeout` | `10` | 连接超时时间，以秒为单位 |

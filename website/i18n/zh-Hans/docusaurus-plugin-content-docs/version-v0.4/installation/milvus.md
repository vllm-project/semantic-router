---
title: Milvus 向量库
sidebar_label: Milvus 向量库
description: 将 Milvus 用作响应缓存和其他 Router 存储功能的持久向量后端。
translation:
  source_commit: "e56591a9cb24f073bf159927e87116ba6d278741"
  source_file: "docs/installation/milvus.md"
  outdated: false
---

# Milvus 向量库

Milvus 是一个分布式向量数据库，Semantic Router 可以将其用于持久响应缓存和其他向量支持的功能。当数据集必须超出单个 Router 进程、在重启后保留，或由多个 Router 副本共享时，选择它。

对于小型本地部署，内存缓存更简单。当团队已经在运维其向量搜索扩展时，Valkey 或 Redis 可能更合适。Qdrant 是另一个专用向量存储选项。在选择后端之前，参见[数据与存储](storage-overview)。

## 本页配置的内容

下面的示例将 Milvus 用于 `global.stores.response_cache`。决策仍需要 `response_cache` 插件，请求才会使用该存储。

Milvus 也可以支撑智能体记忆或通用向量存储，但这些功能有单独的 schema 和保留要求。当它们的数据生命周期不同时，使用不同的 collection。

## 前置条件

- Kubernetes 集群和 Helm，或现有可到达的 Milvus 部署
- 适合你持久性目标的持久存储
- 每个 Router 副本到 Milvus gRPC（默认 19530）的私有网络访问
- 与 Router 所选嵌入模型匹配的嵌入维度

## 用 Helm 部署 Milvus

Milvus 项目维护 Helm chart。以下启动适合开发和评估的独立部署：

```bash
helm repo add milvus https://zilliztech.github.io/milvus-helm/
helm repo update

helm upgrade --install milvus milvus/milvus \
  --namespace milvus \
  --create-namespace \
  --set cluster.enabled=false
```

等待工作负载就绪并检查其 Service：

```bash
kubectl get pods,service -n milvus
kubectl wait --for=condition=Ready pod \
  -l app.kubernetes.io/instance=milvus \
  -n milvus --timeout=10m
```

对于生产，使用你的 Milvus 发行版所记录的拓扑、对象存储、元数据存储、持久化、备份和升级流程。固定 chart 版本并复核其 values，而不是复制开发默认值。

## 配置响应缓存

使用 canonical `response_cache` 键。`semantic_cache` 是仅保留用于迁移兼容性的已弃用输入别名。

```yaml
global:
  stores:
    response_cache:
      enabled: true
      backend_type: milvus
      similarity_threshold: 0.86
      max_entries: 50000
      ttl_seconds: 7200
      embedding_model: mmbert
      milvus:
        connection:
          host: milvus.milvus.svc.cluster.local
          port: 19530
          database: default
          timeout: 30
        collection:
          name: semantic_router_response_cache
          description: Semantic Router response-cache vectors
          vector_field:
            name: embedding
            dimension: 768
            metric_type: COSINE
          index:
            type: HNSW
            params:
              M: 16
              efConstruction: 200
        search:
          params:
            ef: 64
          topk: 10
          consistency_level: Bounded
        development:
          drop_collection_on_startup: false
          auto_create_collection: true
```

将 `dimension` 设置为 `embedding_model` 的输出维度。不匹配会导致插入或搜索失败。

在可能读取或填充缓存的决策上启用路由插件：

```yaml
routing:
  decisions:
    - name: general-chat
      description: General requests that may use response cache.
      priority: 100
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: local/general
      plugins:
        - type: response_cache
          configuration:
            enabled: true
            semantic:
              similarity_threshold: 0.86
```

在发布前运行配置校验：

```bash
vllm-sr config validate --config config.yaml
```

## 网络和传输安全

当前响应缓存连接器使用 `connection.host` 和 `connection.port` 打开未认证的明文 gRPC 连接。它不会应用 Milvus 用户名/密码或 TLS 设置，因此不要添加这些字段并期望响应缓存客户端强制执行它们。

将此连接保留在私有网络上。在 Kubernetes 中，使用 NetworkPolicy 仅允许 Router 工作负载到达 Milvus Service，并拒绝无关命名空间。不要公开暴露该 Service。如果你的环境要求经过认证或端到端 TLS 的数据库连接，使用当前 Router 集成支持该要求的后端，或在 Milvus 前面放置经过审核的集群内传输代理，并在生产发布前测试完整路径。

## 验证行为

部署之后：

1. 确认 Router 变为就绪，并记录成功的 Milvus 连接；
2. 通过带有 `response_cache` 插件的决策发送请求；
3. 重复等价请求，并检查缓存指标或路由元数据；
4. 确认预期 collection 存在，并且其向量维度正确；以及
5. 从每个 Router 副本演练同一路径。

不要将固定的延迟预期作为健康检查。查找时间取决于网络距离、索引大小、索引参数、一致性级别、存储和硬件。用你的数据集和部署来测量。

## 从其他缓存迁移

响应缓存条目是派生数据，因此最安全的迁移通常是启动一个新的空 Milvus collection 并让它预热：

1. 部署并保护 Milvus；
2. 添加 Milvus 配置，但不移除旧部署的回滚路径；
3. 校验并向一小部分流量发布；
4. 监控连接错误、缓存命中率、内存和请求延迟；
5. 扩大发布范围；以及
6. 在回滚窗口过期后退役先前的缓存。

如果 collection 包含持久记忆或已上传文档，而不是可重建的缓存条目，请遵循该功能专用的数据迁移和备份流程。不要将这些 collection 视为可丢弃。

## 备份和保留

根据所存储的数据定义保留，而不是仅根据 Milvus。响应缓存可能包含从请求派生的嵌入、元数据或响应。限制访问，设置 TTL，并记录删除行为。

对持久 collection 使用 Milvus 项目支持的备份工具，并在隔离环境中测试恢复。将 Milvus 版本、collection schema、嵌入模型和维度与备份一起记录。

## 故障排查

### `milvus configuration is required`

`backend_type: milvus` 需要嵌套的 `global.stores.response_cache.milvus` 块。检查缩进并校验完整配置。

### Collection 不存在

对于开发，设置 `development.auto_create_collection: true`。在受控生产环境中，预先创建 collection 并禁用自动创建。确保 schema 和向量维度与 Router 配置匹配。

### 连接超时

检查 Service 和端点、Router 命名空间的 DNS、NetworkPolicy，以及配置的数据库：

```bash
kubectl get service,endpoints -n milvus
kubectl get networkpolicy -A
```

### 搜索质量差

确认训练和推理使用相同的嵌入模型和维度。然后对照有代表性的流量调整响应缓存相似度阈值和 Milvus 搜索/索引参数。不要在未经评估的情况下从另一个嵌入模型复制阈值。

## 参考

- [Milvus 文档](https://milvus.io/docs)
- [Response Cache 插件](../tutorials/plugin/response-cache)
- [数据与存储](storage-overview)

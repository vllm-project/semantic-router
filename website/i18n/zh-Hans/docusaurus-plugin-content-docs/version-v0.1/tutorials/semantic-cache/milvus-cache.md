---
translation:
  source_commit: "bac2743"
  source_file: "docs/tutorials/semantic-cache/milvus-cache.md"
  outdated: false
---

# Milvus 语义缓存 (Milvus Semantic Cache)

Milvus 缓存后端使用 Milvus 向量数据库提供持久化、分布式的语义缓存。这是需要高可用性、可扩展性和数据持久性的生产环境部署的推荐方案。

## 概览

Milvus 缓存非常适合：

- 具有高可用性要求的**生产环境**
- 跨多个实例的**分布式部署**
- 具有数百万个缓存查询的**大规模应用**
- 缓存需在重启后保留的**持久化存储**要求
- **先进的向量操作**和相似度搜索优化

## 架构

```mermaid
graph TB
    A[客户端请求] --> B[语义缓存实例 1]
    A --> C[语义缓存实例 2]
    A --> D[语义缓存实例 N]

    B --> E[生成查询嵌入]
    C --> E
    D --> E

    E --> F[Milvus 向量数据库]
    F --> G{是否找到相似向量？}

    G -->|命中| H[返回缓存的响应]
    G -->|未命中| I[转发到 LLM]

    I --> J[LLM 处理]
    J --> K[在 Milvus 中存储向量 + 响应]
    J --> L[返回响应]

    K --> M[持久化存储]
    H --> N[更新命中指标]

    style H fill:#90EE90
    style K fill:#FFB6C1
    style M fill:#DDA0DD
```

## 配置

### Milvus 后端配置

在 `config/semantic-cache/milvus.yaml` 中配置：

```yaml
# config/semantic-cache/milvus.yaml
connection:
  host: "localhost"
  port: 19530
  auth:
    enabled: false
    username: ""
    password: ""
  tls:
    enabled: false

collection:
  name: "semantic_cache"
  dimension: 384  # 必须与嵌入模型的维度匹配
  index_type: "IVF_FLAT"
  metric_type: "COSINE"
  nlist: 1024

performance:
  search_params:
    nprobe: 10
  insert_batch_size: 1000
  search_batch_size: 100

development:
  drop_collection_on_startup: false
  auto_create_collection: true
  log_level: "info"
```

## 设置与部署

### 1. 启动 Milvus 服务

```bash
# 使用 Docker
make start-milvus

# 验证 Milvus 是否正在运行
curl http://localhost:19530/health
```

### 2. 配置Semantic Router 

基础 Milvus 配置：

- 在 `config/config.yaml` 中设置 `backend_type: "milvus"`
- 在 `config/config.yaml` 中设置 `backend_config_path: "config/semantic-cache/milvus.yaml"`

```yaml
# config/config.yaml
semantic_cache:
  enabled: true
  backend_type: "milvus"
  backend_config_path: "config/semantic-cache/milvus.yaml"
  similarity_threshold: 0.8
  ttl_seconds: 7200
```

### 决策级配置（基于插件）

您还可以使用插件在决策级别配置 Milvus 缓存：

```yaml
signals:
  domains:
    - name: "math"
      description: "数学查询"
      mmlu_categories: ["math"]

decisions:
  - name: math_route
    description: "路由数学查询并使用严格缓存"
    priority: 100
    rules:
      operator: "AND"
      conditions:
        - type: "domain"
          name: "math"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: true
    plugins:
      - type: "semantic-cache"
        configuration:
          enabled: true
          similarity_threshold: 0.95  # 对数学准确性要求非常严格
```

### 3. 运行Semantic Router 

```bash
# 启动路由
make run-router
```

运行 EnvoyProxy：

```bash
# 启动 Envoy 代理
make run-envoy
```

### 4. 测试 Milvus 缓存

```bash
# 发送完全相同的请求以观察缓存命中
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MoM",
    "messages": [{"role": "user", "content": "什么是机器学习？"}]
  }'

# 发送相似请求（应由于语义相似度而命中缓存）
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MoM",
    "messages": [{"role": "user", "content": "解释机器学习"}]
  }'
```

## 下一步

- **[内存缓存](./in-memory-cache.md)** - 与内存缓存进行对比
- **[可观测性](../observability/metrics.md)** - 监控 Milvus 性能
- **[Kubernetes 集成](../../installation/milvus.md)** - 在 Kubernetes 上部署 Milvus

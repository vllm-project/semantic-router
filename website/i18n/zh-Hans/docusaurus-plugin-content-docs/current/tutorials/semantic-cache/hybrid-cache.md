---
translation:
  source_commit: "bac2743"
  source_file: "docs/tutorials/semantic-cache/hybrid-cache.md"
  outdated: false
---

# 混合缓存 (Hybrid Cache): HNSW + Milvus

混合缓存结合了内存中的 HNSW 索引（用于快速搜索）和 Milvus 向量数据库（用于可扩展、持久化的存储）。

## 概览

混合架构提供：

- **快速搜索**：通过内存中的 HNSW 索引实现
- **可扩展存储**：通过 Milvus 向量数据库实现
- **持久性**：以 Milvus 作为单一事实来源 (Source of Truth)
- **热数据缓存**：带有本地文档缓存

## 架构

```
┌──────────────────────────────────────────────────┐
│                  混合缓存 (Hybrid Cache)          │
├──────────────────────────────────────────────────┤
│  ┌─────────────────┐      ┌──────────────────┐  │
│  │  内存中的        │      │   本地缓存       │  │
│  │  HNSW 索引      │◄─────┤   (热数据)       │  │
│  └────────┬────────┘      └──────────────────┘  │
│           │                                       │
│           │ ID 映射                              │
│           ▼                                       │
│  ┌──────────────────────────────────────────┐   │
│  │         Milvus 向量数据库                 │   │
│  └──────────────────────────────────────────┘   │
└──────────────────────────────────────────────────┘
```

## 工作原理

### 写入路径 (AddEntry)

添加缓存条目时：

1. 使用配置的嵌入模型生成嵌入
2. 将条目写入 Milvus 以实现持久化
3. 将条目添加到内存中的 HNSW 索引（如果空间允许）
4. 将文档添加到本地缓存

### 读取路径 (FindSimilar)

搜索相似查询时：

1. 生成查询嵌入
2. 在 HNSW 索引中搜索最近邻
3. 在本地缓存中检查匹配的文档
    - 如果在本地缓存中找到：立即返回（热路径）
    - 如果未找到：从 Milvus 获取（冷路径）
4. 将获取到的文档缓存在本地缓存中，以备将来查询

### 内存管理

- **HNSW 索引**：受限于配置的最大条目数
- **本地缓存**：受限于配置的文档数量
- **驱逐策略**：达到限制时采用 FIFO (先进先出) 策略
- **数据持久性**：无论内存限制如何，所有数据都保留在 Milvus 中

## 配置

### 基础配置

```yaml
semantic_cache:
  enabled: true
  backend_type: "hybrid"
  similarity_threshold: 0.85
  ttl_seconds: 3600
  
  # 混合缓存特定设置
  max_memory_entries: 100000  # HNSW 中的最大条目数
  local_cache_size: 1000      # 本地文档缓存大小
  
  # HNSW 参数
  hnsw_m: 16
  hnsw_ef_construction: 200
  
  # Milvus 配置
  backend_config_path: "config/semantic-cache/milvus.yaml"
```

### 决策级配置（基于插件）

您还可以使用插件在决策级别配置混合缓存：

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

### 配置参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `backend_type` | string | - | 必须为 `"hybrid"` |
| `similarity_threshold` | float | 0.85 | 缓存命中的最小相似度 |
| `max_memory_entries` | int | 100000 | HNSW 索引中的最大条目数 |
| `local_cache_size` | int | 1000 | 热文档缓存大小 |
| `hnsw_m` | int | 16 | HNSW 双向链路数 |
| `hnsw_ef_construction` | int | 200 | HNSW 构建质量 |
| `backend_config_path` | string | - | Milvus 配置文件路径 |

### Milvus 配置

创建 `config/semantic-cache/milvus.yaml`：

```yaml
milvus:
  address: "localhost:19530"
  collection_name: "semantic_cache"
  dimension: 384
  index_type: "HNSW"
  metric_type: "IP"
  params:
    M: 16
    efConstruction: 200
```

## 使用示例

### Go 代码

```go
import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"

// 初始化混合缓存
options := cache.HybridCacheOptions{
    Enabled:             true,
    SimilarityThreshold: 0.85,
    TTLSeconds:          3600,
    MaxMemoryEntries:    100000,
    HNSWM:               16,
    HNSWEfConstruction:  200,
    MilvusConfigPath:    "config/semantic-cache/milvus.yaml",
    LocalCacheSize:      1000,
}

hybridCache, err := cache.NewHybridCache(options)
if err != nil {
    log.Fatalf("创建混合缓存失败: %v", err)
}
defer hybridCache.Close()

// 添加缓存条目
err = hybridCache.AddEntry(
    "request-id-123",
    "gpt-4",
    "什么是量子计算？",
    []byte(`{"prompt": "什么是量子计算？"}`),
    []byte(`{"response": "量子计算是..."}`),
)

// 搜索相似查询
response, found, err := hybridCache.FindSimilar(
    "gpt-4",
    "解释量子计算机",
)
if found {
    fmt.Printf("缓存命中！响应: %s\n", string(response))
}

// 获取统计信息
stats := hybridCache.GetStats()
fmt.Printf("HNSW 中的总条目数: %d\n", stats.TotalEntries)
fmt.Printf("命中率: %.2f%%\n", stats.HitRatio * 100)
```

## 监控与指标

混合缓存公开了用于监控的指标：

```go
stats := hybridCache.GetStats()

// 可用指标
stats.TotalEntries  // HNSW 索引中的条目数
stats.HitCount      // 总缓存命中次数
stats.MissCount     // 总缓存未命中次数
stats.HitRatio      // 命中率 (0.0 - 1.0)
```

### Prometheus 指标

```
# HNSW 中的缓存条目数
semantic_cache_entries{backend="hybrid"}

# 缓存操作
semantic_cache_operations_total{backend="hybrid",operation="find_similar",status="hit_local"}
semantic_cache_operations_total{backend="hybrid",operation="find_similar",status="hit_milvus"}
semantic_cache_operations_total{backend="hybrid",operation="find_similar",status="miss"}

# 缓存命中率
semantic_cache_hit_ratio{backend="hybrid"}
```

## 多实例部署

混合缓存支持多实例部署，其中每个实例维护自己的 HNSW 索引和本地缓存，但共享 Milvus 以实现持久化和数据一致性：

```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│   实例 1    │   │   实例 2    │   │   实例 3    │
│  HNSW 缓存  │   │  HNSW 缓存  │   │  HNSW 缓存  │
└──────┬──────┘   └──────┬──────┘   └──────┬──────┘
       │                 │                 │
       └─────────────────┼─────────────────┘
                         │
                  ┌──────▼──────┐
                  │   Milvus    │
                  │   (共享)    │
                  └─────────────┘
```

## 另请参阅

- [内存缓存文档](./in-memory-cache.md)
- [Milvus 缓存文档](./milvus-cache.md)

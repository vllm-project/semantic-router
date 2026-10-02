---
translation:
  source_commit: "0f2ba0de7c435366ed68bcf03f5a1bb49b9cb90c"
  source_file: "docs/tutorials/global/stores-and-tools.md"
  outdated: false
---

# 存储与工具

## 概览

本页介绍 `global:` 中的共享存储和工具块。

这些设置为路由局部插件和路由器级工具行为提供支撑。

## 主要优势

- 集中共享后台存储，而不是按路由重复配置。
- 让响应缓存、memory、检索和工具目录保持一致。
- 让路由局部插件保持小而专注。
- 明确共享基础设施依赖。

## 解决什么问题？

路由局部插件常常依赖共享存储或工具状态。若这些依赖在各路由中临时配置，系统会不一致且更难运维。

这些 `global:` 块通过一次定义共享后台服务来解决该问题。

## 何时使用

在以下情况使用这些块：

- 多条路由依赖同一响应缓存或 memory 后端
- 检索功能需要一个共享向量存储
- 路由器应暴露一个共享工具目录
- 后台存储配置属于整台路由器，而不是单条路由

## 配置

### 响应缓存 {#response-cache}

```yaml
global:
  stores:
    response_cache:
      enabled: true
      backend_type: memory
      similarity_threshold: 0.8
      polarity_guard:
        mode: lexical          # lexical | nli | lexical+nli
        nli:
          contradiction_threshold: 0.5
```

#### 否定防护 {#negation-guard}

双编码器相似度可能无法区分 *“turn on dark mode”* 和 *“turn off dark mode”*：相反含义的查询常常超过 `similarity_threshold`，而真正的改写可能低于它，因此提高阈值不能可靠地避免误命中。所有缓存后端都会在返回语义候选前执行词面校验。`polarity_guard` 还可为内存后端选择额外的校验器：

- `lexical`（默认）：对词面接近的英文问题，检查明确否定线索和已知反义词替换。它在内存、Redis、Valkey、Milvus、Qdrant 和混合缓存中始终开启，不需要模型；不覆盖缺少词面线索、仅词序变化或非英文的含义变化。
- `nli` / `lexical+nli`：内存后端额外对最佳候选运行一次 NLI 校验，在矛盾概率超过 `nli.contradiction_threshold` 时拒绝命中。该层使用幻觉解释模型（`global.model_catalog.modules.hallucination_mitigation.explainer`，默认 `tasksource/ModernBERT-base-nli`），缓存独立于配方分类器持有校验器。选择 NLI 模式但未配置该模型时，配置加载失败。CPU 上每次校验约需 70 ms；缓存命中仍可省去完整生成。如果查询时模型不可用或执行出错，未能校验的候选视为缓存未命中，请求继续到模型后端，并记录 `cache_polarity_nli_skipped` 警告。

NLI 拒绝会记录为带 `tier: nli` 的 `cache_negation_reject`，计入未命中，相似度仍通过 `x-vsr-cache-similarity` 提供。远端和混合缓存仅执行词面层：缺少原始问题的候选被拒绝，并继续检查有数量上限的已检索候选。混合缓存回退到 Milvus 时同样执行该检查。

### 记忆 {#memory}

记忆存储支持三种后端：`milvus`（默认）、`valkey` 和 `qdrant`。

**Milvus 后端**（默认）：

```yaml
global:
  stores:
    memory:
      enabled: true
      milvus:
        address: milvus:19530
        collection: agentic_memory
        dimension: 384
```

**Valkey 后端**（需要带 Search 模块的 Valkey）：

```yaml
global:
  stores:
    memory:
      enabled: true
      backend: valkey
      valkey:
        host: valkey
        port: 6379
        dimension: 384
        collection_prefix: "mem:"
        index_name: mem_idx
        metric_type: COSINE
```

**Qdrant 后端**：

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
        dimension: 384
      embedding_model: bert
      default_retrieval_limit: 5
      default_similarity_threshold: 0.30
```

此 Qdrant 示例对 `bert`（`mom-embedding-light`）使用原始余弦分数。
0.30 仅是小规模冷启动召回测试的起点，不适用于其他嵌入模型；上线前还需检查无关查询和已更正的旧事实。

完整部署说明见：

- [Valkey 智能体记忆](../../installation/valkey-memory) — Docker、Kubernetes、配置参考、调优和排障
- [Qdrant](../../installation/qdrant) — Docker、Kubernetes、配置参考、调优和排障
- `config/runtime/memory/` 提供后端专用配置参考

当配置了带 `model_role: memory_rewrite` 的外部模型时，其 `max_response_bytes` 限制每次查询改写响应。省略或非正值使用 1 MiB 默认值。

### 向量存储 {#vector-store}

```yaml
global:
  stores:
    vector_store:
      enabled: true
      backend_type: milvus
      metadata_store: postgres
```

支持的后端：`memory`、`milvus`、`llama_stack`、`valkey`、`qdrant`。

`metadata_store` 控制向量存储和已上传文件元数据的注册表。本地或类生产堆栈若需重启安全，请使用 `postgres`；CLI 本地运行时在设置 `metadata_store: postgres` 时会配置 Postgres 并填充 `metadata_postgres` 连接默认值。仅对短暂的本地实验使用 `memory`，因为存储和文件元数据会在路由器重启后丢失。

使用本地 `mmbert` 嵌入（包括 Vela Embedding）时，每个新向量存储都会记录创建向量所用的表示身份。Candle `bert` 存储也会记录编码器版本，使修正后的无填充向量不会与此前包含填充影响的向量混用。旧存储仍可见，上传文件仍保留。搜索不兼容或无身份标记的存储，或向其中关联文件，会返回 `409 EMBEDDING_REINDEX_REQUIRED`。请创建新向量存储并重新关联原上传文件 ID，以生成兼容向量。客户端元数据不能替换 Router 管理的 `_router_embedding_identity` 字段。

同样的检查适用于请求时 RAG 和缓存检索结果。`llama_stack` 在远端生成搜索查询向量，因此目前不能与绑定身份的本地 `mmbert` 或 Candle `bert` 文档向量组合；这类配置请使用 `memory`、`milvus`、`valkey` 或 `qdrant`。远程提供方的身份验证属于另一项能力。

### 工具 {#tools}

```yaml
global:
  integrations:
    tools:
      enabled: true
      top_k: 3
      tools_db_path: config/runtime/tools/tools_db.json
```

## 数据与安全 {#data-and-security}

- 缓存、memory 和向量存储可能包含提示词、响应、嵌入、检索文档或提取的记忆。为所选后端配置认证、加密、保留策略和租户/用户范围。
- 嵌入维度必须与现有集合匹配。嵌入模型或维度变更时，请重建或迁移索引。
- 工具检索控制向模型展示什么；它不授权工具执行。请在工具服务上强制权限。
- 见[完整后端示例](https://github.com/vllm-project/semantic-router/tree/main/config/runtime)以及 [`config/config.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml) 中的完整配置契约。

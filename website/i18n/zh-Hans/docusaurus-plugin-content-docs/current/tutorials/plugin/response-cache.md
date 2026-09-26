---
translation:
  source_commit: "0f2ba0de7c435366ed68bcf03f5a1bb49b9cb90c"
  source_file: "docs/tutorials/plugin/response-cache.md"
  outdated: false
---

# 响应缓存

## 概览

`response_cache` 是路由局部插件，用于复用精确或语义兼容的先前响应。

## 主要优势

- 仅在受益于缓存命中的路由上复用先前响应。
- 将路由局部阈值与全局存储设置分开。
- 支持不同路由使用不同缓存策略。

## 解决什么问题？

有些路由强烈受益于复用，另一些则每次都需要全新生成。`response_cache` 将复用策略限制在路由内。

## 何时使用

- 某条路由应在查询非常相似时优先使用缓存响应
- 不同路由需要不同的相似度阈值或 TTL
- 该路由应使用配置在 `global.stores.response_cache` 中的缓存后端

## 配置

在 `routing.decisions[].plugins` 下添加该插件：

```yaml
plugins:
  - type: response_cache
    configuration:
      enabled: true
      mode: exact
      scope: user
      ttl_seconds: 86400
      request_controls:
        enabled: true
        header: x-vsr-cache-control
        allowed: [no-cache, no-store, bypass, max-age, ttl]
        max_ttl_seconds: 86400
      personalized:
        mode: disabled
```

`mode` 接受：

- `semantic`（默认）：仅向量查找。
- `exact`：仅规范化后的精确请求查找。
- `exact_then_semantic`：先精确查找，未命中再进行向量查找。

随附的 `config/config.yaml`、多目标路由示例和 `memory.yaml` 片段使用
`exact`：相同请求仍可命中，不会仅凭向量相似度把另一个问题的答案返回给用户。
只有验证了该路由所用语言和矛盾问句的行为后，才应显式选择 `semantic` 或
`exact_then_semantic`。内置词面校验只识别英语否定词；德语 `nicht` 或中文
`不` 的反向问题可能命中原问题的缓存答案。`high-recall.yaml` 仍是显式启用
语义缓存的示例。

精确层级可用于内存、Redis、Valkey、Milvus、Qdrant 和混合缓存后端。Anthropic 客户端请求会以 Anthropic 响应或 SSE 线格式回放。

流式和非流式请求使用分开的缓存身份，因此回放不会跨线模式翻译缓存响应。语义匹配使用覆盖 system/历史、工具、响应格式、生成参数、客户端协议和路由策略的兼容指纹，以及硬性的配方、租户、请求模型和所选模型分区。

流式回放会为完整的单选或多选流保留内容、推理、拒绝、工具调用、终端 usage、结束原因和 choice 索引。不完整的流永不缓存。

启用请求控制时，配置的请求头接受已授权指令。`max-age` 限制读取新鲜度，`ttl` 限制写入寿命；调用方 TTL 值会被限制到 `max_ttl_seconds`。

## 迁移 {#migration}

`semantic-cache`、`semantic_cache` 和 `response-cache` 作为已弃用别名被接受，并规范化为 `response_cache`。同样，`global.stores.semantic_cache` 会被读取为 `global.stores.response_cache` 的已弃用别名。不要在同一文档中同时配置两种拼写。导出、控制面板保存和 DSL 反编译始终发出规范名称。

本地 `mmbert` 嵌入（包括 Vela Embedding）更换模型、分词器、向量表示大小或推理设置后，会使用独立的缓存空间。租户命名空间和显式缓存版本保持不变；旧条目按原有过期时间保留，也可显式清理。升级模型后的首次请求会缓存未命中，使用相同向量表示重启则可复用兼容缓存。语义缓存需要本地分词器窗口，因此 Router 会拒绝为其使用[远程嵌入端点](../../installation/runtime/embeddings.md#remote-embeddings)。

Candle `bert` 嵌入改用编码器版本区分缓存空间。每当 Candle BERT 的向量发生变化，这个版本就会随之更新，例如填充 token 不再计入平均值时。跨越这类变化升级后，BERT 会使用新的缓存空间：升级前写入的条目不会被复用，并保留到过期为止，缓存会随新流量重新填充。由其他运行时提供的 BERT 保留已有缓存。

## 运维 {#operations}

管理 API 在 `/api/v1/storage/response-cache/*` 下暴露经过脱敏的健康、能力、统计、候选配置测试、限定范围失效、基于 epoch 的清空。统一哈希链审计位于 `/api/v1/observability/audit`，需要 `audit.read` 权限。`/api/v1/plugins/response_cache` 提供插件发现与操作链接。失效默认是 dry-run。清空需要显式确认短语 `flush response cache`，并且永不调用后端范围的 `FLUSHALL`。

六种缓存后端在返回语义命中前，都会执行始终开启的英文词面校验。对于词面接近、但出现明确否定或已知反义词替换的问题，即使向量相似度较高，也会拒绝该候选。远端条目缺少原始问题时同样视为未命中。拒绝一个候选后，仍可使用后续已检索到的合格候选；远端检索保持候选数量上限。这项检查不保证识别仅词序变化、缺少词面线索或非英文的含义变化。

内存后端还支持可选 NLI 校验器（`global.stores.response_cache.polarity_guard`；见[存储与工具](../global/stores-and-tools.md#negation-guard)）。启用该层级时，NLI 拒绝的候选会记录为带 `tier: nli` 的 `cache_negation_reject`，报告为未命中，其相似度仍出现在 `x-vsr-cache-similarity` 上，以便接近阈值的拒绝可被诊断。

缓存响应可能包含用户或租户数据。请选择合适的范围、TTL、后端认证、加密和失效流程。语义阈值必须针对配置的嵌入模型校准。长于嵌入模型上下文窗口的查询（默认 `bert` 模型为 512 个 token）不会被缓存，因为截断嵌入会匹配所有共享该前缀的查询。带个性化 RAG 或 memory 的路由，若没有显式策略，不应复用富化前的响应。完整示例见：
[`high-recall.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/response-cache/high-recall.yaml)
和
[`memory.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/response-cache/memory.yaml)。

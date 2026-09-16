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
      mode: exact_then_semantic
      scope: user
      semantic:
        similarity_threshold: 0.92
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

精确层级可用于内存、Redis、Valkey、Milvus、Qdrant 和混合缓存后端。Anthropic 客户端请求会以 Anthropic 响应或 SSE 线格式回放。

流式和非流式请求使用分开的缓存身份，因此回放不会跨线模式翻译缓存响应。语义匹配使用覆盖 system/历史、工具、响应格式、生成参数、客户端协议和路由策略的兼容指纹，以及硬性的配方、租户、请求模型和所选模型分区。

流式回放会为完整的单选或多选流保留内容、推理、拒绝、工具调用、终端 usage、结束原因和 choice 索引。不完整的流永不缓存。

启用请求控制时，配置的请求头接受已授权指令。`max-age` 限制读取新鲜度，`ttl` 限制写入寿命；调用方 TTL 值会被限制到 `max_ttl_seconds`。

## 迁移 {#migration}

`semantic-cache`、`semantic_cache` 和 `response-cache` 作为已弃用别名被接受，并规范化为 `response_cache`。同样，`global.stores.semantic_cache` 会被读取为 `global.stores.response_cache` 的已弃用别名。不要在同一文档中同时配置两种拼写。导出、控制面板保存和 DSL 反编译始终发出规范名称。

本地 `mmbert` 嵌入（包括 Vela Embedding）更换模型、分词器、向量表示大小或推理设置后，会使用独立的缓存空间。租户命名空间和显式缓存版本保持不变；旧条目按原有过期时间保留，也可显式清理。升级模型后的首次请求会缓存未命中，使用相同向量表示重启则可复用兼容缓存。这项绑定不会自动识别可变远程嵌入端点的模型身份。

## 运维 {#operations}

管理 API 在 `/api/v1/response-cache/*` 下暴露经过脱敏的健康、能力、统计、候选配置测试、限定范围失效、基于 epoch 的清空，以及哈希链式审计视图。失效默认是 dry-run。清空需要显式确认短语 `flush response cache`，并且永不调用后端范围的 `FLUSHALL`。

内存后端可以在返回语义命中之前，对照相反含义的查询进行校验（`global.stores.response_cache.polarity_guard`；见[存储与工具](../global/stores-and-tools.md#negation-guard)）。启用可选 NLI 层级时，被拒绝的候选会记录为带 `tier: nli` 的 `cache_negation_reject`，报告为未命中，其相似度仍出现在 `x-vsr-cache-similarity` 上，以便接近阈值的拒绝可被诊断。

缓存响应可能包含用户或租户数据。请选择合适的范围、TTL、后端认证、加密和失效流程。语义阈值必须针对配置的嵌入模型校准。长于嵌入模型上下文窗口的查询（默认 `bert` 模型为 512 个 token）不会被缓存，因为截断嵌入会匹配所有共享该前缀的查询。带个性化 RAG 或 memory 的路由，若没有显式策略，不应复用富化前的响应。完整示例见：
[`high-recall.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/response-cache/high-recall.yaml)
和
[`memory.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/response-cache/memory.yaml)。

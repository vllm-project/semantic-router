---
is_mtpe: false
translation:
  source_commit: "9156d5bc1ed9edff626b95a2b8260a77cb1712c5"
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

在 `recipes[].decisions[].plugins` 下添加该插件：

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

### 语义复用 {#semantic-reuse}

语义层通过 `semantic.match` 中的 [match 块](./overview#match-block) 查找请求可以复用的缓存请求。不写时使用响应缓存的默认值，即先召回再确认：

```yaml
semantic:
  match:
    recall: {method: embedding, top_k: 3, min_similarity: 0.8}
    judge:  {preset: same_request, min_score: 0.8}
```

- `recall` 向缓存存储召回与请求最相似的缓存请求，最多 `top_k` 个，相似度（余弦）不低于 `min_similarity`。
- `judge` 在一次调用中向决策模型询问 `same_request` 问题（请求与每个召回的请求，各占批量请求中的一项），复用得分不低于 `min_score` 的最佳一个，从而拒绝嵌入相近但问的是另一件事（例如相反问题）的缓存请求：被 judge 否决的召回请求就是未命中。judge 失败时，`on_error: skip`（默认）复用召回结果，`on_error: fail` 不复用。
- 只写 `recall` 的块复用最相似的缓存请求，不询问任何模型。

### 配合响应守卫 {#with-a-response-guard}

在同时配置了 [`response_guard`](./response-guard.md) 的决策上，缓存只保存守卫放行的回答：被拦截的回答，以及只在 `x-vsr-warnings` 中被标记的回答，都不会存入缓存，因为缓存命中不携带守卫的响应头；带守卫说明的回答会连同说明一起存入。命中无需调用模型即可应答，因此守卫不会对它运行：

- 精确命中是同一个请求，守卫对它的判定仍然成立。
- 语义命中返回守卫为先前一个相似请求放行的回答。守卫中依赖请求本身的条件（例如 `keyword` 或 `fact_check` 条件）不会针对新请求求值，这些条件本应触发的规则对它不生效：为一个无需事实核查的请求放行的回答，会未经检查地返回给一个需要事实核查的相似请求。在这一点重要的场景下，请为该决策的缓存使用 `mode: exact`。

修改决策的守卫会让它的缓存重新开始，因为守卫的配置是缓存键的一部分。修改守卫读取的信号（例如越狱规则的 `predicate`）不会改变缓存键：在配置重载后仍保留条目的存储（例如 Redis）会继续返回旧值下放行的回答，直到它们过期。这类修改之后，请在每个守卫读取该信号的决策的 `response_cache` 配置中为 `revision.policy_revision` 设置新值。`revision` 中的每个值都是缓存键的一部分：

```yaml
plugins:
  - type: response_cache
    configuration:
      enabled: true
      mode: exact
      revision:
        policy_revision: guard-2
```

## 迁移 {#migration}

插件类型为 `response_cache`，存储为 `global.stores.response_cache`。仍写 `semantic-cache`、`semantic_cache`、`response-cache` 插件类型、`global.stores.semantic_cache`、扁平的 `similarity_threshold` 或 `semantic.similarity_threshold`、`allow_request_controls` 和 `control_header` 的配置会加载失败，并给出指明字段的一行错误；`vllm-sr config migrate` 把阈值写为 `semantic.match.recall.min_similarity`，把请求控制字段写入 `request_controls`。

来自模型运行时的嵌入（包括 Vela Embedding）更换模型、分词器、向量表示大小或推理设置后，会使用独立的缓存空间。租户命名空间和显式缓存版本保持不变；旧条目按原有过期时间保留，也可显式清理。升级模型后的首次请求会缓存未命中，使用相同向量表示重启则可复用兼容缓存。语义缓存需要本地分词器窗口，因此 Router 会拒绝为其使用[远程嵌入端点](../../model-runtime/guides/embeddings#use-an-external-embedding-service)。

## 运维 {#operations}

管理 API 在 `/api/v1/storage/response-cache/*` 下暴露经过脱敏的健康、能力、统计、候选配置测试、限定范围失效、基于 epoch 的清空。统一哈希链审计位于 `/api/v1/observability/audit`，需要 `audit.read` 权限。`/api/v1/plugins/response_cache` 提供插件发现与操作链接。失效默认是 dry-run。清空需要显式确认短语 `flush response cache`，并且永不调用后端范围的 `FLUSHALL`。

六种缓存后端在返回语义命中前，都会执行始终开启的英文词面校验。对于词面接近、但出现明确否定或已知反义词替换的问题，即使向量相似度较高，也会拒绝该候选。远端条目缺少原始问题时同样视为未命中。拒绝一个候选后，仍可使用后续已检索到的合格候选；远端检索保持候选数量上限。这项检查不保证识别仅词序变化、缺少词面线索或非英文的含义变化。

每个已返回的语义命中都会记录这项检查能否判断该问题对：`response_cache` 插件 span 上的 `cache.negation_guard`，以及 `cache_hit` 日志事件中的 `negation_guard`。`checked` 表示两个问题使用相同的词（顺序可以不同），或只在 `not`、`never` 等否定词上不同。`not_applicable` 表示还有其他词发生变化，否定词表无法判断，因此该命中只依据向量相似度。改写过的英文问题和大多数非英文命中都会报告 `not_applicable`，例如否定词表不能识别德语 `nicht` 或中文 `不`。

早期版本在内存后端提供的 NLI 层级已经退役；`vllm-sr config migrate` 会保留词面校验。见[存储与工具](../global/stores-and-tools.md#negation-guard)。

缓存响应可能包含用户或租户数据。请选择合适的范围、TTL、后端认证、加密和失效流程。语义阈值必须针对配置的嵌入模型校准。长于嵌入部署输入上限的查询不会被缓存，因为截断嵌入会匹配所有共享该前缀的查询。带个性化 RAG 或 memory 的路由，若没有显式策略，不应复用富化前的响应。完整示例见：
[`high-recall.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/response-cache/high-recall.yaml)
和
[`memory.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/response-cache/memory.yaml)。

---
is_mtpe: false
translation:
  source_commit: "bc5c6bfb747ffe9c5f35d98b203427dada8c4c56"
  source_file: "docs/tutorials/plugin/context.md"
  outdated: false
---

# Context

## 概述 {#overview}

`context` 是组装请求上下文的插件。它由三部分组成：

- **`sources`** 检索文档（`rag`）和用户的记忆（`memory`），各由自己的 `match` 块决定保留哪些。记忆源还会在响应之后把这一轮写入记忆。
- **`compress`** 压缩请求自身的片段，由它的 `match` 块判断哪些片段重要，直到请求放进 `budget`。它的 `signals` 压缩路由信号读取的文本。
- **`injection`** 决定保留下来的内容放在哪里。

## 主要优势 {#key-advantages}

- 一个插件按固定顺序负责检索、记忆和压缩。
- 每个阶段只调用一次 decision model：相关性判断决定 RAG 源保留哪些文档，也为压缩要保留的片段排序。
- 预算只有一个数：请求最多保留的输入 token 数，或模型上下文窗口减去预留的输出。
- 发给所选模型的对话只由 `compress` 改变；路由信号读取原始文本，除非 `compress.signals` 压缩它，jailbreak 和 PII 规则仍读取原文。

## 解决什么问题？ {#what-problem-does-it-solve}

文档、记忆、长历史和工具输出争用同一个上下文窗口。以前它们分在不同插件里检索，各有各的阈值、各自的 reranker 和各自对预算的理解。`context` 插件检索这些源，让 decision model 判断什么相关，再在同一个预算内压缩剩下的内容。

## 何时使用 {#when-to-use}

- 路由基于已入库的文档回答
- 路由用用户的记忆做个性化回答
- 路由的工具输出或历史比回答所需的大
- 路由要关闭记忆，使用 `enabled: false`

## 配置 {#configuration}

在 `recipes[].decisions[].plugins` 下添加插件：

```yaml
plugins:
  - type: context
    configuration:
      enabled: true
      sources:                          # 按顺序检索，每种类型最多一个
        - type: rag
          backend: milvus
          backend_config: {collection: docs, reuse_cache_connection: true}
          match:                        # 缺省值，显式写出
            recall: {method: embedding, top_k: 20, min_similarity: 0.3}
            judge: {preset: relevance, top_k: 6, min_score: 0.5}
          max_context_length: 10000     # 字符数
          on_failure: skip              # skip | warn | block
        - type: memory
          store: {auto: true}           # 响应之后写入这一轮
      compress:
        mode: auto                      # auto | always
        targets:
          tool_outputs: {mode: extractive, min_tokens: 2000, target_tokens: 1000}
        failure_mode: fail_open         # fail_open | fail_closed
      budget:
        max_tokens: auto                # 或输入 token 数
        reserve_output_tokens: auto
      injection:
        mode: tool_role                 # tool_role | system_prompt
```

启用的插件需要 `sources` 或 `compress`；`budget` 需要一个压缩请求的 `compress` 块。`enabled: false` 为该决策关闭上下文，包括全局记忆。决策运行 `global.plugins` 的 `context` 插件，并与配方和它自己的插件逐字段合并：它写出的 `sources` 替换继承来的源。

### RAG 源 {#rag-sources}

选择一个后端：

| 后端 | 用途 | 必需的后端字段 |
| --- | --- | --- |
| `milvus` | 直接从 Milvus collection 检索 | `collection`；可复用 response cache 的连接 |
| `qdrant` | 直接从 Qdrant collection 检索 | `collection`；可复用 response cache 的连接 |
| `external_api` | 自定义 HTTP 请求契约的服务 | `endpoint`、`request_format` |
| `mcp` | 在 MCP 工具调用器可用之前，启动时拒绝 | `server_name`、`tool_name` |
| `openai` | OpenAI 文件检索 | `vector_store_id`、`api_key` |
| `vectorstore` | Router 管理的向量存储服务，需要 `global.stores.vector_store.enabled: true` | `vector_store_id` |
| `hybrid` | 主后端加可选的回退后端 | `primary`，以及各后端自己的嵌套配置 |

- **`hybrid.strategy`** 缺省为 `sequential`：只有主后端失败或没有返回文档时才运行回退后端。`parallel` 同时发起两个查询，主后端一有文档就返回。
- **`max_response_bytes`** 限制每个 `external_api` 响应体，省略或为 `0` 时为 4 MiB。OpenAI `direct_search` 的每个向量存储检索响应也使用同样的缺省值。

源的 `match` 块决定保留哪些文档。缺省情况下：

1. **召回。**后端召回最多 20 个 embedding 相似度不低于 0.3 的文档。
2. **判断。**对回答判断的部署（除非 `match.judge.deployment` 指定其他部署，否则是 decision model）调用一次，对每个文档提出 `relevance` 问题，每个文档是批量请求中的一项。
3. **保留。**保留分数不低于 0.5 的前 6 个最相关文档，按判断给出的顺序排列。

写出的块整体替换缺省值：

- **只有 `recall` 的块**按后端的顺序保留召回的文档。
- **本地知识库**（基于内存向量库的 `vectorstore` 源）还可以用 `method: bm25` 召回，不对查询做嵌入；也可以由只有 `judge` 的块整体判断：judge 读取它最新的分块，最多 `max_candidates` 个（缺省 64）。
- **基于外部向量数据库的源**用 embedding 召回。

External API 的文档带有服务返回的分数（Pinecone `score`、Weaviate `1 - distance`、Elasticsearch `_score`、自定义 `score`）；没有分数的文档按相似度 1 计。

`on_failure` 决定检索失败时怎么做：`skip`（缺省）不带文档继续，`warn` 另外记录一条警告，`block` 以 `503` 拒绝请求。检索没有结果不算失败。`cache_results` 按查询、源和 match 块把保留的文档缓存 `cache_ttl_seconds`（缺省 3600）秒；judge 失败时在 `on_error: skip` 下保留的未经评判的文档只用于当次请求，不缓存。

:::caution
检索到的文档会作为上下文发给模型提供商。请在 collection 级别做访问控制，不要把多个租户混在一个不受限的检索范围里。相似度阈值与 embedding 模型相关。
:::

### 记忆源 {#memory-sources}

记忆源读取 `global.stores.memory` 下配置的存储：Milvus（缺省）、Valkey 或 Qdrant。存储配置见[存储和工具](../global/stores-and-tools)教程，另见 [Valkey 记忆部署指南](../../installation/valkey-memory)和 [Qdrant 部署指南](../../installation/qdrant)。

它的缺省 `match` 块只做召回：使用存储的 `default_retrieval_limit` 和 `default_similarity_threshold`，否则召回 5 条相似度不低于 0.7 的记忆。写出的块替换缺省值，加上 `judge` 就会让 decision model 判断哪些记忆相关。只有 `judge` 的块适合每个用户的小型记忆：judge 读取该用户最新的记忆，最多 `max_candidates` 条（缺省 64）。

- `store.auto` 在响应之后把这一轮写入记忆；未设置时跟随 `global.stores.memory.auto_store`。请求可以关闭写入；只有在 `store.auto` 未设置时，请求才能打开写入。
- `hybrid_search` 按 `hybrid_mode`（`weighted` 或 `rrf`）融合存储的向量分数和词法分数。
- `reflection` 为该决策覆盖存储的 reflection 门控。启用 reflection（缺省的启发式门控）时，检索到的记忆会被裁剪到 `max_inject_tokens`。门控用按文字系统区分的启发式估算长度：每个拉丁词约 1.3 个 token，每个汉字、平假名、片假名或谚文字符约 1.5 个 token，向上取整。

记忆失败从不拒绝请求：Router 在回放记录中记下失败并继续。

:::caution
记忆会持久化从请求派生的内容，并把检索到的记忆发给所选模型。请为这些数据选择合适的用户和租户隔离、保留策略、认证和传输安全。
:::

### 压缩和预算 {#compression-and-budget}

`compress` 压缩请求自身的片段：工具输出、历史，以及各个源注入的文档和记忆。每个片段都是关于最新一条用户消息的候选，由 `match` 块判断哪些重要：

- **缺省情况下**，BM25 先给片段排序，一次 `relevance` 调用为排在前面的 64 个片段打分（`judge.max_candidates`），每个片段是批量请求中的一项。
- **先压缩的：**最不相关的片段和 judge 没有读到的片段，直到请求放进预算。
- **只按词法。**`match: {recall: {method: bm25}}` 改为按词法排序，不调用模型。没有判断结果时，片段也按词法排序。

- `mode: auto`（缺省）压缩超出预算的请求，或有片段超过其目标 `min_tokens` 的请求；`always` 压缩每个请求。
- `targets` 设置每类片段的模式（`preserve`、`extractive`、`truncate`、`recoverable`）及其 `min_tokens` 和 `target_tokens`。`target_tokens` 必须小于 `min_tokens`。历史、RAG、记忆和当前用户文本在其目标启用之前保持不变。
- `budget.max_tokens` 是请求最多保留的输入 token 数：超过的请求被压缩到这个数。`auto`（缺省）是模型的上下文窗口减去 `budget.reserve_output_tokens`，后者缺省为请求的输出上限。预留可以加大，但不会小于请求的输出。
- `failure_mode: fail_open`（缺省）在改写失败时不压缩、照常发送请求；`fail_closed` 拒绝请求。两者都不绕过预算检查：受保护内容放不下的请求得到 `400` 和 `context_length_exceeded`。

- **估算方式。**预算使用保守的 UTF-8 字节估算，不是精确的 tokenizer 计数。
- **从不截断：**系统和开发者指令、安全文本、工具调用、schema、媒体、引用和 JSON 用户载荷。工具调用和结果的 ID 保持成对。
- **最新的用户消息。**使用 `current_user: {mode: truncate}` 时，最新的纯文本用户消息保留开头和结尾，中间用 `[... context omitted by route compression ...]` 分隔。中间部分是被省略，不是被总结。

JSON 字符串只通过其字符串叶子压缩，OpenAI 数组内容压缩文本块并保留图片，Anthropic `tool_result` 内容保留 `tool_use_id`、`is_error`、图片和 cache-control 元数据。`recoverable` 目标把原始内容存进共享的 Redis 或 Valkey 存储，注入保留的 `vsr_context_retrieve` 工具，并在进程内完成非流式的后续调用。恢复限定在请求和可信用户范围内，受 TTL、字节数和检索次数约束。

`request_controls` 允许调用方在 `x-vsr-compression-control` 请求头中发送 `bypass`（跳过压缩）或 `target=N`（工具输出目标，受 `max_target_tokens` 限制）；路由未启用时这些控制被忽略。

### 信号 {#signals}

`compress.signals` 在路由信号运行之前压缩它们读取的文本：保留其 profile 评分最高的句子，丢弃其余句子。信号在选出决策之前读取请求，所以 `signals` 从 `global.plugins` 的 context 插件读取，并与配方自己的 `context` 插件合并；决策的 `context` 插件不能包含它。

```yaml
global:
  plugins:
    - type: context
      configuration:
        enabled: true
        compress:
          signals:
            enabled: true
            max_tokens: 4096            # 超过它的文本被压缩到它
            min_length: 512             # 字符数；不超过这个长度的文本保持原样
            profile: default            # default | coding | medical | security | multi_turn
            skip_signals: [jailbreak, pii]
```

- 只有 `signals` 的 `compress` 块不压缩请求本身。要压缩请求的决策写出自己的 `compress` 字段，例如 `mode: auto`；它继承 `signals`，但不改变信号读取的内容。
- `profile` 设定评分权重和总是保留的首尾句子；`textrank_weight`、`position_weight`、`tfidf_weight`、`novelty_weight`、`position_depth`、`preserve_first_n` 和 `preserve_last_n` 覆盖它。`multi-turn` 按 `multi_turn` 读取；其他名称在加载时失败。
- `skip_signals` 列出读取原始文本的信号类型，缺省为 `jailbreak` 和 `pii`。
- 配方在自己的 `context` 插件中写 `signals: {enabled: false}` 来关闭它。

### 注入 {#injection}

`injection.mode: tool_role`（缺省）在最后一条用户消息之后，为每个源添加一个工具调用及其结果：文档用 `vsr_rag_context`，记忆用 `vsr_memory_context`，按源的顺序排列。`system_prompt` 改为把它们加入系统提示词。压缩会把两者都识别为检索到的上下文。

## 管理和预览 {#management-and-preview}

- `GET /api/v1/plugins/context/compression/capabilities`
- `GET /api/v1/plugins/context/compression/health`
- `GET /api/v1/observability/plugins/context/compression/stats`
- `POST /api/v1/plugins/context/compression/preview`
- `POST /api/v1/plugins/context/preview`（RAG 源的检索）
- `POST /api/v1/storage/context-recovery/invalidate`

压缩预览按词法为片段排序，只返回计划、目标索引、token 计数、分数、警告和跳过原因；它从不返回源内容或被省略的内容，并需要 `compression.preview` 权限。按范围使恢复失效需要 `compression.manage` 权限。

## 可观测性 {#observability}

| 内容 | 在哪里报告 |
| --- | --- |
| `compress.signals` | `llm_prompt_compression_total`（按 `recipe` 和 `outcome`），以及 `llm_routing_stage_duration_seconds` 的 `prompt_compression` 阶段 |
| 每个阶段的判断 | `vsr_match_judge_total{feature="rag"}`、`{feature="memory"}` 或 `{feature="compress"}` |
| 记忆检索 | `llm_memory_retrieval_total`，带 `backend` 和 `status` 标签（`hit`、`miss`、`error`） |
| RAG 后端和记忆结果，包括请求在失败后继续的记忆失败 | 该请求的[路由回放](./router-replay)记录 |

压缩指标报告是否运行、策略、前后计数和省略内容。这些是估算，不是实测的计费 token 节省，也从不包含原始的被省略内容或恢复键。

## 迁移 {#migrating}

`rag`、`memory` 和 `context_compression` 插件现在是 `context` 插件的源、`compress` 和 `budget`。仍使用它们或其已移除字段的配置会在加载时失败并给出一行提示，`vllm-sr config migrate` 会改写它：

- RAG 的 `top_k` 和 `similarity_threshold` 变为 `match.recall`，`rerank.top_k` 变为 `match.judge.top_k`：Vela cross-encoder reranker、它的 `rag.reranker` 绑定和 `provider: rerank` 部署已移除，由相关性判断保留文档。
- RAG 的 `min_confidence_threshold` 只在事实核查规则匹配时才检索：迁移会把一个 `fact_check` 条件以 AND 方式加入决策的规则。
- 记忆的 `retrieval_limit` 和 `similarity_threshold` 变为 `match.recall`，`auto_store` 变为 `store.auto`。
- 压缩的 `scoring` 已移除：`bm25` 变为 `match: {recall: {method: bm25}}`，embedding 打分由相关性判断取代。`budget.target_tokens` 变为 `budget.max_tokens`。
- `global.model_catalog.modules.prompt_compression` 变为 `global.plugins` 中 context 插件的 `compress.signals`。原本为空的决策 `compress` 块得到 `mode: auto`，使它继续压缩请求。

完整示例：[`config/fragments/plugin/context/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/plugin/context)。

---
is_mtpe: false
translation:
  source_commit: "b2db276cf1b5057c31f2ab2bddbd181e5692dbb6"
  source_file: "docs/tutorials/plugin/overview.md"
  outdated: true
---

# 插件

## 概览

插件在决策匹配后添加路由局部行为。它们可以改写请求、检索上下文、短路生成、检查响应，或控制保留哪些运维数据。

共享服务和存储属于 `global:`；决策插件只为一条路由启用并调优该行为。

## 主要优势

- 把行为附着在需要它的路由上。
- 复用共享存储和服务，而不重复其配置。
- 让请求改写、检索和响应检查可审计。

## 解决什么问题？

即使共享同一 Router，各路由也常常需要不同行为。插件把这些差异放在决策旁边，而不是藏进应用中间件或全局启用。

## 何时使用

当行为应仅在特定路由匹配后生效时，使用插件。当每条路由共享同一服务或后台存储时，改用 `global:`。插件条目位于 `recipes[].decisions[].plugins` 下。

## 配置

```yaml
recipes:
  - name: default
    decisions:
      - name: cached-support
        description: Reuse cached responses for support requests.
        priority: 100
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: support-model
        plugins:
          - type: response_cache
            configuration:
              enabled: true
              ttl_seconds: 3600
```

## 插件清单 {#plugin-inventory}

| 类型 | 目标 | 共享依赖 | 指南 |
|---|---|---|---|
| `fast_response` | 不调用模型即返回配置的响应 | 无 | [快速响应](./fast-response) |
| `system_prompt` | 插入或替换路由专用指令 | 无 | [系统提示词](./system-prompt) |
| `header_mutation` | 添加、更新或删除 Router 不拥有的发往提供商的请求头 | 无 | [请求头修改](./header-mutation) |
| `request_params` | 强制请求参数限制 | 无 | [请求参数](./request-params) |
| `tools` | 允许、阻止或移除工具和工具历史；由决策模型过滤或添加请求所需的工具 | `select` 需要决策模型；`add` 需要工具目录 | [工具](./tools) |
| `response_cache` | 复用兼容的先前响应 | `global.stores.response_cache` | [响应缓存](./response-cache) |
| `prompt_cache` | 在指令和工具上插入 Anthropic 提示词缓存断点 | 无 | [提示词缓存](./prompt-cache) |
| `context` | 检索文档和记忆，在预算内压缩请求，并在响应后写入记忆 | RAG 后端、`global.stores.memory`、相关性判断所用的 decision model | [Context](./context) |
| `router_replay` | 覆盖单条路由的回放采集，并把抽样副本影子发送到候选模型 | `global.services.router_replay` | [路由回放](./router-replay) |
| `response_guard` | 按检查回答所用的信号拦截或标注回答 | 其规则读取的响应阶段信号 | [响应守卫](./response-guard) |
| `masking` | 把命名的 `pii` 规则在发往提供商的请求中找到的个人数据替换为占位符 | 绑定到能回答 span 问题的部署的 `pii` 规则 | [Masking](./masking) |

[内容安全](./content-safety) 打包了三个受支持插件，而不是额外的插件类型。

## 相似度：`match` 块 {#match-block}

所有查找相似项的功能都使用同一个 `match` 块：

```yaml
match:
  recall: {method: embedding, top_k: 20, min_similarity: 0.3}                # 缩小候选范围：embedding | bm25
  judge:  {preset: relevance, top_k: 6, min_score: 0.5, max_candidates: 64}  # 用一道 System One 题决定
```

- **阶段。** 块只运行写出的阶段，至少一个；空块在加载时报 `match_empty`。功能没有写块时运行其缺省值（见下表）；写出的块整体替换缺省值。只有 `recall` 时就是经典的 embedding 或 BM25 相似度；只有 `judge` 时由 System One 题判断每个候选，适合天然有界的小候选集；两者都有时 `recall` 缩小范围、`judge` 决定。
- **`recall`** 按索引相似度缩小候选范围：
  - `method: embedding`（缺省）在存储的 embedding 空间中比较向量（存储没有指定自己的部署时用 `global.model_catalog.embeddings.default`），`min_similarity` 是 [-1, 1] 内的余弦相似度；`method: bm25` 使用词法索引，不需要模型，`min_similarity` 是 BM25 分数；
  - `top_k` 限定候选数量；候选来自索引时必须填写。
- **`judge`** 在一次调用中，由决策模型或指定的 `deployment` 回答一道关于请求和候选的 System One 题：
  - `preset: relevance | same_request | tool_rank`，或自定义 `question`（`choice`、`noul` 或 `score`）；
  - choice 题（`tool_rank`）在一个标准请求中从候选中选择，候选就是它的选项，每个候选的分数是它的概率；noul 或 score 题（`relevance`、`same_request`）对每个候选各问一次，每个候选是批量请求中的一项，分数是 P(yes) 或期望档位；
  - `top_k`、`min_score`（绝对值）和 `min_ratio`（相对最高分的比例，取值 (0, 1]）决定保留哪些候选，按分数从高到低排列；
  - judge 失败或超时时，`on_error: skip`（缺省）保留召回结果，没有召回时保持候选原样；`on_error: fail` 让该功能的匹配失败；`timeout_ms` 限定这次调用（未设置时取部署的 `timeout_ms`，再缺省为 1000 毫秒）。
- **`judge.max_candidates`** 是硬上限：choice judge 最多 255（一道题的选项数，也是它的缺省值），逐个候选提问的 judge 缺省 64。候选更多时从不失败：judge 按召回排名读取前 `max_candidates` 个，没有召回时按功能列出的顺序（最新的在前）读取。每次截断都会在请求的 trace span 上记一个 `match.judge.truncated` 事件，并计入 `vsr_match_judge_truncated_total`。

| 功能 | 缺省 `match` | 说明 |
|---|---|---|
| `response_cache.semantic.match` | `recall: {method: embedding, top_k: 3, min_similarity: 0.8}`，`judge: {preset: same_request, min_score: 0.8}` | judge 确认每个召回的命中；被否决的就是未命中。exact 模式保留 |
| `context.sources[rag].match` | `recall: {method: embedding, top_k: 20, min_similarity: 0.3}`，`judge: {preset: relevance, top_k: 6, min_score: 0.5}` | `recall: {method: bm25}` 和没有召回的 judge 读取本地知识库：基于内存向量库的 `vectorstore` 源。基于外部向量数据库的源用 `embedding` 召回 |
| `context.sources[memory].match` | `recall: {method: embedding, top_k: 5, min_similarity: 0.7}` | 记忆存储的 `default_retrieval_limit` 和 `default_similarity_threshold` 替代召回缺省值。没有召回的 judge 读取用户最新的记忆 |
| `context.compress.match` | `recall: {method: bm25}`，`judge: {preset: relevance}` | 请求的每个片段都是关于最新一条用户消息的候选；BM25 先为 judge 的 64 个名额预筛；保留多少由预算决定 |
| `tools.select.match` | `judge: {preset: tool_rank, top_k: 3, min_ratio: 0.1}` | 最多 255 个工具；更多的会被截断，或由 `recall: {method: bm25}` 缩小；没有 judge 的块在加载时失败 |

每个阶段都计入 `vsr_match_recall_total`（feature、method、outcome）或 `vsr_match_judge_total`（feature、preset、outcome），并在 `vsr_match_stage_duration_seconds` 中计时。

## 运维边界 {#operational-boundaries}

- 当多个插件改写绑定提供商的请求或响应时，它们会相互作用。Router 管道固定其执行顺序；在 YAML 中重排条目不会改变该顺序。
- 检索、memory、缓存和回放可能持久化从请求派生的内容。为所选后端配置保留策略、租户/用户范围、认证和加密。
- 请求头和提示词改写可能跨越信任边界。不要把不受信任的调用方元数据复制到特权请求头或系统指令中。
- 部署前校验完整配方，以便不支持的插件名或不兼容设置在流量到达 Router 之前失败。

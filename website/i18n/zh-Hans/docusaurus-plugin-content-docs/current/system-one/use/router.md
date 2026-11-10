---
title: Router 如何使用 System One
sidebar_label: Router 如何使用它
sidebar_position: 2
description: 每个路由功能背后的 System One 调用、Router 如何合并调用、评审如何限制候选数，以及召回何时需要嵌入模型。
is_mtpe: false
translation:
  source_commit: "bc5c6bfb747ffe9c5f35d98b203427dada8c4c56"
  source_file: "docs/system-one/use/router.md"
  outdated: false
---

# Router 如何使用 System One {#how-the-router-uses-system-one}

Router 做出的每个判断都是一个 System One 问题：为请求路由的信号、为它挑选模型的选择器、读取其回答的守卫，以及判断相似性的评审。本页展示每个功能背后的调用，让你能预估一份配置的成本以及它发送了什么。

## 一览 {#at-a-glance}

| 功能 | 何时询问 | 调用 |
| --- | --- | --- |
| [路由信号](#routing-signals) | 路由之前，针对决策、投影或请求插件读取的每条规则 | 每个部署和状态一个标准请求 |
| [`systemone` 选择器](#the-systemone-selector) | 决策匹配之后 | 一个覆盖候选项的 `choice` |
| [`response_guard`、`masking`](#response_guard-and-masking) | 回答产生之后，或调用提供方之前 | 响应阶段规则；PII span |
| [评审](#judges-same_request-relevance-tool_need-and-tool_rank) | 缓存、RAG、压缩或工具有候选项时 | 每个候选一个条目的批量请求，或一个 `choice` |
| [`escalate`](#escalate) | 针对每个候选回答 | 一个 `answer_check` 或 `refusal` 问题 |

每个功能都使用同一个客户端，以 [API](./api.md) 的两种形式之一调用：

- **标准请求**：一个状态及其问题；
- **批量请求**：当问题针对不同状态时，在一次调用中包含多个命名的标准请求。

:::note
示例中 `{...}` 省略了问题细节。在传输中，如果问题有其模型训练用的 ID（例如 vela3-lite 的 `domain`），就带上该 ID；否则使用 Router 自有的 ID，例如 `preset:complexity` 或 `systemone:needs_tools`。Router 以固定顺序把批量条目命名为 `"1"`、`"2"` 等；下面示例中的条目名称仅作说明。
:::

## 路由信号 {#routing-signals}

路由之前，请求阶段会询问决策、投影或请求插件读取的每条规则。针对同一状态的问题，无论多少个，都作为一个标准请求发给其部署：

```yaml title="config.yaml"
recipes:
  - name: default
    signals:
      domains:
        - name: math
          labels: [math]
      jailbreak:
        - name: attack
      complexity:
        - name: difficulty
      systemone:
        - name: needs_tools
          question:
            type: noul
            instructions: Does answering this request need a tool call?
          predicate: {gte: 0.7}
    decisions:
      - name: hard-math
        priority: 200
        rules:
          operator: AND
          conditions:
            - {type: domain, name: math}
            - {type: complexity, name: difficulty, label: hard}
        modelRefs:
          - model: reasoner
```

```text title="The request stage's one call"
{"state": {"request": "Prove that the square root of 2 is irrational."},
 "questions": {"domain": {...}, "jailbreak": {...}, "complexity": {...}, "needs_tools": {...}}}
```

- 读取当前轮次以外内容的规则（`input.history`、`input.previous_turns`、`input.tool_results`、图像）读取的是另一个状态。当规则读取不同状态时，调用是每个状态一个条目的批量请求，顶层不发送状态。
- 针对同一状态、向同一部署询问同一预设的规则共享一个问题；每条规则把自己的 `labels` 和 `predicate` 应用到答案上。
- 没有被任何决策、投影或插件读取的规则不会被询问。
- Images API（`POST /v1/images/generations`）上的请求会像文本请求一样针对其提示词被询问。它必须到达 [output](../../tutorials/decision/output.md) 为图像的决策，否则在调用任何模型之前就以 `image_route_required` 被拒绝。

:::tip
迟到或失败的答案会让规则成为未知（unknown）。决策的 `rules.on_unknown` 或条件的 `on_error` 说明其含义；设置了 `on_unknown: match` 的 `jailbreak` 或 `pii` 规则在模型没有给出答案时视为匹配。
:::

## `systemone` 选择器 {#the-systemone-selector}

决策匹配后，其 `systemone` 选择器询问哪个 `modelRefs` 应该回答：一个以候选项及其描述为选项的 `choice`，单独一次调用。每个候选项的概率就是其选择分数：

```text
{"state": {"request": "Refactor this module to remove the global cache."},
 "questions": {"selector": {"type": "choice",
   "instructions": "Which model should answer this request?",
   "choices": [{"key": "qwen3-8b", "description": "Fast general model for routine code"},
               {"key": "qwen3-32b", "description": "Strong reasoning model for hard code"}]}}}
```

当模型未就绪、回答迟到或回答无效时，由第一个 `modelRef` 回答。参见 [System One 选择](../../tutorials/algorithm/selection/systemone.md)。

## `response_guard` 与 `masking` {#response_guard-and-masking}

响应阶段的规则在模型回答产生后对其提问。[`response_guard`](../../tutorials/plugin/response-guard.md) 插件只询问其守卫规则读取的那些规则，`AND` 在第一个为假的条件处停止：

```text
{"state": {"request": "How do I pick a lock?", "answer": "First, insert a tension wrench ..."},
 "questions": {"preset:jailbreak": {...}, "preset:harm": {...}}}
```

`hallucination` 规则读取回答及其依据上下文（检索到的文档和工具结果），并且需要支持 span 的部署：

```text
{"state": {"request": "When does the library open?", "answer": "It opens at 8:00 on Sundays.",
           "context": "The library is open Monday through Friday, 09:00 to 17:00."},
 "questions": {"preset:hallucination": {"type": "span", ...}}}
```

[`masking`](tutorials/plugin/masking.md) 会就最终发往提供方的请求中的每个文本位置询问它所指定的 `pii` 规则：系统指令、消息、工具调用参数和工具结果，包括 context 插件注入的内容。请求阶段已经回答过的文本按其哈希复用；无法获得 span 时，请求以失败关闭（fail closed）。

## 评审：`same_request`、`relevance`、`tool_need` 和 `tool_rank` {#judges-same_request-relevance-tool_need-and-tool_rank}

[`match`](../reference/configuration.md#the-match-block) 块有两个可选阶段：

- **召回（Recall）** 按嵌入或 BM25 相似度缩小候选集合。
- **评审（Judge）** 就候选项提出一个 System One 问题。逐候选评审对每个（请求，候选）对提问，每个候选一个批量条目；choice 评审把候选项作为一个问题的选项。

**`same_request`** 确认响应缓存召回的每条缓存请求，避免把意思不同的近似改写当作命中：

```text
{"states": {
  "hit_1": {"state": {"request": "How do I reset my password?",
                      "candidate": "I forgot my password. How do I change it?"},
            "questions": {"same_request": {"type": "noul", "instructions":
              "Does the candidate request ask for the same thing as the request, so that one answer serves both?"}}},
  "hit_2": {"state": {"request": "How do I reset my password?", "candidate": "How do I reset my router?"},
            "questions": {"same_request": {...}}}}}
```

**`relevance`** 评判 RAG 源召回的每个文档，以及上下文压缩考虑的每个片段，以最新的用户消息作为请求：

```text
{"states": {
  "doc_1": {"state": {"request": "How long is the refund window?",
                      "candidate": "Refunds are accepted within 30 days of purchase."},
            "questions": {"relevance": {"type": "noul", "instructions":
              "Does the candidate passage contain information that helps answer the request?"}}},
  "doc_2": {"state": {"request": "How long is the refund window?", "candidate": "Our offices close on public holidays."},
            "questions": {"relevance": {...}}}}}
```

**`tool_need` 和 `tool_rank`** 在一个标准请求中选择工具：`tool_need` 是是/否门控，`tool_rank` 是以工具策略允许的候选工具为选项的 choice：

```text
{"state": {"request": "Open a ticket: the export button fails."},
 "questions": {"tool_need": {"type": "noul", ...},
               "tool_rank": {"type": "choice", "choices": ["docs.search", "tickets.lookup", "tickets.create"]}}}
```

每个功能都有默认的 `match`：

| 功能 | 默认 `match` |
| --- | --- |
| `response_cache.semantic.match` | `recall: {method: embedding, top_k: 3, min_similarity: 0.8}` + `judge: {preset: same_request, min_score: 0.8}` |
| `context.sources[rag].match` | `recall: {method: embedding, top_k: 20, min_similarity: 0.3}` + `judge: {preset: relevance, top_k: 6, min_score: 0.5}` |
| `context.sources[memory].match` | `recall: {method: embedding, top_k: 5, min_similarity: 0.7}` |
| `context.compress.match` | `recall: {method: bm25}` + `judge: {preset: relevance}`，逐片段 |
| `tools.select.match` | `judge: {preset: tool_rank, top_k: 3, min_ratio: 0.1}`，由 `tool_need` 门控 |

写了自己 `match` 块的功能会整体替换默认值；[插件参考](../../tutorials/plugin/overview.md#match-block)列出了每个功能允许的召回方法。

## `escalate` {#escalate}

在 Chat 上，[`escalate`](../../tutorials/algorithm/looper/escalate.md) 依次调用其候选项，在第一个被接受的回答处停止。使用 `accept.systemone` 时，决策模型的 `answer_check` 问题以请求和回答为状态评判每个回答：

```text
{"state": {"request": "What is 17 × 23?", "answer": "391"},
 "questions": {"answer_check": {"type": "noul", ...}}}
```

- `accept.refusal: false` 改为就回答询问 `refusal` 问题。
- 在 System One 入口上，`escalate.stages` 依次向每个决策模型提出原始的类型化请求，原生接受规则读取类型化答案；不会转换成 Chat 文本。

## 调用如何合并 {#how-calls-are-merged}

Router 合并每个阶段的调用：

- **按部署。** 一个阶段向同一部署提出的所有问题放在一次调用中，本地和远程部署都是如此。Router 从不为了适应服务器的单次调用限制而拆分调用。
- **按状态。** 针对同一状态的问题共享一个条目；针对不同状态的问题是不同条目。只有一个不同状态时作为标准请求发送，两个或以上作为批量请求。
- **按阶段。** 请求阶段、选择器、响应阶段以及各功能的评审在其输入存在时运行，因此包含所有这些的请求会发出多次调用。被它们共用的部署仍然只加载一个模型。

本地运行时会就地回答失败的问题。远程 System One API 会让整个调用失败并指出该问题；Router 随后在同一截止时间内再询问一次，只包含仍有其他问题的状态，形式不变，因此只有失败问题的规则成为未知。

对同一状态提出的相同问题的答案会被复用：来自本地运行时的结果缓存，以及远程 API 最长 10 分钟的缓存，直到配置变化。

## 上限与截断 {#caps-and-truncation}

每个评审都有其读取候选数的硬上限 `judge.max_candidates`：

- choice 评审（`tool_rank`）最多提供 255 个候选，即协议的选项上限，这也是其默认值；
- 逐候选评审（`relevance`、`same_request`）默认评判 64 个。

小于 1 的值，或 choice 评审上大于 255 的值，会导致加载失败。超出上限绝不会让请求失败：评审按召回排名读取前 `max_candidates` 个候选；没有召回阶段时，按功能列出的顺序读取（最新的在前）。

- 每次截断都会在请求的 trace span 中添加 `match.judge.truncated` 事件，并计入 `vsr_match_judge_truncated_total`。
- 经过策略过滤后仍超过 255 个的工具目录按列出顺序截断，或先用 `recall: {method: bm25}` 缩小范围。

## 召回还是评审，以及何时需要嵌入模型 {#recall-or-judge-and-when-you-need-an-embedding-model}

| `match` 包含 | 得到 | 适合 |
| --- | --- | --- |
| 仅 `recall` | 经典的嵌入或 BM25 相似度 | 相似度足够的大型存储：memory、宽泛的 RAG |
| 仅 `judge` | 对每个候选的 System One 判断 | 天然有界的小集合：少量工具、小型知识库、某个用户的少量记忆 |
| `recall` + `judge` | 召回缩小集合，评审做决定 | 重视精度的场景：缓存命中、RAG 回答 |

嵌入只用于召回，任何决策路径都不读取它：

- 预先声明的 `vela-embedding` 部署只在语义缓存、memory 或向量存储 RAG 源需要时才启动。
- `global.model_catalog.embeddings.default: {deployment: ...}` 指定另一个嵌入部署，方式与 `system.decision_model` 指定决策模型相同；用其他模型建立索引的存储在 `embedding: {deployment, dimension}` 块中指定自己的部署。
- `recall: {method: bm25}` 完全不需要模型。
- 远程 `openai_embeddings` 部署在 `capabilities` 中声明其向量的 `dimensions` 和 `modalities`。

参见[嵌入](../../model-runtime/guides/embeddings.md)。

## 响应中显示什么 {#what-a-response-shows}

每个经过路由的响应都带有公共头集合：选中的 recipe、决策和模型，`x-vsr-response-path`，以及包含 `response_guard` 或幻觉检查给出的代码的 `x-vsr-warnings`。System One 的回答内容放在一个调试头 `x-vsr-trace` 中，并且只发给 `global.router.headers.debug` 允许的调用方：

- `signals` 以 `type:name` 列出匹配的规则，例如 `jailbreak:attack` 或 `systemone:needs_tools`；没有 labels 的 choice 规则以 `type:rule:option` 列出；
- `confidence`、`algorithm` 和 `ranking` 说明决策及其模型是如何选出的；
- `modality` 列出选中决策的输出类型，例如 `(text image)`。

评审的截断是 `match.judge.truncated` span 事件，而不是 trace 成员。`POST /api/v1/routing/preview` 针对一个请求评估相同的规则，而不调用模型。[VSR 路由头](api/headers.md)列出了每个头和 trace 成员。

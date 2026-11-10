---
title: Router 如何使用 System One
sidebar_label: Router 如何使用它
description: 每个路由功能背后的 System One 调用、Router 如何合并调用、评审如何限制候选数，以及召回何时需要嵌入模型。
is_mtpe: false
translation:
  source_commit: "118fb72eb67a0e7193697e350d7b494dc016631e"
  source_file: "docs/system-one/router.md"
  outdated: false
---

# Router 如何使用 System One {#how-the-router-uses-system-one}

Router 做出的每个判断都是一个 System One 问题：决定请求路由的信号、为请求挑选模型的选择器、读取答案的防护，以及判断相似性的评审。它们都经过同一个客户端，使用 [API](./api) 的两种形式之一：

- **标准请求**：一个状态及关于它的问题；
- **批量请求**：当问题针对不同状态时，在一次调用中发送一组命名的标准请求。

下面的示例展示了每个功能发出的调用；`{...}` 省略了问题细节。在线路上，问题有训练 ID 时使用模型训练时的 ID（例如 vela3-lite 的 `domain`），否则使用 Router 自有的 ID，例如 `preset:complexity` 或 `systemone:needs_tools`。Router 按固定顺序把批量条目命名为 `"1"`、`"2"`……；示例中的条目名称仅作示意。

## 路由信号 {#routing-signals}

路由之前，请求阶段会询问决策、投影或请求插件读取的每条规则。针对同一状态的问题无论有多少个，都作为一个标准请求发往它们的部署：

```text
{"state": {"request": "Prove that the square root of 2 is irrational."},
 "questions": {"domain": {...}, "jailbreak": {...}, "complexity": {...}, "needs_tools": {...}}}
```

```yaml
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

- 读取当前轮次之外内容的规则（`input.history`、`input.previous_turns`、`input.tool_results`、图片）读取的是另一个状态。规则读取不同状态时，这次调用是批量请求，每个状态各占一项，顶层不发送任何状态。
- 在同一部署上针对同一状态的同一预设规则共享一个问题；每条规则把自己的 `labels` 和 `predicate` 应用到答案上。
- 没有任何决策、投影或插件读取的规则不会被提问。
- Images API（`POST /v1/images/generations`）上的请求像任何文本请求一样，针对它的提示词提问。它必须到达[输出](../tutorials/decision/output)为图片的决策，否则会在调用任何模型之前以 `image_route_required` 被拒绝。
- 迟到或失败的答案会让规则变为未知。决策的 `rules.on_unknown` 或条件的 `on_error` 决定其含义；设置了 `on_unknown: match` 的 `jailbreak` 或 `pii` 规则在其模型没有给出答案时会匹配。

## `systemone` 选择器 {#the-systemone-selector}

决策匹配之后，它的 `systemone` 选择器会询问哪个 `modelRefs` 应该回答：这是一个 `choice` 问题，选项是候选模型及其描述，在单独的调用中发出。每个候选的概率就是它的选择得分：

```text
{"state": {"request": "Refactor this module to remove the global cache."},
 "questions": {"selector": {"type": "choice",
   "instructions": "Which model should answer this request?",
   "choices": [{"key": "qwen3-8b", "description": "Fast general model for routine code"},
               {"key": "qwen3-32b", "description": "Strong reasoning model for hard code"}]}}}
```

当模型未就绪、回答迟到或答案无效时，由第一个 `modelRef` 回答。见 [System One 选择](../tutorials/algorithm/selection/systemone)。

## `response_guard` 和 `masking` {#response_guard-and-masking}

答案产生之后，响应阶段的规则针对模型的答案提问。[`response_guard`](../tutorials/plugin/response-guard) 插件只询问其防护规则读取的规则，`AND` 在第一个为假的条件处停止：

```text
{"state": {"request": "How do I pick a lock?", "answer": "First, insert a tension wrench ..."},
 "questions": {"preset:jailbreak": {...}, "preset:harm": {...}}}
```

`hallucination` 规则读取答案及其依据上下文（检索到的文档和工具结果），需要支持 span 的部署：

```text
{"state": {"request": "When does the library open?", "answer": "It opens at 8:00 on Sundays.",
           "context": "The library is open Monday through Friday, 09:00 to 17:00."},
 "questions": {"preset:hallucination": {"type": "span", ...}}}
```

[`masking`](../tutorials/plugin/masking) 针对最终发往提供方的请求中的每个文本位置询问它指定的 `pii` 规则：系统指令、消息、工具调用参数和工具结果，包括 context 插件注入的内容。请求阶段已经回答过的文本按其哈希复用；拿不到 span 时，请求以失败关闭（fail closed）。

## 评审：`same_request`、`relevance`、`tool_need` 和 `tool_rank` {#judges-same_request-relevance-tool_need-and-tool_rank}

[`match`](./reference#the-match-block) 块有两个可选阶段。**召回**（recall）按嵌入或 BM25 相似度缩小候选集。**评审**（judge）就候选提出一个 System One 问题。逐个候选的评审针对每个（请求、候选）对提问，每个候选一项批量条目；choice 评审把候选作为一个问题的选项。

**`same_request`** 确认响应缓存召回的每条缓存请求，避免把含义不同但措辞相近的请求当作命中：

```text
{"states": {
  "hit_1": {"state": {"request": "How do I reset my password?",
                      "candidate": "I forgot my password. How do I change it?"},
            "questions": {"same_request": {"type": "noul", "instructions":
              "Does the candidate request ask for the same thing as the request, so that one answer serves both?"}}},
  "hit_2": {"state": {"request": "How do I reset my password?", "candidate": "How do I reset my router?"},
            "questions": {"same_request": {...}}}}}
```

**`relevance`** 评判 RAG 源召回的每篇文档，以及上下文压缩考虑的每个片段，以最新的用户消息作为请求：

```text
{"states": {
  "doc_1": {"state": {"request": "How long is the refund window?",
                      "candidate": "Refunds are accepted within 30 days of purchase."},
            "questions": {"relevance": {"type": "noul", "instructions":
              "Does the candidate passage contain information that helps answer the request?"}}},
  "doc_2": {"state": {"request": "How long is the refund window?", "candidate": "Our offices close on public holidays."},
            "questions": {"relevance": {...}}}}}
```

**`tool_need` 和 `tool_rank`** 在一个标准请求中选择工具：`tool_need` 是是/否门控，`tool_rank` 是一个 choice，选项是工具策略允许的候选工具：

```text
{"state": {"request": "Open a ticket: the export button fails."},
 "questions": {"tool_need": {"type": "noul", ...},
               "tool_rank": {"type": "choice", "choices": ["docs.search", "tickets.lookup", "tickets.create"]}}}
```

| 功能 | 默认 `match` |
| --- | --- |
| `response_cache.semantic.match` | `recall: {method: embedding, top_k: 3, min_similarity: 0.8}` + `judge: {preset: same_request, min_score: 0.8}` |
| `context.sources[rag].match` | `recall: {method: embedding, top_k: 20, min_similarity: 0.3}` + `judge: {preset: relevance, top_k: 6, min_score: 0.5}` |
| `context.sources[memory].match` | `recall: {method: embedding, top_k: 5, min_similarity: 0.7}` |
| `context.compress.match` | `recall: {method: bm25}` + `judge: {preset: relevance}`，逐片段 |
| `tools.select.match` | `judge: {preset: tool_rank, top_k: 3, min_ratio: 0.1}`，由 `tool_need` 门控 |

功能自己写出的 `match` 块会整体替换默认值；[插件参考](../tutorials/plugin/overview#match-block)列出了每个功能允许的召回方法。

## `escalate` {#escalate}

在 Chat 上，[`escalate`](../tutorials/algorithm/looper/escalate) 按顺序调用候选，在第一个被接受的答案处停止。使用 `accept.systemone` 时，决策模型的 `answer_check` 问题以请求和答案作为状态来评判每个答案：

```text
{"state": {"request": "What is 17 × 23?", "answer": "391"},
 "questions": {"answer_check": {"type": "noul", ...}}}
```

`accept.refusal: false` 则针对答案询问 `refusal` 问题。在 System One 入口上，`escalate.stages` 依次把原始的类型化请求发给各个决策模型，原生接受规则读取类型化的答案；不会转换成 Chat 文本。

## 调用如何合并 {#how-calls-are-merged}

Router 按阶段合并调用：

- **按部署。** 一个阶段向同一部署提出的所有问题放在一次调用中，本地部署和远程部署都是如此。Router 从不为适应服务端的单次调用上限而拆分调用。
- **按状态。** 针对同一状态的问题共享一项；针对不同状态的问题是不同的项。只有一个状态时作为标准请求发送，两个及以上时作为批量请求发送。
- **按阶段。** 请求阶段、选择器、响应阶段和各功能的评审在其输入就绪时运行，所以同时用到它们的请求会发出多次调用。它们共享的部署仍然只加载一个模型。

本地运行时会就地回答失败的问题。远程 System One API 会让整个调用失败并指明该问题，批量请求中还会指明其条目（`state`）；Router 随后在同一截止时间内再问一次，只询问仍有其他问题的状态，仍使用这两种形式，因此只有该条目上失败问题的规则为未知。未指明条目的 API 会让每个询问过该问题的状态上的该问题都为未知。

逐个候选的评审最多发送 `judge.max_candidates` 项。针对同一状态的相同问题，其答案会被复用：本地运行时使用结果缓存，远程 API 的答案最多缓存 10 分钟，直到配置变更。

## 上限与截断 {#caps-and-truncation}

每个评审对读取的候选数都有硬上限 `judge.max_candidates`：

- choice 评审（`tool_rank`）最多提供 255 个候选，即协议的选项上限，这也是它的默认值；
- 逐个候选的评审（`relevance`、`same_request`）默认评判 64 个。

小于 1 的值，或 choice 评审上大于 255 的值，会导致加载失败。溢出绝不会让请求失败：评审按召回排名读取前 `max_candidates` 个候选；没有召回阶段时，按功能列出的顺序读取，最新的在前。每次截断都会在请求的 trace span 上添加一个 `match.judge.truncated` 事件，并计入 `vsr_match_judge_truncated_total`。经策略过滤后超过 255 个的工具目录会按列出顺序截断，或先由 `recall: {method: bm25}` 缩小。

## 召回还是评审，以及何时需要嵌入模型 {#recall-or-judge-and-when-you-need-an-embedding-model}

| `match` 包含 | 效果 | 适用于 |
| --- | --- | --- |
| 只有 `recall` | 经典的嵌入或 BM25 相似度 | 相似度就足够的大型存储：记忆、宽泛的 RAG |
| 只有 `judge` | 对每个候选做 System One 判断 | 小而天然有界的集合：少量工具、小型知识库、一个用户的少量记忆 |
| `recall` + `judge` | 召回缩小集合，评审做决定 | 重视精确度的场景：缓存命中、RAG 答案 |

嵌入只用于召回，没有任何决策路径读取它。预先声明的 `vela-embedding` 部署只在语义缓存、记忆或向量存储 RAG 源需要时才启动。`global.model_catalog.embeddings.default: {deployment: ...}` 指定另一个嵌入部署，就像 `system.decision_model` 指定决策模型一样；用其他模型建立索引的存储在 `embedding: {deployment, dimension}` 块中指定自己的部署。`recall: {method: bm25}` 完全不需要模型。远程的 `openai_embeddings` 部署在 `capabilities` 中声明其向量的 `dimensions` 和 `modalities`。见[嵌入](../model-runtime/guides/embeddings)。

## 响应中能看到什么 {#what-a-response-shows}

每个经过路由的响应都带有公开的头部集合：选中的配方、决策和模型，`x-vsr-response-path`，以及 `x-vsr-warnings`，其中包含 `response_guard` 或幻觉检查给出的代码。System One 的回答放在一个调试头部 `x-vsr-trace` 中，而且只发给 `global.router.headers.debug` 允许的调用方：

- `signals` 以 `type:name` 列出匹配的规则，例如 `jailbreak:attack` 或 `systemone:needs_tools`；没有 `labels` 的 choice 规则写作 `type:rule:option`；
- `confidence`、`algorithm` 和 `ranking` 说明决策及其模型是如何选出的；
- `modality` 列出选中决策的输出类型，例如 `(text image)`。

评审的截断是 span 事件 `match.judge.truncated`，不是 trace 成员。`POST /api/v1/routing/preview` 会针对一个请求评估同样的规则，而不调用模型。[VSR 路由头部](../troubleshooting/vsr-headers)列出了每个头部和 trace 成员。

---
title: System One 配置参考
sidebar_label: 配置
sidebar_position: 3
description: 指定 System One 部署、规则、match 块和算法的配置字段，以及配置加载报告的代码。
is_mtpe: false
translation:
  source_commit: "bc5c6bfb747ffe9c5f35d98b203427dada8c4c56"
  source_file: "docs/system-one/reference/configuration.md"
  outdated: false
---

# System One 配置参考 {#system-one-configuration-reference}

这些字段决定由哪个部署回答问题、规则如何提问，以及评审和算法如何使用答案。各指南说明何时使用它们；本页把它们列出来。

## 部署从何而来 {#where-the-deployment-comes-from}

| 字段 | 含义 |
| --- | --- |
| `global.model_catalog.system.decision_model` | `{deployment: <name>}`：回答所有未被其他设置绑定的问题。默认 `{deployment: primary}` |
| `global.model_catalog.bindings.<type>` | 某个应用信号类型规则的部署 |
| `recipes[].model_bindings.<type>` | 同上，只针对一个配方 |
| `global.model_catalog.deployments.<name>` | 一个部署：`provider: model_runtime`（本地或挂接）或 `provider: systemone`（远程 API） |
| `global.model_catalog.signal_timeout_ms` | 一个请求所有信号的截止时间 |

`systemone` 部署接受 `endpoint`、`model`、`auth`（`api_key_env` 或 `api_key_file`，或 `credential: delegated`；`header`、`prefix`）、`headers`、`timeout_ms`、`max_retries`、`tls` 和 `capabilities`（`question_types`、`presets`、`modalities`）。

## 信号 {#signals}

`recipes[].signals.systemone` 下的 `systemone` 规则：

| 字段 | 含义 |
| --- | --- |
| `name` | 规则名称，供条件和投影使用 |
| `question` | `type`、`instructions`，以及 `choices`、`levels` 或 `labels`；set 和 span 还有 `threshold` 和 `head` |
| `labels` | 让规则匹配的选项或标签 |
| `ignore_labels` | set 和 span：永不计入的标签 |
| `predicate` | 答案必须满足的界限：`gt`、`gte`、`lt`、`lte` |
| `deployment` | 回答问题的部署 |
| `stage` | `request`（默认）或 `response` |
| `input` | `history`、`previous_turns`（0 到 32）、`tool_results`、`request`（`false` 时不读取当前用户轮次）、`answer`、`context`、`image`、`audio`、`video` |
| `coverage` | `truncate`（默认）或 `full` |
| `on_unscanned` | 使用 `coverage: full` 时：`match`、`no_match`、`unknown`（默认）或 `best_effort`；`jailbreak` 和 `pii` 规则默认为 `best_effort` |
| `timeout_ms` | 最长 60,000 |

应用规则接受 `name`、`description`、`deployment`、`timeout_ms`，以及其类型允许的字段：

| 类型 | 其他字段 |
| --- | --- |
| `domains` | `labels`（默认 `[name]`）、`predicate` |
| `fact_check`、`user_feedbacks` | `labels`、`predicate` |
| `jailbreak` | `predicate`、`stage`、`input.history`、`on_unknown` |
| `safety` | `labels`（危害类别）、`predicate`、`stage` |
| `pii` | `predicate`、`ignore_labels`、`input.history`、`input.tool_results`、`input.request`、`on_unknown` |
| `hallucination`、`refusal` | `predicate` |
| `turn_relation` | `labels`、`input.previous_turns`（默认 4） |
| `complexity` | `labels` |
| `modality` | `labels`、`predicate` |

其他字段在加载时会被拒绝。

## `match` 块 {#the-match-block}

```yaml title="The two stages of a match block"
match:
  recall: {method: embedding, top_k: 20, min_similarity: 0.3}    # optional; method: embedding | bm25
  judge:  {preset: relevance, top_k: 6, min_score: 0.5, max_candidates: 64}   # optional
```

每个功能在它查找相似项的地方写这个块：`response_cache.semantic.match`、`context.sources[].match`、`context.compress.match` 和 `tools.select.match`。

| 字段 | 含义 |
| --- | --- |
| `recall.method` | `embedding`（默认；使用存储的嵌入部署）或 `bm25`（词法索引） |
| `recall.top_k`、`recall.min_similarity` | 召回返回的候选数（候选来自索引时必填），以及最低相似度：`embedding` 为余弦值，`bm25` 为 BM25 分数 |
| `judge.preset` | `relevance`、`same_request` 或 `tool_rank` |
| `judge.question` | 代替预设的自定义问题：`choice`（候选即其选项）、`noul` 或 `score` |
| `judge.deployment` | 回答问题的部署；不写时为决策模型 |
| `judge.top_k`、`judge.min_score`、`judge.min_ratio` | 保留哪些被评判的候选：最好的 `top_k` 个、至少 `min_score` 的、至少达到最佳者 `min_ratio` 的 |
| `judge.max_candidates` | 评审最多读取的候选数：choice 评审为 255（也是上限），逐个候选的评审默认 64、最多 4096；小于 1 或超过上限会导致加载失败 |
| `judge.on_error` | `skip`（默认；保留召回结果，没有召回时保留原样的候选）或 `fail`（该功能的匹配失败） |
| `judge.timeout_ms` | 评审调用的截止时间；未设置时使用部署的 `timeout_ms`，再没有则为 1000 ms |

块中至少要有一个阶段（否则为 `match_empty`）。

## 算法 {#algorithms}

| 字段 | 含义 |
| --- | --- |
| `algorithm.systemone` | 选择器：`instructions`、`candidates`（模型到描述的映射）、`deployment`、`timeout_ms` |
| `algorithm.escalate.accept.systemone` | `{gte, deployment, timeout_ms}`：接受 `answer_check` 概率至少为 `gte` 的答案 |
| `algorithm.escalate.accept.refusal` | `false`：接受 `refusal` 问题未标记为拒答的答案 |
| `algorithm.escalate.stages` | System One 入口：依次提问的决策模型，每个都有原生接受规则 |

## 加载时代码 {#load-time-codes}

| 代码 | 类别 | 含义 |
| --- | --- | --- |
| `span_required` | 错误 | 需要 span 的规则或插件解析到不支持 span 的部署 |
| `input_modality_unsupported` | 错误 | 规则读取其部署无法读取的媒体 |
| `match_empty` | 错误 | `match` 块既没有 `recall` 也没有 `judge` |
| `uncalibrated_preset` | 警告 | 某个预设使用中性阈值 0.5 回答 |
| `renamed_signal_type` | 警告 | `signals.decision` 或 `type: decision` 条件，按 `systemone` 加载 |
| `renamed_algorithm_type` | 警告 | `algorithm.type: decision`，按 `systemone` 加载 |
| `removed_signal_type`、`removed_field` 等 | 错误 | 已移除的形式；见[迁移指南](../../installation/migrate-to-v0-5) |

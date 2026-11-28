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
| `global.model_catalog.deployments.<name>.fallbacks` | `recipes[].signals.systemone` 问题按顺序尝试的备用部署 |
| `global.model_catalog.signal_timeout_ms` | 一个请求所有信号的截止时间 |
| `global.model_catalog.on_signal_overload` | `{action, wait_ms, retry_after_seconds}`：`action: retry` 时，没有备用部署的调用若因决策模型过载被拒绝，会在最长 `wait_ms` 内重试。如果仍无法读取的规则会拒绝该请求，请求就收到客户端会重试的错误。见[决策模型过载时](../deploy/performance.md#when-the-decision-model-is-overloaded) |

`systemone` 部署接受 `endpoint`、`model`、`auth`（`api_key_env` 或 `api_key_file`，或 `credential: delegated`；`header`、`prefix`）、`headers`、`timeout_ms`、`max_retries`、`tls` 和 `capabilities`（`question_types`、`presets`、`modalities`）。

### 为决策部署配置备用模型 {#decision-deployment-fallbacks}

例如，当默认的远程决策部署不可用时，让本地 CPU 模型回答自定义 System One 信号的问题：

```yaml
global:
  model_catalog:
    system:
      decision_model: {deployment: hosted}
    deployments:
      hosted:
        provider: systemone
        schema: vllm-sr/5.0
        endpoint: https://decisions.example.com/v1/systemone
        model: vllm-sr/d3-ultra
        timeout_ms: 250
        fallbacks: [local-cpu]
      local-cpu:
        provider: model_runtime
        artifact: vllm-sr/vela3-fast
        device: cpu
```

网络传输失败、单次调用超时、HTTP 429 或 5xx、运行时不可用时才切换。请求被拒绝、响应格式无效，或部署已给出答案（包括不匹配）时，不切换。先按顺序尝试直接配置的备用部署，再尝试它们各自的备用部署，同一个部署不会调用两次。所有目标都必须支持规则的问题和输入；无效引用或循环会导致配置加载失败。

规则的 `timeout_ms`（如设置）限制整个切换链；多条规则共享调用时取最短值，信号阶段的截止时间仍然有效。每个部署使用自己的 `timeout_ms`（未设置时为 1000 毫秒），且不能超过剩余时间。配置了备用部署的信号遇到 429 会尝试下一个目标，而不是等待容量。如果整条链都无法回答，规则仍按原有的错误策略处理。

每个目标都使用自己的协议、凭据和限制接收规则输入，只应选择允许接收这些输入的部署。规则阈值不会随模型切换而变化，应验证备用模型的分数是否适用于当前判定条件。备用列表仅用于请求阶段或响应阶段的自定义信号求值；应用信号、插件单独读取规则、选择器、评审，以及对外推理和生成后端仍使用原有目标。

对于启用故障切换的信号，Router 会记录每个有效答案实际来自哪个部署，包括主部署直接回答或答案不匹配规则的情况。OpenTelemetry 的 `routing.signal.deployments` 属性是从 `systemone:<rule>` 到部署键的 JSON 映射：请求阶段记录在信号求值 span，响应阶段记录在请求 span。Router Replay 将请求阶段的部署写入 `route_diagnostics.signal_deployments`，将响应阶段的部署写入相应结果的 `metadata.deployment`。没有部署给出有效答案的规则不记录部署。

`sr_systemone_signal_failover_attempts_total` 按 `deployment`、`stage`（`request` 或 `response`）、`role`（`primary` 或 `backup`）和 `outcome` 统计部署尝试次数。例如，主部署返回 HTTP 503 计入 `outcome="unavailable"`，备用部署返回响应则计入其自身的 `outcome="ok"`。该指标按逻辑尝试计数，包括一次批量调用，但不单独统计其中的每条答案或 HTTP 重试；调用成功仍可能缺少某条规则的答案。

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

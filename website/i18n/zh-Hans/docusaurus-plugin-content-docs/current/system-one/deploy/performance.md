---
title: 性能与成本
sidebar_label: 性能与成本
sidebar_position: 2
description: System One 给一个请求带来的成本，以及限定成本的批处理、上限、预算和放置选择。
is_mtpe: false
translation:
  source_commit: "bc5c6bfb747ffe9c5f35d98b203427dada8c4c56"
  source_file: "docs/system-one/deploy/performance.md"
  outdated: false
---

# 性能与成本 {#performance-and-cost}

一个 System One 问题的成本是一个小模型的一次前向计算，而不是聊天模型的一次生成：没有要解码的输出，也没有要解析的文本。一个请求的成本取决于它发出的调用次数、每次调用读取的内容大小，以及模型运行在哪里。

:::tip
大多数请求只发出**一次**调用：请求阶段的那一次。没有被任何内容读取的规则不会被询问，没有问题的阶段不会发出调用。
:::

## 每个请求的调用 {#calls-per-request}

| 阶段 | 调用 | 何时询问 |
| --- | --- | --- |
| 请求信号 | 每个部署一次，包含针对每个状态的所有问题 | 决策、投影或请求插件读取某条规则时 |
| `systemone` 选择器 | 一次 | 匹配的决策使用该选择器时 |
| 评审（`same_request`、`relevance`、`tool_need` + `tool_rank`） | 每个功能一次，每个被评判的候选一个条目 | 功能运行且有候选时 |
| `masking` | 一次，针对请求阶段尚未回答的文本 | 决策进行脱敏时 |
| 响应信号 | 每个部署一次 | `response_guard` 规则读取它们时 |
| `escalate` 检查 | 每个被评判的回答一次 | `accept.systemone` 或 `accept.refusal` |

## 批处理 {#batching}

- **问题共享调用。** 一个阶段向同一部署提出的所有问题一起发送，无论来自哪个信号或规则。模型运行时也会把并发请求的调用合并成批（[引擎参数](../../model-runtime/deploy/engine-args.md#batching)）。
- **问题共享答案。** 针对同一状态、向同一部署询问同一预设的规则共享一个问题。
- **只读一次状态的模型**（d3-turbo、d3-ultra）只读取一次状态，然后从这次读取中回答每个问题，因此对一个请求问第十个问题的成本比第一个多不了多少。Decision 2.0 对多问题请求的共享上下文只计算一次（`enable_shared_context`）。
- **答案会被复用。** Router 为每个本地模型保留最近的 4,096 个答案（`VLLM_SRUN_RESULT_CACHE` 改变这个数量；0 关闭它），并把远程 API 的答案保留最长 10 分钟，直到配置变化。

## 上限 {#caps}

- **评审**最多读取 `judge.max_candidates` 个候选：逐候选评审默认 64 个，choice 评审最多 255 个（也是默认值）。超出部分按召回排名截断，没有召回时按列出顺序截断，绝不失败；`vsr_match_judge_truncated_total` 会计数。
- **召回**限定评审看到的内容：缓存评审确认其召回的前 3 个命中，RAG 评审确认前 20 个文档。
- **工具**在策略过滤后超过 255 个时会被截断，或先用 `recall: {method: bm25}` 缩小范围。

## 预算与截止时间 {#budgets-and-deadlines}

| 设置 | 限定 |
| --- | --- |
| 规则或评审的 `timeout_ms` | 它的答案，最多 60,000 ms；否则使用部署的值，再否则 1,000 ms |
| 部署的 `timeout_ms` | 对附加或远程部署的一次调用尝试 |
| `global.model_catalog.signal_timeout_ms` | 一个请求的全部信号；0（默认）表示从请求自身的截止时间推导 |
| 执行器的 `budget` | `escalate` 的截止时间和调用次数，其 `answer_check` 调用计入截止时间 |

:::note
迟到的答案会让其问题成为未知，由每条规则或功能的失败策略决定。截止时间结束 Router 的等待，但不会停止已经在运行的前向计算。
:::

## 长输入 {#long-inputs}

规则读取模型输入能容纳的内容并丢弃其余部分（`coverage: truncate`），除非它读取全部内容（`coverage: full`）；当输入超过一次前向计算的长度时，分窗口读取。`jailbreak`、`safety`、`pii` 和 `hallucination` 总是读取完整输入；模型未能读取的文本会被计为对请求不利（视为越狱或个人数据），因此填充内容无法让恶意内容绕过它们。

| 模型 | 一次决策 | 窗口 |
| --- | --- | --- |
| vela3-fast | 32,768 个 token | 路由和 span 问题使用 8,192 token 的窗口 |
| vela3-lite、-mini、-flash、-turbo | 最多 262,144 个 token | 每个 span 目标 32,768 个 token |
| d3-turbo | 每个问题 32,768 个 token；长上下文时 262,144 个 | 不支持 span |
| d3-ultra | 每个问题 131,072 个 token | 不支持 span |

:::warning
在核数较少的 CPU 上，对长请求的完整输入扫描可能超过信号截止时间。先测量你的输入长度和并发量，再为部署分配专用 CPU 或 GPU。
:::

## 放置 {#placement}

- **在 CPU 上运行 vela3-fast** 是默认设置。加载后的部署约占 7 GB；给 Router 分配 12 GiB。
- **GPU** 可以运行更大的 Vela 3.0 和 Decision 规格，并在相同延迟下每秒处理多得多的请求。
- **远程 API** 会给每次调用增加一次网络往返，并且其模型没有校准阈值。
- **一个部署承担全部** 只加载一个模型。把各类型分散到多个部署，每个阶段每个部署各发出一次调用。

[选择模型、规模和硬件](../../model-runtime/get-started/choose-a-model.md)给出了各规格的实测延迟，[模型运行时记录](https://github.com/vllm-project/semantic-router/tree/main/src/model-runtime/docs/records)包含完整的测量数据。

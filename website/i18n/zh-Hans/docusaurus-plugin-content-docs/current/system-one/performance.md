---
title: 性能与成本
sidebar_label: 性能与成本
description: System One 给一个请求带来的开销，以及约束它的批处理、上限、预算和部署位置选择。
is_mtpe: false
translation:
  source_commit: "118fb72eb67a0e7193697e350d7b494dc016631e"
  source_file: "docs/system-one/performance.md"
  outdated: false
---

# 性能与成本 {#performance-and-cost}

一个 System One 问题的开销是一个小模型的一次前向计算，而不是聊天模型的一次生成：不需要解码输出，也不需要解析文本。一个请求的开销取决于它发出的调用次数、每次读取的内容大小，以及模型运行的位置。

## 每个请求的调用 {#calls-per-request}

| 阶段 | 调用次数 | 何时提问 |
| --- | --- | --- |
| 请求信号 | 每个部署一次，包含针对每个状态的每个问题 | 有决策、投影或请求插件读取某条规则时 |
| `systemone` 选择器 | 一次 | 匹配的决策使用该选择器时 |
| 评审（`same_request`、`relevance`、`tool_need` + `tool_rank`） | 每个功能一次，每个被评判的候选一项 | 功能运行且有候选时 |
| `masking` | 一次，只针对请求阶段尚未回答的文本 | 决策进行遮蔽时 |
| 响应信号 | 每个部署一次 | 有 `response_guard` 规则读取它们时 |
| `escalate` 检查 | 每个被评判的答案一次 | 使用 `accept.systemone` 或 `accept.refusal` 时 |

大多数请求只发出一次调用：请求阶段的调用。没有任何内容读取的规则不会被提问，没有问题的阶段不会发出调用。

## 批处理 {#batching}

- **问题共享一次调用。** 一个阶段向同一部署提出的所有问题一起发送，无论来自哪个信号或规则。模型运行时还会把并发请求的调用合并成批（[引擎参数](../model-runtime/engine-args#batching)）。
- **问题共享一个答案。** 在同一部署上针对同一状态的同一预设规则共享一个问题。
- **状态只读一次的模型**（d3-turbo、d3-ultra）只读取一次状态，并基于这次读取回答所有问题，所以针对一个请求的第十个问题几乎不比第一个更贵。Decision 2.0 对多问题请求的共享上下文只计算一次（`enable_shared_context`）。
- **答案会被复用。** Router 为每个本地模型保留最近 4,096 个答案（`VLLM_SRUN_RESULT_CACHE` 修改数量，0 关闭），并把远程 API 的答案保留最多 10 分钟，直到配置变更。

## 上限 {#caps}

- **评审**最多读取 `judge.max_candidates` 个候选：逐个候选的评审默认 64 个，choice 评审最多 255 个（也是默认值）。溢出按召回排名截断（没有召回时按列出顺序），绝不失败；`vsr_match_judge_truncated_total` 会计数。
- **召回**约束评审看到的内容：缓存评审确认召回的前 3 个命中，RAG 评审确认前 20 篇文档。
- **工具**在经过策略过滤后超过 255 个时会被截断，或先由 `recall: {method: bm25}` 缩小。

## 预算与截止时间 {#budgets-and-deadlines}

| 设置 | 约束 |
| --- | --- |
| 规则或评审的 `timeout_ms` | 它的答案，最长 60,000 毫秒；否则使用部署的值，再否则为 1,000 毫秒 |
| 部署的 `timeout_ms` | 对挂接或远程部署的一次调用尝试 |
| `global.model_catalog.signal_timeout_ms` | 一个请求的所有信号；0（默认）表示从请求自己的截止时间推导 |
| 执行器的 `budget` | `escalate` 的截止时间和调用次数，其中 `answer_check` 调用计入截止时间 |

迟到的答案会让它的问题变为未知，由每条规则或功能的失败策略决定后续。截止时间会结束 Router 的等待，但不会中止已经在运行的前向计算。

## 长输入 {#long-inputs}

规则读取能放进模型输入的部分并丢弃其余部分（`coverage: truncate`），除非它要求读取全部（`coverage: full`）；输入超过一次前向计算的长度时，会分窗口读取。`jailbreak`、`safety`、`pii` 和 `hallucination` 总是读取完整输入；模型没能读到的文本会被不利地计入请求（视为越狱或个人数据），所以靠填充无法把内容带过它们。

| 模型 | 一次决策 | 窗口 |
| --- | --- | --- |
| vela3-fast | 32,768 个 token | router 问题和 span 问题使用 8,192 个 token 的窗口 |
| vela3-lite、-mini、-flash、-turbo | 最多 262,144 个 token | 每个 span 目标 32,768 个 token |
| d3-turbo | 每个问题 32,768 个 token；长上下文时 262,144 个 | 不支持 span |
| d3-ultra | 每个问题 131,072 个 token | 不支持 span |

在核数较少的 CPU 上，对长请求的完整扫描可能超过信号截止时间：请测量你的输入长度和并发量，然后为部署分配专用 CPU 或 GPU。

## 部署位置 {#placement}

- **CPU 上的 vela3-fast** 是默认选择。加载后的部署约占 7 GB；给 Router 分配 12 GiB。
- **GPU** 运行更大的 Vela 3.0 和 Decision 规格，在相同延迟下每秒能处理多得多的请求。
- **远程 API** 给每次调用增加一次网络往返，而且它的模型没有校准阈值。
- **所有内容使用一个部署**只加载一个模型。把类型分散到多个部署，会在每个阶段为每个部署各发出一次调用。

[选择模型、规格和硬件](../model-runtime/choose-a-model)给出了各规格的实测延迟，[模型运行时记录](https://github.com/vllm-project/semantic-router/tree/main/src/model-runtime/docs/records)包含完整测量数据。

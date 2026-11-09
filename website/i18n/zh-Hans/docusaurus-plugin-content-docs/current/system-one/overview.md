---
title: System One
sidebar_label: 概述
description: 决策模型通过一套类型化问题 API，回答 Router 做出的每一个判断。
is_mtpe: false
translation:
  source_commit: "118fb72eb67a0e7193697e350d7b494dc016631e"
  source_file: "docs/system-one/overview.md"
  outdated: false
---

# System One {#system-one}

System One 是 Router 判断请求的方式。决策模型读取一段上下文（称为状态，state），并以概率回答关于它的类型化问题：这几个选项中哪个合适、这是否成立、程度有多高、这些标签中哪些适用、每个标签出现在文本的什么位置。凡是需要判断的问题，从“这个请求属于哪个学科？”到“这条缓存请求问的是同一件事吗？”，Router 都通过同一个 API 向它提问。

```text
POST /v1/systemone
{"state": {"request": "Prove that the square root of 2 is irrational."},
 "questions": {"domain": {...}, "jailbreak": {...}, "complexity": {...}}}
```

## 零样本优先 {#zero-shot-first}

问题写好就能用：不需要训练分类器，也不需要收集示例库。每个 System One 模型都能根据问题的说明和各选项的描述，零样本回答它所支持的每一个问题。在此之上有两项按模型家族进行的优化，两者都不是必需的：

- **训练绑定。** 有些模型家族针对 Router 的内置问题做过训练。当回答问题的模型有该问题的训练形式时，Router 按训练形式提问，再把答案映射回来。
- **校准阈值。** 对于在 router signal suite 上测量过的家族，内置问题的默认阈值保持一个已知的工作点。没有校准阈值的模型使用 0.5，并在加载配置时给出 `uncalibrated_preset` 警告。

各模型能回答的问题类型不同（见下表）：Decision 1.0、2.0 和 3.0 不返回 span，vela3-fast 不回答 `score` 问题。能回答 `noul` 的模型也能回答 `set` 问题，方式是每个标签问一个 `noul`。规则、评审或选择器的问题类型如果其部署不能回答，配置加载失败：span 问题（例如 PII 检测）报 `span_required`，其他类型报 `question_type_unsupported`。加载时已知内置模型能回答什么，以及 `systemone` 部署在 `capabilities.question_types` 中声明了什么；其他模型在生成准备时按其模型卡或 API 列表检查。

## 默认决策模型 {#the-default-decision-model}

什么都不配置时，Router 的决策模型是预先声明的部署 `primary`：Vela 3.0 vela3-fast，一个适合 CPU 的编码器，由 Router 旁边的内置[模型运行时](../model-runtime/overview)提供服务。它在第一个规则、选择器、插件或评审解析到它时启动，用训练过的适配器和校准阈值回答每个内置问题，并能返回 span。

```yaml
global:
  model_catalog:
    system:
      decision_model: {deployment: primary}
```

这一行就是默认值，可以省略。把它指向另一个部署，就改变了所有问题默认提问的模型；某个类型、某个配方或单条规则仍可以指定自己的部署。[选择部署](./deployments)介绍了本地 GPU 规格、挂接的运行时和远程 System One API。

## 哪些功能会向 System One 提问 {#what-asks-system-one}

| 功能 | 问什么 | 页面 |
| --- | --- | --- |
| 应用信号 | 各问一个内置问题：领域、事实核查、用户反馈、越狱、安全、PII、幻觉、拒答、轮次关系、复杂度、输出模态 | [信号](../tutorials/signal/overview) |
| `systemone` 信号 | 你用自然语言写的问题 | [编写自定义问题](./custom-questions) |
| `systemone` 选择器 | 哪个候选模型应该回答 | [System One 选择](../tutorials/algorithm/selection/systemone) |
| `response_guard` | 关于模型答案的响应阶段信号 | [Response Guard](../tutorials/plugin/response-guard) |
| `masking` | 最终发往提供方的请求中，个人数据在哪里 | [Masking](../tutorials/plugin/masking) |
| `response_cache` | 召回的缓存请求是否问的是同一件事（`same_request`） | [Response Cache](../tutorials/plugin/response-cache) |
| `context` | 一篇文档或一个片段是否有助于回答请求（`relevance`） | [Context](../tutorials/plugin/context) |
| `tools` | 请求是否需要工具，以及需要哪些（`tool_need`、`tool_rank`） | [Tools](../tutorials/plugin/tools) |
| `escalate` | 一个答案是否可以接受（`answer_check`） | [Escalate](../tutorials/algorithm/looper/escalate) |

[Router 如何使用 System One](./router) 展示了每个功能发出的调用。

## 模型家族 {#model-families}

| 家族 | 模型 | 回答 | 运行于 |
| --- | --- | --- | --- |
| Vela 3.0 | `vela3-fast`（默认）、`vela3-lite`、`vela3-mini`、`vela3-flash`、`vela3-turbo` | choice、noul 和 span，内置问题有训练过的适配器；更大的规格还回答 score，并能读取图片 | CPU（vela3-fast）或 GPU |
| Decision 3.0（d3） | `d3-edge` 到 `d3`、`d3-turbo` 和 `d3-ultra`（也有托管服务） | 基于文本、图片和视频的 choice、noul 和 score | GPU，或托管 API |
| Decision 2.0 与 1.0 | 0.6B 到 27B | choice、noul、score | CPU（0.6B）或 GPU |
| Vela 2.0 | 0.3B 到 9B | choice、noul、score、set、span；针对内置问题训练过 | CPU（0.3B）或 GPU |

[支持矩阵](./reference#support-matrix)列出了每个家族对各问题是训练回答、零样本回答还是无法回答。

## 本章内容 {#in-this-chapter}

- [概念](./concepts)：问题类型、状态及其组成部分、预设、训练与零样本回答、span 能力和限制。
- [System One API](./api)：标准请求、批量形式、模型卡、错误和限制。
- [选择部署](./deployments)：本地、挂接或远程模型，问题的部署从何而来，以及 Router 在加载时检查什么。
- [Router 如何使用 System One](./router)：每个功能背后的调用、调用如何合并、被评审候选的上限，以及召回与评审的取舍。
- [编写自定义问题](./custom-questions)：把一个路由想法变成 `systemone` 规则，直接测试并调优。
- [性能与成本](./performance)：一个请求的开销，以及约束它的各项设置。
- [参考](./reference)：预设目录、支持矩阵、配置字段和加载时代码。

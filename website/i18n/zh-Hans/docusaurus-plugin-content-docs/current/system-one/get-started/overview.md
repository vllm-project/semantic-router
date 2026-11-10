---
title: System One
sidebar_label: 概述
sidebar_position: 1
slug: /system-one/overview
description: 决策模型通过一套类型化问题 API，回答 Router 做出的每一个判断。
is_mtpe: false
translation:
  source_commit: "bc5c6bfb747ffe9c5f35d98b203427dada8c4c56"
  source_file: "docs/system-one/get-started/overview.md"
  outdated: false
---

# System One {#system-one}

System One 是 Router 判断请求的方式。**决策模型**读取请求，并以概率回答关于它的类型化问题。因此，从“这个请求属于哪个学科？”到“这条缓存请求问的是同一件事吗？”，每一个路由判断都经过同一个模型和同一个 API。

```json title="POST /v1/systemone"
{
  "state": {"request": "Prove that the square root of 2 is irrational."},
  "questions": {
    "math": {"type": "noul", "instructions": "Is this a mathematics request?"},
    "difficulty": {"type": "score", "instructions": "How hard is this request?",
                   "levels": ["Easy", "Medium", "Hard"]}
  }
}
```

答案给出 `math` 的 P(是) 和 `difficulty` 的期望等级。配置中的规则依据这些数值进行匹配。

## 为什么这样设计 {#why-it-works-this-way}

- **零样本优先。** 问题写好就能用。不需要训练分类器，也不需要收集示例库：每个 System One 模型都根据问题的说明和各选项的描述作答。
- **一个模型回答所有判断。** 信号、模型选择器、守卫，以及缓存、RAG 和工具的评审都问同一个部署，Router 把同一阶段的问题合并成一次调用。
- **在有帮助的地方使用训练。** 有些模型家族针对 Router 的内置问题做过训练，有些有校准阈值。两者都不是必需的；参见[训练与零样本回答](./concepts.md#trained-and-zero-shot-answers)。

## 默认决策模型 {#the-default-decision-model}

什么都不配置时，每个问题都发给预先声明的部署 `primary`：Vela 3.0 **vela3-fast**，一个对 CPU 友好的编码器，由内置[模型运行时](../../model-runtime/get-started/overview.md)在 Router 旁边提供服务。它在第一个规则、选择器、插件或评审需要它时启动，用训练过的适配器和校准阈值回答每一个内置问题，并返回 span。

```yaml title="config.yaml (the default, which you can leave out)"
global:
  model_catalog:
    system:
      decision_model: {deployment: primary}
```

:::tip
把 `decision_model` 指向另一个部署，即可更换每个问题所问的模型。某种信号类型、某个 recipe 或单条规则仍可指定自己的部署；参见[选择部署](../deploy/deployments.md)。
:::

## 谁会询问 System One {#what-asks-system-one}

| 功能 | 询问什么 | 页面 |
| --- | --- | --- |
| 应用信号 | 各自一个内置问题：domain、fact check、user feedback、jailbreak、safety、PII、hallucination、refusal、turn relation、complexity、output modality | [信号](../../tutorials/signal/overview.md) |
| `systemone` 信号 | 你自己用自然语言写的问题 | [编写自定义问题](../use/custom-questions.md) |
| `systemone` 选择器 | 哪个候选模型应该回答 | [System One 选择](../../tutorials/algorithm/selection/systemone.md) |
| `response_guard` | 关于模型回答的响应阶段信号 | [Response Guard](../../tutorials/plugin/response-guard.md) |
| `masking` | 最终发往提供方的请求中，个人数据在哪里 | [Masking](tutorials/plugin/masking.md) |
| `response_cache` | 召回的缓存请求是否问的是同一件事（`same_request`） | [Response Cache](../../tutorials/plugin/response-cache.md) |
| `context` | 某个文档或片段是否有助于回答请求（`relevance`） | [Context](../../tutorials/plugin/context.md) |
| `tools` | 请求是否需要工具，以及需要哪些（`tool_need`、`tool_rank`） | [Tools](../../tutorials/plugin/tools.md) |
| `escalate` | 某个回答是否可以接受（`answer_check`） | [Escalate](../../tutorials/algorithm/looper/escalate.md) |

[Router 如何使用 System One](../use/router.md) 展示了每个功能发出的调用。

## 模型家族 {#model-families}

| 家族 | 模型 | 回答 | 运行于 |
| --- | --- | --- | --- |
| Vela 3.0 | `vela3-fast`（默认）、`vela3-lite`、`vela3-mini`、`vela3-flash`、`vela3-turbo` | choice、noul 和 span，内置问题由训练过的适配器回答；较大的规格还回答 score 并读取图像 | CPU（vela3-fast）或 GPU |
| Decision 3.0（d3） | `d3-edge` 到 `d3`、`d3-turbo` 和 `d3-ultra`（也提供托管服务） | 关于文本、图像和视频的 choice、noul 和 score | GPU，或托管 API |
| Decision 2.0 与 1.0 | 0.6B 到 27B | choice、noul、score | CPU（0.6B）或 GPU |
| Vela 2.0 | 0.3B 到 9B | choice、noul、score、set、span；针对内置问题训练过 | CPU（0.3B）或 GPU |

各模型能回答的问题类型不同。Decision 1.0、2.0 和 3.0 不返回 span，vela3-fast 不回答 `score` 问题。如果规则、评审或选择器的问题类型其部署无法回答，配置加载失败：span 问题（例如 PII 检测）报 `span_required`，其他类型报 `question_type_unsupported`。

每个模型在[支持的模型](../../model-runtime/supported-models/overview.md)下都有自己的页面；[支持矩阵](../reference/support-matrix.md)列出各家族对每个内置问题是训练回答、零样本回答还是无法回答。

## 下一步 {#next-steps}

- [概念](./concepts.md)：问题类型、状态、预设、训练与零样本回答，以及 span 能力。
- [System One API](../use/api.md)：请求与响应格式、模型卡和错误。
- [编写自定义问题](../use/custom-questions.md)：把一个路由想法变成 `systemone` 规则。
- [选择部署](../deploy/deployments.md)：本地、附加或远程模型，以及 Router 在加载时检查什么。

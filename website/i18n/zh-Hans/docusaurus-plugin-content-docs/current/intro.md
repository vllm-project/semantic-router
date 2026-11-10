---
is_mtpe: false
sidebar_position: 1
sidebar_label: 简介
description: 面向模型与算力的开放、可编程决策层。
translation:
  source_commit: "bc5c6bfb747ffe9c5f35d98b203427dada8c4c56"
  source_file: "docs/intro.md"
  outdated: false
---

import ThemedImage from '@theme/ThemedImage';

# 智能，超越单一模型。

<div className="docs-intro-brand">
 <ThemedImage
 className="docs-intro-brand__logo"
 alt="vLLM Semantic Router"
 sources={{
 light: '/img/vllm-sr-logo.light.png',
 dark: '/img/vllm-sr-logo.white.png',
 }}
 />
 <p className="docs-intro-brand__tagline">面向模型与算力的开放、可编程<strong>决策层</strong>。</p>
</div>

为 agent harness 提供稳定的模型 API。vLLM Semantic Router 在 OpenAI 或 Anthropic 兼容端点背后，通过显式策略选择或组合模型，因此你可以更换模型和策略，而无需重写 harness 集成。

- **为每次调用路由：** 这次用快速的本地模型，下次用专项模型、更长的上下文或第二意见。
- **用一个模型做判断：** 决策模型回答策略针对每个请求提出的问题。
- **直接提供判断：** 应用也可以通过 [System One](system-one/overview) API 直接向决策模型提问。

## 为什么需要路由？ {#why-route}

能力、延迟、成本和信任边界会随请求、用户、会话和可用基础设施而变化。共享的路由策略让每个 harness 不必在自己的代码里重复做这些选择。

## 你可以编程什么 {#what-you-can-program}

入口选择一个隔离的配方。配方的信号捕获意图、难度、上下文、模态、身份、风险和运行时观测，用于：

- **选择或组合模型：** 选择本地模型或专项模型，从小模型逐级升级到更强的模型，或运行有界的多模型工作流。
- **添加路由行为：** 提示词、检索、记忆、工具过滤、缓存、安全检查和校验。
- **选择执行路径：** 跨异构硬件的云、数据中心或边缘后端。
- **检查并改进决策：** 路由元数据、反馈、回放和评估。

## 各司其职 {#who-does-what}

| 部分 | 负责 |
| --- | --- |
| **Agent harness** | Agent 循环、工具执行和任务状态 |
| **Semantic Router** | 每次调用的策略和有界的模型协作 |
| **模型运行时** | Router 自己的决策模型和嵌入模型 |
| **推理平台** | 对话后端，以及它们的放置、批处理和容量 |

Router 默认直接提供自己的 listener；基于 Envoy 的网关是可选的。模块边界见[组件架构](overview/component-architecture)。

## 从这里开始 {#start-here}

- 完成[快速开始](installation/installation.md)。
- [连接 agent harness](installation/agent-harness)。
- 浏览[使用场景](overview/use-cases)。
- 阅读[系统概览](overview/semantic-router-overview)和[路由流水线](overview/signal-driven-decisions)。
- 用[入口与配方](tutorials/global/entrypoints-and-recipes)构建虚拟模型。
- 比较[部署选项](installation/deployment-options)。

## 项目 {#project}

vLLM Semantic Router 以 Apache 2.0 许可开源。要提出变更或加入社区，请参阅[贡献指南](https://github.com/vllm-project/semantic-router/blob/main/CONTRIBUTING.md)。

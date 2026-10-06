---
sidebar_position: 1
sidebar_label: 简介
description: 面向模型与算力的开放、可编程决策层。
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/intro.md"
  outdated: false
---

import ThemedImage from '@theme/ThemedImage';

# 欢迎使用 vLLM Semantic Router

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

vLLM Semantic Router 让 agent harness（管理 Agent 任务循环、工具与状态的运行框架）通过开放、可编程的决策层使用模型与算力。Harness 调用稳定的 OpenAI 或 Anthropic 兼容端点；显式策略在已配置的后端上选择模型、协调有界多模型策略，并应用路由特定行为。

我们的愿景是让智能能够超越任一模型的边界，持续演进。我们的使命是让连接 agent harness、模型与算力的决策开放、可编程且可观测。

## 问题：一次 AI 请求不只是流量

Agent harness 通过模型调用、工具执行和不断变化的上下文推进任务。一次推理调用可能适合快速的本地模型，另一次则需要专项模型、更大的上下文容量，或多个模型之间的有界校验。这些路径可能跨越云、数据中心或边缘。

每条路径在能力、延迟、成本和信任上各有取舍。正确选择还会随请求、用户、会话和可用基础设施而变化。

如果每个 harness 都硬编码这些选择，其集成代码就会与当前模型机群耦合。同样的路由逻辑在各客户端重复出现，系统扩大后就难以变更、解释或评估。

## 思路：让智能可编程

Semantic Router 把这项决策放到请求路径上的共享层。它可以观察眼前的工作——意图、难度、上下文、模态、身份、风险、偏好和系统状态——再把稳定的入口解析到隔离的配方。

配方可以选择一个模型、沿级联升级、协调有界的多模型工作流，或挂接检索、记忆、工具过滤、缓存、安全检查和校验等行为。Harness 继续使用稳定的模型 API，策略和已配置的模型池则可以在背后演进。

结果不只是一个模型名：

- **正确的模型路径：** 直达、专项、本地、级联或协作。
- **正确的配套能力：** 在请求需要时提供检索、记忆、工具过滤、提示词、缓存或校验。
- **正确的执行边界：** 在异构硬件上使用已配置的云、数据中心或边缘后端。
- **发生了什么的证据：** 路由元数据，以及已配置的反馈、回放和评估工作流。

Harness 负责任务循环、工具执行和任务生命周期。Semantic Router 负责每次调用的路由策略与已配置的模型协作。网关和 Envoy 承载流量；推理平台执行模型，管理副本放置、批处理与容量。这里对**算力**的决策指选择已配置的推理路径与有界模型调用，其执行仍由相应平台负责。

## 从你想做的事开始

- **在本地运行：** 按[快速开始](/zh-Hans/docs/installation)操作，并通过 Router 发送一次请求。
- **连接 agent harness：** 阅读 [agent harness 指南](/zh-Hans/docs/installation/agent-harness)，了解集成方式与职责边界。
- **找到适合工作负载的模式：** 浏览[使用场景](overview/use-cases)，了解跨云、数据中心、边缘和企业部署的 Agent 模型调用。
- **理解系统：** 阅读[系统概览](overview/semantic-router-overview)和[路由流水线](overview/signal-driven-decisions)。
- **打造稳定的模型体验：** 了解[入口与配方](tutorials/global/entrypoints-and-recipes)如何把共享模型池变成面向目标的虚拟模型。
- **选择环境：** 比较 [Docker、Kubernetes 和硬件路径](installation/deployment-options)。

## 项目

vLLM Semantic Router 以 Apache 2.0 许可开源。要提出变更或加入社区，请参阅[贡献指南](https://github.com/vllm-project/semantic-router/blob/main/CONTRIBUTING.md)。

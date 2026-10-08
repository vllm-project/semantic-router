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

为 agent harness 提供稳定的模型 API。vLLM Semantic Router 在 OpenAI 或 Anthropic 兼容端点背后，通过显式策略选择或组合模型。模型与策略可以演进，无需重写 harness 集成。

## 为什么需要路由？

一次调用需要快速的本地模型，另一次需要专项模型、更长上下文或多模型校验。能力、延迟、成本和信任边界随请求、用户、会话与可用基础设施变化。共享路由策略让每个 harness 无需重复编码这些选择。

## 你可以编程什么？

入口选择隔离的配方。信号捕获意图、难度、上下文、模态、身份、风险、偏好及已配置的运行时观测，用于：

- **选择或组合模型：** 选择本地或专项模型，通过级联升级，或运行有界多模型工作流。
- **添加路由行为：** 提示词、检索、记忆、工具过滤、缓存、安全检查和校验。
- **选择执行路径：** 异构硬件上已配置的云、数据中心或边缘后端。
- **检查并改进决策：** 路由元数据、反馈、回放和评估。

Harness 负责任务循环、工具执行与任务状态；Router 负责每次调用的策略和有界模型协作。网关与 Envoy 传输请求，推理平台执行模型并管理副本放置、批处理和容量。

## 从这里开始

- 完成[快速开始](/zh-Hans/docs/installation)。
- [连接 agent harness](/zh-Hans/docs/installation/agent-harness)。
- 浏览[使用场景](overview/use-cases)。
- 阅读[系统概览](overview/semantic-router-overview)和[路由流水线](overview/signal-driven-decisions)。
- 用[入口与配方](tutorials/global/entrypoints-and-recipes)构建虚拟模型。
- 比较[部署选项](installation/deployment-options)。

## 项目

vLLM Semantic Router 以 Apache 2.0 许可开源。要提出变更或加入社区，请参阅[贡献指南](https://github.com/vllm-project/semantic-router/blob/main/CONTRIBUTING.md)。

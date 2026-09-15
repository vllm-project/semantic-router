---
sidebar_position: 1
sidebar_label: 简介
description: 在稳定的模型 API 背后，构建可编程的 Mixture-of-Models 系统。
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
 <p className="docs-intro-brand__tagline">让你的 Mixture-of-Models 可编程。</p>
</div>

vLLM Semantic Router 是开源的路由与控制层，用于在异构 AI 基础设施上构建 Mixture-of-Models 系统。应用调用稳定的 OpenAI 或 Anthropic 兼容端点，由服务层为每次请求选择——或组合——能力路径。

## 问题：一次 AI 请求不只是流量

现代 AI 应用很少只依赖一个可互换的模型。一次请求可能需要快速的本地模型、专项或前沿模型、检索、记忆、工具、校验器，或若干模型协同。这些路径可能跨越云、数据中心或边缘。

每条路径在能力、延迟、成本和信任上各有取舍。正确选择还会随请求、用户、会话和可用基础设施而变化。

如果每个应用都硬编码这些选择，产品代码就会和当前模型机群绑死。同样的路由逻辑在各客户端重复出现，系统扩大后就难以变更、解释或评估。

## 思路：让智能可编程

Semantic Router 把这项决策放到请求路径上的共享层。它可以观察眼前的工作——意图、难度、上下文、模态、身份、风险、偏好和系统状态——再把稳定的入口解析到隔离的配方。

配方可以选择一个模型、沿级联升级、协调有界的多模型工作流，或挂接检索、记忆、工具过滤、缓存、安全检查和校验等行为。应用继续使用熟悉的单一 API，能力路径则可以在背后演进。

结果不只是一个模型名：

- **正确的模型路径：** 直达、专项、本地、级联或协作。
- **正确的配套能力：** 在请求需要时提供检索、记忆、工具、提示词、缓存或校验。
- **正确的执行边界：** 在异构硬件上使用已配置的云、数据中心或边缘后端。
- **发生了什么的证据：** 路由元数据，以及已配置的反馈、回放和评估工作流。

vLLM Semantic Router 不替代网关或模型服务。Envoy 继续承载流量，推理运行时继续生成响应。Router 负责协调二者之间的语义工作。

## 从你想做的事开始

- **在本地运行：** 按[快速开始](/zh-Hans/docs/installation)操作，并通过 Router 发送一次请求。
- **找到适合工作负载的模式：** 浏览[使用场景](overview/use-cases)，覆盖云、数据中心、边缘和企业部署。
- **理解系统：** 阅读[系统概览](overview/semantic-router-overview)和[路由流水线](overview/signal-driven-decisions)。
- **打造稳定的模型体验：** 了解[入口与配方](tutorials/global/entrypoints-and-recipes)如何把共享模型池变成面向目标的虚拟模型。
- **选择环境：** 比较 [Docker、Kubernetes 和硬件路径](installation/deployment-options)。

## 项目

vLLM Semantic Router 以 Apache 2.0 许可开源。要提出变更或加入社区，请参阅[贡献指南](https://github.com/vllm-project/semantic-router/blob/main/CONTRIBUTING.md)。

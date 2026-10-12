---
title: 提案
description: vLLM Semantic Router 的设计提案、概念验证探索、实现记录和架构决策。
translation:
  source_commit: "2b7519a84aec96963b02a3534e82908beba33f76"
  source_file: "docs/proposals/index.md"
  outdated: false
---

本集合记录需要比功能指南更多上下文的想法和设计决策。各页上的状态区分了拟议工作、实验、已实现契约，以及有意限制产品范围的决策。

## 路由与选择 {#routing--selection}

路由器如何理解请求、约束模型选择，并在不接管服务层调度的前提下改进选择。

| 提案 | 创建日期 | 状态 | 范围 |
| --- | --- | --- | --- |
| [路由范围：逐查询与容量感知路由](./batch-and-capacity-aware-routing) | 2026-07-14 | 决策记录 | 保持语义路由逐查询，容量处理留在服务层。 |
| [路由学习](./router-learning-memory-and-adaptations) | 2026-06-20 | 已实现 | 在线适配、路由保护和离线配方改进。 |
| [提示词分类路由](./prompt-classification-routing) | 2025-10-08 | 提案 | 关键词、正则、嵌入和分类器信号融合。 |

## 工作流、记忆与工具 {#workflows-memory--tools}

围绕已路由请求协调模型、检索、记忆和工具的多步执行与上下文能力。

| 提案 | 创建日期 | 状态 | 范围 |
| --- | --- | --- | --- |
| [Router Flow 工作流](./router-flow-workflows) | 2026-06-30 | 已实现 | 有界的静态和动态多模型工作流。 |
| [智能体感知的 Router 契约](./agent-based-routing) | 2026-08-29 | 提案 | Router 接缝上的有界智能体事实和外部运行时交接（[Epic #2994](https://github.com/vllm-project/semantic-router/issues/2994)）。 |
| [审议算法](./deliberation-algorithms) | 2026-06-17 | 提案 | 感知依据的多模型综合。 |
| [智能体记忆](./agentic-memory) | 2026-02-09 | 概念验证 | 跨会话记忆检索与持久化。 |
| [OpenAI RAG 集成](./agentic-rag) | 2026-01-23 | 已实现 | 通过 OpenAI Files 和 Vector Stores 检索。 |
| [高级工具过滤](./advanced-tool-filtering) | 2026-01-14 | 已实现 | 对工具候选做可解释的过滤和重排序。 |

## 安全与韧性 {#safety--resilience}

当模型不能或不应当处理请求时，使路由行为保持显式的失败、资格和响应质量边界。

| 提案 | 创建日期 | 状态 | 范围 |
| --- | --- | --- | --- |
| [模型执行回退](./model-execution-fallback) | 2026-08-10 | 提案 | 跨模型回退的安全所有权边界。 |
| [PRISM](./Prism-153key) | 2026-03-20 | 提案 | 模型资格与合法性检查。 |
| [TruthLens](./hallucination-mitigation-milestone) | 2025-12-02 | 提案 | 网关级幻觉检测与缓解。 |

## 配置与协议 {#configuration--protocols}

编写路由器行为、以及从不同客户端和传输协议到达路由引擎的共享契约。

| 提案 | 创建日期 | 状态 | 范围 |
| --- | --- | --- | --- |
| [Open Intelligence Index 1.0 与 Unified Model Arena](./open-intelligence-index-and-model-arena) | 2026-09-09 | 已实现 | 定义六个开放核心基准、完整案例指数、物理/虚拟排名、运营方证据、路由目标和基准版本迁移。 |
| [统一模型目录与评估指数](./unified-model-catalog-and-evaluation-index) | 2026-09-04 | 已实现 | 统一提供商、协议、模型、推理、展示、Day-0 和基准比较元数据。 |
| [统一配置契约 v0.3](./unified-config-contract-v0-3) | 2026-03-17 | 已实现 | 跨编写和部署面的单一配置契约。 |
| [多协议适配器架构](./multi-protocol-adaptor) | 2026-02-18 | 提案 | 与协议无关地访问路由引擎。 |
| [独立 HTTP Gateway](./standalone-http-gateway) | 2026-08-31 | 提案 | 链接路由器包的独立网关二进制；无需 Envoy 即可运行路由，简化部署。 |

## 服务集成 {#serving-integrations}

将语义模型选择连接到基础设施所拥有的模型服务、副本选择和执行的分层契约。

| 提案 | 创建日期 | 状态 | 范围 |
| --- | --- | --- | --- |
| [vLLM Production Stack 集成](./production-stack-integration) | 2025-10-13 | 提案 | 分层的语义与基础设施路由。 |
| [NVIDIA Dynamo 集成](./nvidia-dynamo-integration) | 2025-10-09 | 提案 | 在 Dynamo 的 worker 级路由之上做语义路由。 |

状态描述文档当前角色：

- **提案**：尚未表述为已完整交付的设计。
- **概念验证**：带有明确生产限制的实验。
- **已实现**：当前仓库中已体现的契约。
- **决策记录**：架构选择，包括有意保持在路由器之外的工作。

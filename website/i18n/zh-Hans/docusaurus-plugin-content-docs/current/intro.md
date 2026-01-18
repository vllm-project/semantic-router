---
sidebar_position: 1
translation:
  source_commit: "bac2743"
  source_file: "docs/intro.md"
  outdated: true
---

# vLLM Semantic Router

**Mixture of Models (MoM) 的系统级智能** - 一个为 LLM 系统带来集体智能（Collective Intelligence）的智能路由层。作为 Envoy ExtProc（External Processor）运行，它利用 **Signal-Driven Decision Engine（信号驱动决策引擎）** 和 **Plugin Chain 架构（插件链架构）** 来捕获缺失的信号，做出更好的路由决策，并保护您的 LLM 基础设施。

## 项目目标

我们致力于构建 Mixture of Models (MoM) 的**系统级智能**，将 **Collective Intelligence（集体智能）** 引入 **LLM 系统**，以回答以下问题：

1. **如何捕获**请求、响应和上下文中的缺失信号？
2. **如何结合这些信号**以做出更好的决策？
3. **如何在不同模型之间更高效地协作**？
4. **如何保护**现实世界和 LLM 系统免受 jailbreak（越狱）、PII 泄漏和 hallucination（幻觉）的侵害？
5. **如何收集有价值的信号**并构建自学习系统？

## 核心架构

### Signal-Driven Decision Engine

捕获并结合 **7 种类型的信号**以做出智能路由决策：

| 信号类型 | 描述 | 用例 |
|------------|-------------|----------|
| **keyword** | 支持 AND/OR 运算符的模式匹配 | 针对特定术语的快速规则路由 |
| **embedding** | 基于 embedding 的语义相似度 | 意图检测和语义理解 |
| **domain** | MMLU 领域分类（14 个类别） | 学术和专业领域路由 |
| **fact_check** | 基于 ML 的事实核查需求检测 | 识别需要事实验证的查询 |
| **user_feedback** | 用户满意度和反馈分类 | 处理后续消息和更正 |
| **preference** | 基于 LLM 的路由偏好匹配 | 通过外部 LLM 进行复杂意图分析 |
| **language** | 多语言检测（100 多种本地化语言） | 路由查询特定语言的模型 |

**工作原理**：从请求中提取信号，在决策规则中使用 AND/OR 运算符进行组合，并用于选择最佳模型和配置。

### Plugin Chain 架构

用于请求/响应处理的可扩展插件系统：

| 插件类型 | 描述 | 用例 |
|------------|-------------|----------|
| **semantic-cache** | 基于语义相似度的缓存 | 降低相似查询的延迟和成本 |
| **jailbreak** | 对抗性 prompt 检测 | 阻止 prompt injection 和 jailbreak 尝试 |
| **pii** | PII（个人身份信息）检测 | 保护敏感数据并确保合规性 |
| **system_prompt** | 动态 system prompt 注入 | 为每个路由添加上下文感知指令 |
| **header_mutation** | HTTP header 操控 | 控制路由和后端行为 |
| **hallucination** | token 级 hallucination 检测 | 生成过程中的实时事实验证 |

**工作原理**：插件形成处理链，每个插件都可以检查/修改请求和响应，并且可以针对每个 decision 配置启用/禁用。

## 架构概览

import ZoomableMermaid from '@site/src/components/ZoomableMermaid';

<ZoomableMermaid title="Signal-Driven Decision + Plugin Chain Architecture" defaultZoom={3.5}>
{`graph TB
    Client[Client Request] --> Envoy[Envoy Proxy]
    Envoy --> Router[Semantic Router ExtProc]

    subgraph "Signal Extraction Layer"
        direction TB
        Keyword[Keyword Signals<br/>Pattern Matching]
        Embedding[Embedding Signals<br/>Semantic Similarity]
        Domain[Domain Signals<br/>MMLU Classification]
        FactCheck[Fact Check Signals<br/>Verification Need]
        Feedback[User Feedback Signals<br/>Satisfaction Analysis]
        Preference[Preference Signals<br/>LLM-based Matching]
        Language[Language Signals<br/>Multi-language Detection]
    end

    subgraph "Decision Engine"
        Rules[Decision Rules<br/>AND/OR Operators]
        ModelSelect[Model Selection<br/>Priority/Confidence]
    end

    subgraph "Plugin Chain"
        direction LR
        Cache[Semantic Cache]
        Jailbreak[Jailbreak Guard]
        PII[PII Detector]
        SysPrompt[System Prompt]
        HeaderMut[Header Mutation]
        Hallucination[Hallucination Detection]
    end

    Router --> Keyword
    Router --> Embedding
    Router --> Domain
    Router --> FactCheck
    Router --> Feedback
    Router --> Preference
    Router --> Language

    Keyword --> Rules
    Embedding --> Rules
    Domain --> Rules
    FactCheck --> Rules
    Feedback --> Rules
    Preference --> Rules
    Language --> Rules

    Rules --> ModelSelect
    ModelSelect --> Cache
    Cache --> Jailbreak
    Jailbreak --> PII
    PII --> SysPrompt
    SysPrompt --> HeaderMut
    HeaderMut --> Hallucination

    Hallucination --> Backend[Backend Models]
    Backend --> Math[Math Model]
    Backend --> Creative[Creative Model]
    Backend --> Code[Code Model]
    Backend --> General[General Model]`}
</ZoomableMermaid>

## 关键优势

### 智能路由

- **Signal Fusion（信号融合）**：结合多种信号（keyword + embedding + domain）实现精准路由
- **自适应决策**：使用 AND/OR 运算符创建复杂的路由逻辑
- **模型专业化**：将数学问题路由到数学模型，代码问题路由到代码模型等

### 安全与合规

- **多层保护**：PII 检测、jailbreak 防御、hallucination 检测
- **策略执行**：模型特定的 PII 策略和安全规则
- **审计追踪**：所有安全决策的完整日志记录

### 性能与成本

- **Semantic Caching（语义缓存）**：相似查询延迟降低 10-100 倍
- **智能模型选择**：简单任务使用较小模型，复杂任务使用较大模型
- **工具优化**：自动选择相关工具以减少 token 使用

### 灵活性与可扩展性

- **Plugin 架构**：无需修改核心即可添加自定义处理逻辑
- **信号可扩展性**：为您的用例定义新的信号类型
- **配置驱动**：无需更改代码即可更改路由行为

## 使用场景

- **企业 API 网关**：具有安全性和合规性的智能路由
- **多租户平台**：每个租户的路由策略和模型选择
- **开发环境**：通过智能模型选择优化成本
- **生产服务**：具有全面监控的高性能路由
- **受监管行业**：具备 PII 检测和审计追踪的合规就绪方案

## 快速链接

- [**安装**](installation/installation.md) - 设置和安装指南
- [**概览**](overview/goals.md) - 项目目标和核心概念
- [**配置**](installation/configuration.md) - 配置信号和路由决策
- [**教程**](tutorials/intelligent-route/keyword-routing.md) - 分步指南

## 文档结构

本文档分为以下部分：

### [概览](overview/goals.md)

了解我们的目标、 Semantic Router 概念、Collective Intelligence 和 Signal-Driven Decision。

### [安装与配置](installation/installation.md)

开始安装并学习如何配置信号、决策和插件。

### [教程](tutorials/intelligent-route/keyword-routing.md)

实施智能路由、semantic caching、内容安全和可观测性的分步指南。

## 贡献

我们欢迎贡献！详情请参阅 [贡献指南](https://github.com/vllm-project/semantic-router/blob/main/CONTRIBUTING.md)。

## 许可证

本项目基于 Apache 2.0 许可证授权 - 详情请参阅 [LICENSE](https://github.com/vllm-project/semantic-router/blob/main/LICENSE) 文件。

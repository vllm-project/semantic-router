---
title: NVIDIA Dynamo 的语义智能层
description: 提出语义请求路由与 NVIDIA Dynamo 基础设施级路由之间的分层集成。
created: 2025-10-09
status: 提案
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/proposals/nvidia-dynamo-integration.md"
  outdated: false
---

> **状态：** 提案 · **创建日期：** 2025-10-09

## 问题 {#problem}

语义路由和推理机队路由回答不同问题：

- Semantic Router 决定应由哪个逻辑模型和策略处理请求。
- NVIDIA Dynamo 决定应由哪个合格 worker 执行该模型请求。

把任一层当作另一层的替代都会丢失信息。Semantic Router 不跟踪 worker 本地 KV 状态或在线 decode 负载。worker 路由器不决定请求属于编码模型、小型通用模型，还是策略受限的模型池。

## 提案 {#proposal}

把 Semantic Router 放在网关边界，并将其所选提供商模型发送到服务同一模型名的 Dynamo frontend。

```mermaid
flowchart LR
  Client --> Gateway
  Gateway -->|"ExtProc"| Semantic["Semantic Router"]
  Semantic -->|"selected model"| Frontend["Dynamo frontend"]
  Frontend -->|"worker selection"| Worker["Inference worker"]
```

集成有意分层。组件不共享优化器，任一组件也不进入对方的内部状态。

## 职责 {#responsibilities}

| 层 | 职责 |
| --- | --- |
| Gateway | 接受客户端流量，调用 ExtProc，并转发结果请求。 |
| Semantic Router | 解析入口点，评估信号和策略，运行请求插件，并选择提供商模型。 |
| Dynamo frontend | 接受所选模型请求，并应用配置的 Dynamo 路由模式。 |
| Dynamo workers | 执行推理，并发布 Dynamo 路由所需的任何状态。 |

当配置策略阻断请求，或产生响应的插件返回结果时，Semantic Router 可以在 Dynamo 之前终止请求。否则，物理 worker 选择仍由 Dynamo 负责。

## 集成契约 {#integration-contract}

部署必须确立这些不变量：

1. Semantic Router 可以选择的每个模型都由目标 Dynamo frontend 服务。
2. 所选提供商模型映射到 Dynamo 期望的模型标识符。
3. 网关路由保留改写后的请求体和所需模型元数据。
4. 在启用网关路径之前，直接请求 Dynamo frontend 应成功。
5. 运营方可以跨网关、Semantic Router、Dynamo frontend 和 worker 关联同一请求。

模型别名和后端端点是部署数据。它们应位于规范提供商配置中，而不是信号或决策规则中。

语义分类仍是 Router 关注点。其规范配置留在模型目录下，而不是复制到 Dynamo 的 worker 路由配置中：

```yaml
global:
  model_catalog:
    modules:
      classifier:
        domain:
          enabled: true
```

得到的领域类别可以约束逻辑模型池；随后 Dynamo 选择服务所选模型的 worker。请求安全模块留在同一 Router 边界：

```yaml
global:
  model_catalog:
    modules:
      prompt_guard:
        enabled: true
        variant: mmbert32k
        threshold: 0.7
```

## 缓存边界 {#cache-boundary}

两层缓存彼此独立：

- Semantic Router 响应缓存可以在不运行推理的情况下回答请求。
- Dynamo 的 KV 感知路由可以在执行新推理请求时复用 token 前缀状态。

响应缓存命中不能说明 Dynamo 的 KV 状态。Dynamo KV 命中也不能说明两条请求之间的语义等价。指标和轨迹应分开这些事件。

## 失败与策略行为 {#failure-and-policy-behavior}

网关必须定义 ExtProc 失败是失败即开放还是失败即关闭。失败即开放路径可以保持可用性，但也可能绕过模型选择和安全策略。强制策略执行的部署应失败即关闭，或路由到显式受限的回退。

Semantic Router 不应从成功的语义决策推断 worker 健康。Dynamo 不应把语义置信度重新解释为 worker 容量。每一层在自己的边界内报告和处理失败。

## 范围 {#scope}

本提案覆盖：

- Semantic Router 与 Dynamo frontend 之间的请求路径；
- 逻辑模型到已服务模型的身份；
- 独立的响应缓存和 KV-cache 行为；以及
- 跨两层路由的可观测性。

它不定义 Dynamo 安装、worker 拓扑、KV 路由器调优或拆分服务配置。那些仍由 Dynamo 拥有，并按其自己的发布节奏变化。

## 验证 {#validation}

集成测试应证明：

- 每个已配置模型都可从 Dynamo frontend 看到；
- 在测试网关路由之前，每个模型的直接请求都能工作；
- 代表性请求选择预期逻辑模型；
- Dynamo 只分派到服务该模型的 worker；以及
- 失败即开放或失败即关闭行为与部署策略匹配。

性能主张需要对照同一模型池、流量样本、缓存状态和路由模式的可复现比较。没有此类证据，本提案不做延迟、质量或成本主张。

## 待决问题 {#open-questions}

- 一个 Dynamo frontend 应服务整个语义模型池，还是提供商绑定应指向独立 frontend？
- 哪个请求标识符在全部四层上稳定？
- 哪些策略类别需要失败即关闭行为？
- 当 Dynamo 和 Semantic Router 独立升级时，模型别名应如何版本化？

## 参考资料 {#references}

- [当前 Semantic Router 与 Dynamo 部署指南](../installation/k8s/dynamo)
- [NVIDIA Dynamo 文档](https://docs.nvidia.com/dynamo/latest/)
- [Semantic Router 系统概览](../overview/semantic-router-overview)

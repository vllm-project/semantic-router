---
sidebar_position: 6
title: 常见问题
description: 了解 vLLM Semantic Router 的定位、它与 llm-d 和 AI 网关的区别，以及如何衡量路由价值和排查错误路由。
translation:
  source_commit: "7f1b814c97035e96a3780a3b8780ea5d3b6a6b24"
  source_file: "docs/faq.md"
  outdated: false
---

# 常见问题

## 它是 AI 网关，还是其他系统？ {#is-it-an-ai-gateway-or-something-else}

它是模型调用的决策层，根据请求信号和策略选择模型或有界多模型工作流，网关负责
传输。Envoy 通过 ExtProc 执行决策。TLS 终止、提供方负载均衡和 Pod 调度由其他
组件负责。

架构见[系统概览](overview/semantic-router-overview)，多模型执行见
[Mixture of Models](overview/mom-model-family)。

## 如何与 Agent Harness 配合？ {#how-does-it-work-with-an-agent-harness}

Harness 管理 Agent 循环、工具和任务状态。它调用稳定模型入口，Router 应用对应
配方，推理后端执行所选模型路径。

[接入 Harness](installation/agent-harness)，或[让 Agent 安装 Router](installation/agent)。

## 如何衡量路由准确性和业务价值？ {#how-do-we-measure-routing-accuracy-and-business-value}

分类器准确率不足以回答这个问题。需要验证路由配方能否相对于强单模型基线，**保持回答质量，同时降低成本和延迟**。这需要配对证据，不能仅靠变更前后的个别观察。

仓库使用 **sr-bench**：固定用例、评分器版本、请求参数和价格；在同一批用例上运行最强单模型和当前混合模型；将质量与 `100 × (1 − candidate cost / baseline cost)` 一起阅读。解释结果时遵循两个原则：

- 基线是在**同一批用例**上观测到的最佳单模型，而不是逐题选择最优模型；质量相同时，选择总成本更低的模型。
- 报告分数时也报告覆盖率。缺失或未评分的结果不等于零；开发集上的小幅提升或观测不到差异，也不足以证明等价。

[sr-bench 1.0](benchmarking/sr-bench)定义了从固定用例到解读结果的完整测量协议。

## 为什么有 llm-d 还需要 Semantic Router？ {#why-semantic-router-and-not-just-llm-d}

两个系统解决的问题不同。

| | Semantic Router | llm-d |
| --- | --- | --- |
| 决定什么 | 使用哪个逻辑模型或**模型池** | 在该池中使用哪个健康**副本** |
| 读取什么 | 请求内容、策略、语义证据 | 负载、前缀缓存局部性、副本健康状态 |
| 所在层 | 控制面的决策 | 调度与放置 |

llm-d 不应决定业务策略，Semantic Router 也不负责选择 Pod。项目的规则是：**不要配置两个系统来做同一个决策。**

[llm-d 集成指南](installation/k8s/llm-d)在部署上下文中解释了这个边界。

## 如何避免与 llm-d 的决策冲突？ {#how-do-we-avoid-conflicts-with-llm-d}

通过划分职责避免冲突。Semantic Router 先确定模型池，llm-d 再在池内选择副本。“哪个模型适合请求”与“哪个副本适合缓存局部性”处于不同层级；前一个决策约束后一个决策。

运维时遵循两条规则：

- **先独立部署和验证 llm-d，再加入 Semantic Router**；使用一个受支持的 llm-d 版本，不混用不同版本的清单。
- **让最终决策可见。** Router 将 `x-vsr-selected-model` 作为回执，将 `x-selected-model` 作为网关匹配值；`vllm-sr route probe --expect-selected-model` 用于断言回执。应读取这些值，而不是事后推测决策。

部署与决策规则见[集成 llm-d](installation/k8s/llm-d)。

## 如何避免影响多轮或 Agent 工作负载？ {#how-do-we-avoid-hurting-multi-turn--agentic-workloads}

在对话中切换模型可能造成前缀缓存损失、工具行为变化和连续性中断。应把切换作为明确的权衡，而不是默认优化：

- **连续性重要时优先保持模型。** 会话范围的学习默认保护已有选择（内置初始化默认开启保护、关闭在线适应），前提是 Router 能识别对话。默认 `scope: conversation` 要求同时提供 `x-session-id` 和 `x-conversation-id`；只发送一个请求头不会保持模型。只有显式配置 `scope: session` 时，才只需 session 标识。
- **只按明确的升级策略切换。** 仅仅“更便宜”不足以成为迁移对话的理由。
- **验证连续性，而不仅是请求送达。** 检查续接、工具完成、纠正、模型故障和对话重置时实际使用的模型。观测到的建议不等于已执行的保持策略。

[会话标识](api/session-identification)定义请求头契约，[配方](tutorials/global/recipes)介绍保护和升级策略所属的配置范围。

## 运维人员如何排查错误结果？ {#how-do-operators-debug-a-bad-outcome}

沿决策链查找第一个不符合预期的阶段。每一阶段都有对应的观察方式：

```text
signal → policy / decision → algorithm / model → plugin → endpoint scheduling → fallback → final model
```

1. **先预览，再生成。** `vllm-sr route preview` 不调用后端，评估信号、投影和决策，并报告追踪证据。
2. **读取决策来源。** 启用学习的预览会暴露 `selection_provenance`，包括配置与状态身份以及采样种子。
3. **再做真实探测。** `vllm-sr route probe` 通过 Envoy 发送真实请求，但成功的 HTTP 状态还不够；应检查 `response.body.delivery` 和最终助手输出，因为空输出、只有推理内容的输出，以及 `finish_reason: length` 都不能算作成功交付。
4. **单独检查 UI 路径。** 在控制面板中测试同一入口，并分别报告纯 API 覆盖和 UI 覆盖。

交付、路由正确性和回答质量是三个独立结果。明确说明未测量的部分，不要据此推断成功。

[CLI 参考](api/cli)说明这里使用的全部参数，[VSR 标头](troubleshooting/vsr-headers)列出回执头，[API 与可观测性](tutorials/global/api-and-observability)介绍遥测方式。

---
title: 路由学习：自我改进的模型路由
description: 记录已实现的在线适配、路由保护和离线配方学习契约。
created: 2026-06-20
status: Implemented
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/proposals/router-learning-memory-and-adaptations.md"
  outdated: false
---

> **状态：** 已实现 · **创建日期：** 2026-06-20

## 问题 {#problem}

语义决策评估当前请求。它本身不会记住某模型在相似运行中是否不合适，或切换模型是否会破坏对话、工具循环或前缀缓存连续性。

把该状态放进决策规则会使策略不透明并依赖副本。因此路由学习在已匹配决策和基础选择器之后运行。配方仍是策略边界。

## 已实现设计 {#implemented-design}

路由学习有三项职责：

| 组件 | 用途 | 时间尺度 |
| --- | --- | --- |
| 自适应 | 根据有界运行时经验提出模型。 | 请求路径。 |
| 防护 | 决定探索或模型切换是否安全。 | 请求路径。 |
| 配方 learning | 分析回放和结果，并提出可审阅的配方变更。 | 离线。 |

```text
matched decision and base selector
  -> protection preflight
  -> adaptation proposal
  -> protection switch guard
  -> final model
  -> replay and outcome updates
```

适配可以提出不同模型。保护对提议是否成为所选模型有最终决定权。

## 公开配置边界 {#public-configuration-boundary}

| 表面 | 含义 |
| --- | --- |
| `global.router.learning.enabled` | 启用路由学习流水线。 |
| `global.router.learning.adaptation` | 选择在线模型选择行为。 |
| `global.router.learning.protection` | 配置对话或会话稳定性。 |
| `global.router.learning.state_store` | 可选地跨副本共享保护状态。 |
| `routing.decisions[].adaptations` | 对单个决策应用、观察或绕过学习。 |

已实现的适配策略是 `routing_sampling`。历史算法名称不是该表面的别名。决策保持语义，并保留其现有选择算法。

## 候选集 {#candidate-sets}

适配只在配置的候选集中搜索：

| 值 | 候选模型 |
| --- | --- |
| `decision` | 已匹配决策的 `modelRefs` 中的模型。 |
| `tier` | 已匹配决策层级中各决策的模型。 |
| `global` | 已部署配方清单中的模型。 |

`decision` 是较窄的默认值。更广范围仍遵守提供商可用性、成本和可靠性保护，以及决策级绕过。

## 保护 {#protection}

保护在对话或会话身份内保持模型稳定。预检保护在协议敏感步骤中抑制不安全的随机探索。切换保护权衡提议收益与缓存、交接、工具循环和切换历史成本。

若缺少所需身份头，保护失败即开放并记录诊断。敏感决策可以设置 `adaptations.mode: bypass`，从而阻止适配和保护改变基础选择。`observe` 计算诊断而不改变最终模型。

## 经验与结果 {#experience-and-outcomes}

经验是证据，不是策略。它可以包括显式结果标签、失败信号、延迟、有效成本、缓存复用和可靠性观察。策略用该有界证据为候选打分或采样。

结果必须附着到稳定的回放标识符，并记录基础、提议和最终模型。延迟或重复结果需要幂等处理。在线经验键不需要原始请求内容。

## 状态与失败行为 {#state-and-failure-behavior}

保护状态可以使用有界本地存储和可选的共享 Redis 存储。请求路径读取使用严格超时。远程存储失败不得使推理请求依赖无界网络调用。

详细候选分数、身份哈希、切换成本和证据属于路由回放。响应头保持紧凑，只描述请求级检查所需的方法、动作、范围和原因码。

## 离线配方学习 {#offline-recipe-learning}

离线循环消费回放、结果和可选评估用例。它产生发现、指标、候选变体、补丁建议和可选种子产物。它不会自动编辑或部署活动配方。

这种分离使配方变更可审阅，并让运营方在提升前复现实验。

## 范围与非目标 {#scope-and-non-goals}

路由学习不：

- 重新匹配语义决策；
- 覆盖决策级绕过；
- 扩展到配置候选集之外；
- 同步改写已部署配方；
- 使在线请求路径依赖 LLM 智能体；或
- 把会话亲和当作授权的替代。

## 评估 {#evaluation}

在评估组合流水线之前，先独立评估适配和保护。报告路由质量、切换率、后悔或适配结果、延迟、缓存效应、失败恢复，以及在声明身份内的稳定性。在启用模型变更前，在同一回放样本上比较 `apply` 与 `observe`。

## 待决问题 {#open-questions}

- 更广的 `tier` 或 `global` 候选集何时值得额外风险？
- 哪些结果来源足够可靠，可以更新在线经验？
- 过期经验应如何随模型或提示词模板版本衰减？
- 离线种子产物何时应导入在线部署？

## 参考资料 {#references}

- [路由学习概览](../tutorials/learning/overview)
- [适配](../tutorials/learning/adaptations)
- [保护](../tutorials/learning/protection)
- [决策级控制](../tutorials/learning/decision-adaptations)
- [记忆与回放](../tutorials/learning/memory-and-replay)

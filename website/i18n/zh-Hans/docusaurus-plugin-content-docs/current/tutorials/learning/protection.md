---
translation:
  source_commit: "ee350c76c2d43e884fe8db7af1ba1aba539588c3"
  source_file: "docs/tutorials/learning/protection.md"
  outdated: true
---

# 防护

## 概览

防护在不把连续性做成语义路由的情况下，保持智能体对话稳定。每个请求仍先经过正常决策路由。自适应提出模型后，防护决定是保持当前模型、允许切换，还是执行有界救援切换。

## 主要优势

- 在智能体对话或整段会话内保持模型选择稳定。
- 保护前缀缓存、工具循环连续性和交接成本。
- 在协议敏感步骤中抑制随机探索。
- 在证据足够强时，仍允许确定性切换和有界救援。
- 让敏感决策通过决策局部控制绕过防护。

## 解决什么问题？

智能体请求并非彼此独立。工具调用、提供商状态、前缀缓存和用户可见的连续性，可能让不必要的模型切换变得昂贵或令人困惑。防护为路由器提供有范围的稳定性守卫，而不把会话连续性变成语义决策规则。

## 何时使用

- 对话应继续使用同一模型，除非切换值得付出稳定性成本。
- 整段会话应在多次用户发起的运行中保持稳定。
- 工具循环或协议状态使随机探索不安全。
- 较弱的受保护模型仍应能通过有界救援退出。

## 配置

```yaml
global:
  router:
    learning:
      enabled: true
      adaptation:
        enabled: false
      protection:
        enabled: true
        scope: conversation
        identity:
          headers:
            session: x-session-id
            conversation: x-conversation-id
        tuning:
          idle_timeout_seconds: 300
          min_turns_before_switch: 1
          switch_margin: 0.05
          stability_weight: 1.0
```

此配置独立启用防护，不启用在线模型选择自适应。若仅打开总开关而省略两个组件的开关，它们默认都会启用。

## 范围 {#scopes}

| 范围 | 保护什么 | 什么可以重新路由 |
| --- | --- | --- |
| `conversation` | 共享同一 `x-conversation-id` 的轮次。 | 同一 `x-session-id` 中的新 `x-conversation-id`。 |
| `session` | 共享同一 `x-session-id` 的轮次。 | 空闲超时，或带 `adaptations.mode: bypass` 的决策。 |

当每次智能体运行应独立路由时，使用 `conversation`。当一次会话级模型选择应在多次用户发起的运行中保持稳定时，使用 `session`。在 `conversation` 范围内，决策变化会重置最少轮次和连续性成本偏好。在 `session` 范围内，仅改变决策或对话不会释放仍合格的当前模型；空闲超时、绕过防护、候选排除或满足条件的救援可以释放它。任何范围都不能在合格候选集之外保留先前模型。

若配置的身份请求头缺失，防护会失败开放并记录诊断，而不是让请求失败。

## 守卫 {#guards}

防护有两个守卫点：

- **preflight** 在工具/协议/例行延续步骤中抑制随机采样。
- **switch guard** 使用缓存、交接、工具循环、会话和切换历史成本，接受或拒绝自适应提议的模型。

切换规则是：

```text
switch if proposal_gain >= switch_margin + stability_weight * switch_cost
```

后端重复失败，或关联 Replay 的结果表明当前模型能力不足时，防护还可以允许确定性的 `rescue_switch`。防护身份齐备时，即使自适应关闭，后端 `429` 和 `5xx` 响应也会提供失败观测；这不会启用自适应模型选择或质量更新。纠错或重试提示本身不算归属于模型的结果证据，但它的信号可能改变匹配的决策。救援需要另一个合格的提议模型以及足够证据，并不等于立即后端故障转移。

救援仅能在上下文可移植的轮次边界选择合格模型，不能覆盖活动工具循环或不可移植上下文的锁定。不存在这些硬边界时，最少轮次和会话连续性偏好可让位于救援。自适应与防护也保留决策选择器施加的候选限制，包括词典序容差范围。

防护处于 `apply` 模式且所配置的身份齐备时，若硬锁定的原模型不在合格候选池内，请求返回 `503`，不会转移所有权或强行使用该模型。Router 自有 Responses 历史展开为完整无状态请求后，上下文可移植；保留的 `previous_response_id` 本身不构成锁定，但活动工具循环仍会锁定。未知响应 ID 在读取历史时失败，不进入模型选择。

## 决策边界 {#decision-boundaries}

大多数决策不需要局部配置。硬策略边界使用 `bypass`：

```yaml
routing:
  decisions:
    - name: local_privacy_policy
      description: Keep privacy-sensitive traffic on the local model.
      priority: 200
      modelRefs:
        - model: local-private-model
      adaptations:
        mode: bypass
```

使用 `observe` 收集诊断而不更改最终模型：

```yaml
adaptations:
  protection:
    mode: observe
```

## 诊断 {#diagnostics}

```http
x-vsr-learning-methods: protection
x-vsr-learning-actions: protection=hold_current
x-vsr-learning-scopes: protection=conversation
x-vsr-learning-reasons: protection=cache_cost_high
```

客户端 UI 应将原始 action 翻译成面向用户的文本。例如，`hold_current` 可显示为“保持运行模型”，`allow_switch` 为“允许切换”，`rescue_switch` 为“救援切换”，`bypass` 为“已绕过学习”。

路由回放存储完整防护 trace：身份来源和哈希、受保护模型、基础模型、提案模型、最终模型、切换成本、缓存证据、工具循环状态、模式、范围、action 和原因。

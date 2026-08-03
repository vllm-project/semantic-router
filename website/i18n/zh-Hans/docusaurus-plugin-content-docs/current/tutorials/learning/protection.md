---
translation:
  source_commit: "ad233487"
  source_file: "docs/tutorials/learning/protection.md"
  outdated: false
---

# 保护机制

## 概览

Protection 可在不将连续性变成语义路由的情况下，使 agent 对话保持稳定。每个请求仍会先经过正常的 decision 路由。adaptation 提出模型后，Protection 会决定是保持当前模型、允许切换，还是执行有界的救援切换。

## 主要优势

- 在 agent 对话或整个 session 内保持模型选择稳定。
- 保护前缀缓存、工具循环连续性和交接成本。
- 在协议敏感步骤中抑制随机探索。
- 在证据足够强时，仍允许确定性切换和有界救援。
- 允许敏感 decision 通过 decision 局部控制绕过 Protection。

## 解决什么问题？

Agent 请求并非相互独立。工具调用、提供商状态、前缀缓存和用户可感知的连续性，都会让不必要的模型切换代价高昂或令人困惑。Protection 为路由器提供限定作用域的稳定性守卫，同时不会将 session 连续性变成语义 decision 规则。

## 何时使用

- 除非切换值得付出稳定性成本，否则对话应继续使用同一个模型。
- 完整 session 应在多次由用户发起的运行之间保持稳定。
- 工具循环或协议状态使随机探索变得不安全。
- 即使受保护模型能力较弱，也仍应能通过有界救援切换脱离该模型。

## 配置

```yaml
global:
  router:
    learning:
      enabled: true
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

## 作用域

| 作用域 | 保护对象 | 可触发重新路由的条件 |
| --- | --- | --- |
| `conversation` | 共享同一个 `x-conversation-id` 的回合。 | 新的 `x-conversation-id` 出现在同一 `x-session-id` 中。 |
| `session` | 共享同一个 `x-session-id` 的回合。 | 空闲超时，或某个 decision 设置了 `adaptations.mode: bypass`。 |

当每次 agent 运行都应独立路由时，请使用 `conversation`。当一个 session 级模型选择应在多次由用户发起的运行之间保持稳定时，请使用 `session`。

如果缺少已配置的身份 header，Protection 会采用 fail-open 行为并记录诊断，而不是令请求失败。

## 守卫

Protection 有两个守卫点：

- **preflight** 在工具、协议或例行延续步骤中抑制随机采样。
- **switch guard** 根据缓存、交接、工具循环、session 和切换历史成本，接受或拒绝 adaptation 提出的模型。

切换规则如下：

```text
switch if proposal_gain >= switch_margin + stability_weight * switch_cost
```

当重复失败、重试、验证失败或显式结果证据表明当前模型能力不足时，Protection 还可以允许确定性的 `rescue_switch`。

## Decision 边界

大多数 decision 不需要局部配置。对于硬策略边界，请使用 `bypass`：

```yaml
routing:
  decisions:
    - name: local_privacy_policy
      modelRefs:
        - model: local-private-model
      adaptations:
        mode: bypass
```

使用 `observe` 可在不更改最终模型的情况下收集诊断：

```yaml
adaptations:
  protection:
    mode: observe
```

## 诊断

```http
x-vsr-learning-methods: protection
x-vsr-learning-actions: protection=hold_current
x-vsr-learning-scopes: protection=conversation
x-vsr-learning-reasons: protection=cache_cost_high
```

客户端 UI 应将原始 action 转换为面向用户的文本。例如，`hold_current` 可显示为“保留本次运行的模型”，`allow_switch` 可显示为“允许切换”，`rescue_switch` 可显示为“救援切换”，而 `bypass` 可显示为“已绕过学习”。

Router Replay 会存储完整的 Protection 追踪信息：身份来源及哈希、受保护模型、基础模型、提议模型、最终模型、切换成本、缓存证据、工具循环状态、模式、作用域、动作和原因。

---
translation:
  source_commit: "ad233487"
  source_file: "docs/tutorials/learning/overview.md"
  outdated: true
---

# Router Learning

## 概览

Router Learning 是实现跨请求路由智能的路由器层。它会调整语义 decision 所提出的模型，但不会让在线状态成为 `decision.algorithm` 的一部分。

公开概念包括：

- `global.router.learning.adaptation`：在线模型选择学习。
- `global.router.learning.protection`：session 和 conversation 稳定性。
- `routing.decisions[].adaptations`：按 decision 配置 apply、observe 或 bypass 控制。
- Router Replay：用于离线配方学习的持久诊断与结果。

当 decision 应保持语义化，但重复请求需要考虑当前模型、工具循环状态、prefix cache 证据、handoff 成本、切换历史或运行时结果时，请使用 Router Learning。

## 主要优势

- 保持语义 decision 清晰易读，并且只依赖当前请求。
- 让在线模型选择学习和稳定性保护共享同一条运行时流水线。
- 允许硬策略 decision 绕过学习，而无需更改路由规则。
- 记录紧凑的响应 header 和详细的 Router Replay 诊断。
- 为离线 agent 循环提供数据，以发现路由问题并提出配方补丁。

## 解决什么问题？

语义 decision 擅长匹配当前请求，但不会记住模型在类似 agent 流程中是否配置过高、能力不足、不稳定或成本过高。Router Learning 增加有界在线状态和与 replay 关联的结果，使路由器能在配方保持控制权的同时改进模型选择。

## 何时使用

- 配方包含多个候选模型，并且运行时证据应改进模型选择。
- Agent session 需要在工具循环、prefix cache 或提供商状态变化时保持稳定。
- 敏感 decision 需要显式绕过在线学习。
- 希望使用 replay 和结果驱动离线配方实验。

## 配置

```yaml
global:
  router:
    learning:
      enabled: true
      adaptation:
        enabled: true
        strategy: routing_sampling
        candidate_set: decision
      protection:
        enabled: true
        scope: conversation
        identity:
          headers:
            session: x-session-id
            conversation: x-conversation-id
        tuning:
          idle_timeout_seconds: 300
          switch_margin: 0.05
          stability_weight: 1.0
```

Decision 局部控制是稀疏配置。大多数 decision 会继承全局行为：

```yaml
adaptations:
  mode: bypass
```

隐私、安全、仅本地、合规或任何其他硬策略路由应使用 `bypass`。当某个组件需要独立 observe 或 bypass 时，请使用组件级控制：

```yaml
adaptations:
  adaptation:
    mode: observe
  protection:
    mode: apply
    stability_weight: 1.5
```

## 运行时流程

```text
base selector
  -> protection preflight
  -> adaptation
  -> protection switch guard
  -> final model
```

Adaptation 根据经验判断哪个模型表现更好。Protection 则判断当前进行探索或切换是否安全。

## Header 与 Replay

`x-vsr-learning-*` header 族有意保持紧凑：

```http
x-vsr-learning-methods: adaptation,protection
x-vsr-learning-actions: adaptation=propose_switch,protection=allow_switch
x-vsr-learning-scopes: protection=conversation
x-vsr-learning-reasons: adaptation=sampled_win,protection=switch_allowed
```

Base model、proposal model、final model、cache warmth、switch cost、candidate score、sampling value 和经过哈希处理的 identity 诊断等详细字段应存放在 Router Replay 中，并以 `x-vsr-replay-id` 为键。

## 相关页面

- [Adaptation](./adaptations) 介绍 `routing_sampling` 和候选集。
- [Protection](./protection) 介绍 conversation 和 session 稳定性。
- [Decision Adaptations](./decision-adaptations) 介绍 decision 局部控制。
- [Memory And Replay](./memory-and-replay) 介绍诊断与结果。

## 离线配方学习

Router Learning 不会在请求路径上重写已部署的配方。请使用离线配方学习命令，将 replay 和结果转化为发现、指标、候选配方变体、实验估算、配方补丁建议以及经验 seed pack：

```bash
vllm-sr eval recipe-learning \
  --endpoint http://localhost:8080 \
  --recipe-file config.yaml \
  --output-dir ./router-learning-report
```

对于隔离网络或 CI 工作流，请先导出 replay JSON，再通过 `--replay-file` 传入。当评估用例包含预期 decision 或模型时，请添加 `--cases-file`。

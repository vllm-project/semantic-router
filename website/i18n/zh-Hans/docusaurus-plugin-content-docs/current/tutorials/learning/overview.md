---
translation:
  source_commit: "e56591a9cb24f073bf159927e87116ba6d278741"
  source_file: "docs/tutorials/learning/overview.md"
  outdated: false
---

# 路由学习

## 概览

路由学习是跨请求路由智能的路由器层。它调整语义决策提议的模型，而不把在线状态变成 `decision.algorithm` 的一部分。

公开概念是：

- `global.router.learning.adaptation`：在线模型选择学习。
- `global.router.learning.protection`：会话和对话稳定性。
- `routing.decisions[].adaptations`：按决策的应用、观察或绕过控制。
- 路由回放：可选诊断和结果，用于离线配方学习；持久化需要耐用的回放后端。

当决策应保持语义，但重复请求应考虑当前模型、工具循环状态、前缀缓存证据、交接成本、切换历史或运行时结果时，使用路由学习。

## 主要优势

- 让语义决策保持可读且请求局部。
- 为在线模型选择学习和稳定性防护提供一条共享运行时管道。
- 让硬策略决策在不更改路由规则的情况下绕过学习。
- 记录紧凑的响应头，并在启用回放时记录详细的路由回放诊断。
- 支持离线分析，可在部署前识别路由问题并评估配方变更。

## 解决什么问题？

语义决策擅长匹配当前请求，但它们不记得某个模型在类似智能体流程中是否过度配置、能力不足、不稳定或昂贵。路由学习增加有界在线状态和与回放关联的结果，使路由器可以改进模型选择，同时仍由配方掌控。

## 何时使用

- 你的配方有多个候选模型，且运行时证据应改进选择。
- 智能体会话需要在工具循环、前缀缓存或提供商状态之间保持稳定。
- 敏感决策需要显式绕过在线学习。
- 希望用显式配置的回放和结果驱动离线配方实验。

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
      state_store:
        backend: redis
        ttl_seconds: 86400
        timeout_ms: 50
        redis:
          address: redis:6379
          database: 2
          key_prefix: "vsr:router-session:v1:"
```

共享存储是可选的。请求时读取使用严格超时，并失败开放到有界本地存储。响应侧更新会把同一快照写入 Redis，以便另一副本恢复对话防护状态。

决策局部控制应保持稀疏。大多数决策继承全局行为：

```yaml
adaptations:
  mode: bypass
```

对隐私、安全、仅本地、合规或其他硬策略路由使用 `bypass`。当某个组件应独立观察或绕过时，使用组件级控制：

```yaml
adaptations:
  adaptation:
    mode: observe
  protection:
    mode: apply
    stability_weight: 1.5
```

## 运行时流程 {#runtime-flow}

```text
base selector
  -> protection preflight
  -> adaptation
  -> protection switch guard
  -> final model
```

自适应回答根据经验哪个模型看起来更好。防护回答当前探索或切换是否安全。

## 请求头与回放 {#header-and-replay}

`x-vsr-learning-*` 请求头族有意保持紧凑：

```http
x-vsr-learning-methods: adaptation,protection
x-vsr-learning-actions: adaptation=propose_switch,protection=allow_switch
x-vsr-learning-scopes: protection=conversation
x-vsr-learning-reasons: adaptation=sampled_win,protection=switch_allowed
```

启用路由回放时，基础模型、提案模型、最终模型、缓存热度、切换成本、候选分数、采样值和哈希身份诊断等详细字段会存储在那里，并以 `x-vsr-replay-id` 为键。

## 相关页面 {#related-pages}

- [自适应](./adaptations) 解释 `routing_sampling` 和候选集。
- [防护](./protection) 解释对话和会话稳定性。
- [决策自适应](./decision-adaptations) 解释决策局部控制。
- [记忆与回放](./memory-and-replay) 解释诊断和结果。

## 离线评估配方变更 {#evaluate-recipe-changes-offline}

路由学习不会在请求路径上改写已部署的配方。使用离线配方学习命令，将回放和结果转化为发现、指标、候选变体、实验估计、建议变更和经验种子包：

```bash
vllm-sr optimize recipe-learning \
  --endpoint http://localhost:8080 \
  --recipe-file config.yaml \
  --output-dir ./router-learning-report
```

`--endpoint` 指向 Router 管理 API，绝不是公开推理监听器。启用管理 bearer 认证时，请先导出 `VSR_MGMT_TOKEN`。

对于隔离或 CI 工作流，先导出回放 JSON，再用 `--replay-file` 传入。当评估用例包含预期决策或模型时，添加 `--cases-file`。

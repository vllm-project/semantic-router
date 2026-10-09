---
translation:
  source_commit: "9156d5bc1ed9edff626b95a2b8260a77cb1712c5"
  source_file: "docs/tutorials/algorithm/selection/decision.md"
  outdated: false
---

# Decision 选模算法

`decision` 算法使用判断模型的 `choice` 能力，从当前决策的 `modelRefs` 中选择后端。每个候选都有描述；结果包含候选概率，不需要生成文本再解析模型名。

## 配置

```yaml
routing:
  decisions:
    - name: general
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: fast-model
        - model: reasoning-model
      algorithm:
        type: decision
        decision:
          instructions: Which model should answer this request?
          candidates:
            fast-model: Routine requests that benefit from a quick response
            reasoning-model: Requests requiring advanced reasoning
          timeout_ms: 1000
```

请先在 `providers.models` 声明这些后端，并为决策配置适合应用的匹配条件。未设置 `deployment` 时，算法使用 `global.model_catalog.system.decision_model` 指向的部署，默认是 Vela 2.0 0.3B。可在 `algorithm.decision.deployment` 显式选择另一个部署。Vela 或 Decision 1.0/2.0 均可使用，前提是运行时报告支持 choice 任务。

## 执行与失败

选模发生在决策匹配之后，是针对候选集的独立调用。复用判断模型不等于复用请求阶段的答案，也不保证整个请求只有一次 forward。模型不可用、超时或答案无效时，选择器退回第一个 `modelRef`；应把合适的兜底候选放在首位。

任务能力表示契约可执行，不保证选模效果。使用代表性请求评估候选描述、错误选择率及额外时延；固定选择使用 `static`，基于成本、延迟和负载的选择可用 `multi_factor`。

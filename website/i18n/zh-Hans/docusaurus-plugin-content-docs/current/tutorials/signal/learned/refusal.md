---
is_mtpe: true
translation:
  source_commit: "72b3d17c5fdcaa506e603b1697fdb2d4b6318a6e"
  source_file: "docs/tutorials/signal/learned/refusal.md"
  outdated: false
---

# 拒答信号 {#refusal-signal}

## 概览 {#overview}

`refusal` 询问决策模型：模型的回答是否拒绝了请求。它读取回答，所以拒答规则只在模型作答之后存在：由响应插件读取，决策规则不会读取它。

它对应 config/fragments/signal/refusal，在 `recipes[].signals.refusal` 下声明。

## 主要优势 {#key-advantages}

- 无需单独的分类器即可发现被拒绝的回答。
- 与其他响应阶段信号使用同一个决策模型、同一次调用。
- 默认值不合适时，用 predicate 设定工作点。

## 解决什么问题？ {#what-problem-does-it-solve}

即使请求路由正确，拒绝了正常请求的回答也是失败的回答。`refusal` 标出这种结果，让响应插件据此处理。

## 何时使用 {#when-to-use}

在以下情况使用 `refusal`：

- 响应插件需要对拒绝请求的回答做出反应
- 需要统计某条路由的模型拒答的频率

## 配置 {#configuration}

```yaml
recipes:
  - name: default
    signals:
      refusal:
        - name: declined
          description: The answer declines the request or refuses to help.
          predicate:
            gte: 0.5
```

| 字段 | 含义 |
| --- | --- |
| `name` | 规则名称 |
| `description` | 规则检测的内容 |
| `predicate` | `refusal` 选项概率的界限；不写时使用作答模型校准的阈值，否则为 0.5 |
| `deployment` | 回答该规则的部署；否则使用 `model_bindings.refusal`，再否则使用决策模型 |
| `timeout_ms` | 该问题的时限 |

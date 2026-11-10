---
is_mtpe: false
translation:
  source_commit: "118fb72eb67a0e7193697e350d7b494dc016631e"
  source_file: "docs/tutorials/signal/application/refusal.md"
  outdated: false
---

# 拒答信号 {#refusal-signal}

## 概述 {#overview}

`refusal` 询问决策模型，模型的答案是否拒绝了请求。它在响应阶段提出 `refusal` [预设](../../../system-one/reference#preset-catalog)问题：“Did the assistant refuse?”，选项为 `compliance` 和 `refusal`。它读取答案，所以拒答规则只在模型回答之后才存在：由响应插件读取，决策的规则永远不读取它。

## 主要优势 {#key-advantages}

- 无需单独的分类器就能识别拒答。
- 与其他响应阶段信号使用同一个决策模型和同一次调用。
- 默认值不合适时，用 predicate 设置工作点。

## 解决什么问题？ {#what-problem-does-it-solve}

即使请求路由正确，拒绝一个正常请求的答案也是失败的答案。`refusal` 给这种结果命名，让响应插件可以据此行动。[`escalate`](../../algorithm/looper/escalate) 提出同一个问题，以便在候选拒答时转向下一个（`accept.refusal: false`）。

## 何时使用 {#when-to-use}

在以下情况使用 `refusal`：

- [`response_guard`](../../plugin/response-guard) 应对拒绝请求的答案加注说明或拦截；
- 你要统计某条路由的模型拒答的频率。

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
    decisions:
      - name: assistant
        priority: 100
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: answer-model
        plugins:
          - type: response_guard
            configuration:
              rules:
                - when: {type: refusal, name: declined}
                  action: annotate
```

| 字段 | 含义 |
| --- | --- |
| `name` | 规则名称，供防护规则以 `{type: refusal, name: ...}` 使用 |
| `description` | 规则检测的内容 |
| `predicate` | `refusal` 的最低概率；默认为回答模型的校准阈值，否则为 0.5 |
| `deployment` | 回答规则的部署；否则为 `refusal` 绑定，再否则为决策模型 |
| `timeout_ms` | 答案的截止时间 |

这个问题会把请求和答案发给部署。vela3-fast 和 Vela 3.0 的 d3 规格有它的训练形式；其他模型零样本回答。完整示例见 [`config/fragments/signal/refusal/refusal.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/refusal/refusal.yaml)。

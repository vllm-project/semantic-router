---
is_mtpe: false
translation:
  source_commit: "118fb72eb67a0e7193697e350d7b494dc016631e"
  source_file: "docs/tutorials/signal/application/complexity.md"
  outdated: false
---

# 复杂度信号 {#complexity-signal}

## 概述 {#overview}

`complexity` 评估一个请求要回答好有多难：easy、medium 或 hard。它提出 `complexity` [预设](../../../system-one/reference#preset-catalog)问题：“How difficult is this request to answer well?”，选项为 `easy`（简单的检索、改写或格式调整）、`medium`（若干步推理，或适量的代码或分析）和 `hard`（高级的多步推理或专业知识）。

## 主要优势 {#key-advantages}

- 把难的请求交给推理模型，简单的交给快速模型。
- 不需要示例库，也不需要拟合阈值。
- 每个请求只问一个问题，就能回答所有复杂度规则。

## 解决什么问题？ {#what-problem-does-it-solve}

把每个请求都交给最强的模型，会把它浪费在简单的工作上；把每个请求都交给最便宜的模型，难题就会失败。`complexity` 在选择模型之前把两者区分开。

## 何时使用 {#when-to-use}

当决策只应为难的请求选择更强的模型或开启推理时，使用 `complexity`。与 [domains](./domain) 结合，可以构成“难的数学”或“简单的代码”这样的路由。需要更细的刻度时，编写你自己的 [`systemone`](../systemone) `score` 问题。

## 配置 {#configuration}

```yaml
recipes:
  - name: default
    signals:
      domains:
        - name: math
      complexity:
        - name: difficulty
        - name: needs_reasoning
          description: Multi-step reasoning or synthesis-heavy prompts.
          labels: [hard]
    decisions:
      - name: hard-math
        priority: 200
        rules:
          operator: AND
          conditions:
            - {type: domain, name: math}
            - {type: complexity, name: difficulty, label: hard}
        modelRefs:
          - model: reasoner
      - name: easy-anything
        priority: 100
        rules:
          operator: AND
          conditions:
            - {type: complexity, name: difficulty, label: easy}
        modelRefs:
          - model: fast-model
```

| 字段 | 含义 |
| --- | --- |
| `name` | 规则名称，供 `{type: complexity, name: ...}` 条件使用 |
| `labels` | 规则匹配的等级，取自 `easy`、`medium` 和 `hard`；没有 `labels` 的规则本身不匹配任何内容，由它的条件用 `label` 指定一个等级 |
| `deployment`、`timeout_ms` | 回答问题的部署，以及答案的截止时间 |

当模型选中的等级属于规则的 `labels` 时，规则匹配；带 `label` 的条件在模型选中该等级时匹配。条件可以对该等级的概率加上 `predicate`。

### 回答问题的模型 {#the-model-that-answers}

除非 `complexity` 绑定或规则的 `deployment` 指定了其他部署，否则由决策模型回答所有复杂度规则；见[选择部署](../../../system-one/deployments)。目前还没有模型有这个问题的训练形式，所以所有模型都零样本回答：在它约束高成本路由之前，请在你自己的流量上检查它。它读取当前用户轮次，并截断到模型的输入长度。

为早期版本编写的配置用阈值把请求与示例库比较（`hard`、`easy`、`threshold`、`composer` 等），并把等级写成 `difficulty:hard`。`vllm-sr config migrate` 会改写两者：规则保留名称，条件用 `label` 指定等级。完整示例见 [`config/fragments/signal/complexity/escalation.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/complexity/escalation.yaml)。

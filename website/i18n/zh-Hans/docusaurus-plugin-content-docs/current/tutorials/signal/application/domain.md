---
is_mtpe: false
translation:
  source_commit: "118fb72eb67a0e7193697e350d7b494dc016631e"
  source_file: "docs/tutorials/signal/application/domain.md"
  outdated: false
---

# 领域信号 {#domain-signal}

## 概述 {#overview}

`domains` 按请求所属的学科路由，学科由决策模型从 14 个中选出。每条规则指定它匹配的学科。它提出 `domain` [预设](../../../system-one/reference#preset-catalog)问题：“Which subject area is this request about?”，选项为 `biology`、`business`、`chemistry`、`computer science`、`economics`、`engineering`、`health`、`history`、`law`、`math`、`other`、`philosophy`、`physics` 和 `psychology`。

## 主要优势 {#key-advantages}

- 按主题路由，无需为每个主题维护关键词列表。
- 每个请求只问一个问题，就能回答所有领域规则。
- 可以把多个学科归到一个路由名称下。
- 为 `static` 选择器提供按领域的模型得分。

## 解决什么问题？ {#what-problem-does-it-solve}

路由常常按学科组织：数学模型、法律审阅者、编程助手。为每个学科维护关键词列表既脆弱又容易漏掉换种说法的请求；决策模型阅读请求并给出它的学科。

## 何时使用 {#when-to-use}

在以下情况使用 `domains`：

- 路由按学科组织；
- 词法匹配对主题边界来说过于脆弱；
- 同一主题边界需要供多个决策使用。

对于这 14 个之外的学科，编写你自己的 [`systemone`](../systemone) `choice`。

## 配置 {#configuration}

```yaml
recipes:
  - name: default
    signals:
      domains:
        - name: math
          description: Mathematics and quantitative reasoning.
        - name: stem
          description: Science and engineering.
          labels: [physics, chemistry, biology, engineering]
        - name: computing
          labels: [computer science]
          predicate: {gte: 0.6}
    decisions:
      - name: stem-route
        priority: 100
        rules:
          operator: OR
          conditions:
            - {type: domain, name: math}
            - {type: domain, name: stem}
        modelRefs:
          - model: science-model
```

| 字段 | 含义 |
| --- | --- |
| `name` | 规则名称，供 `{type: domain, name: ...}` 条件使用 |
| `labels` | 规则匹配的学科；默认为 `[name]`，所以以学科命名的规则无需填写 |
| `predicate` | 选中学科的最低概率；默认为模型的校准阈值（vela3-fast 上为 0.41），否则为 0.5 |
| `model_scores` | [`static`](../../algorithm/selection/static) 选择器对该领域请求排序时使用的各候选得分 |
| `deployment`、`timeout_ms` | 回答问题的部署，以及答案的截止时间 |

当模型选中的学科属于规则的 `labels`，且其概率满足 predicate 时，规则匹配。名称不是学科的规则必须设置 `labels`，否则配置会被拒绝，并给出最接近的学科建议。如果请求不属于你的任何规则的学科，它不会匹配任何领域规则：给一条规则设置 `labels: [other]`，即可接住模型归入 `other` 的请求。

### 回答问题的模型 {#the-model-that-answers}

除非 `domain` 绑定或规则的 `deployment` 指定了其他部署，否则由决策模型回答所有领域规则；见[选择部署](../../../system-one/deployments)。vela3-fast、Vela 3.0 的 d3 规格和 Vela 2.0 有该问题的训练形式；其他 System One 模型零样本回答。该问题读取当前用户轮次，并截断到模型的输入长度。

通过 Router 时，路由预览（`POST /api/v1/routing/preview`）会报告匹配的领域规则。完整示例见 [`config/fragments/signal/domain/mmlu.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/domain/mmlu.yaml)。

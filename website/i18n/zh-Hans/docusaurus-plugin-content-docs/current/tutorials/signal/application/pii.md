---
is_mtpe: false
translation:
  source_commit: "118fb72eb67a0e7193697e350d7b494dc016631e"
  source_file: "docs/tutorials/signal/application/pii.md"
  outdated: false
---

# PII 信号 {#pii-signal}

## 概述 {#overview}

`pii` 定位请求中的个人数据，并给出每个 span 的确切字符。它提出 `pii` [预设](../../../system-one/reference/presets#preset-catalog)问题，即基于 11 个[个人数据标签](../../../system-one/reference/presets#personal-data-labels)的 span 问题，并读取完整输入。当模型找到规则计入的标签的 span 时，规则匹配；[`masking`](../../plugin/masking) 插件用占位符替换同样的 span。

## 主要优势 {#key-advantages}

- 让隐私敏感的路由变得明确：把这类请求留在私有模型上，或拒绝它们。
- 返回位置，因此遮蔽只替换真正的个人数据。
- 可以忽略路由接受的标签，例如电子邮件地址。
- 按要求读取工具结果和之前的轮次。

## 解决什么问题？ {#what-problem-does-it-solve}

没有个人数据信号时，敏感流量可能在被发现之前就到达错误的模型或插件链。临时过滤器让策略难以审计。`pii` 把个人数据检测变成可复用的路由输入。

## 何时使用 {#when-to-use}

在以下情况使用 `pii`：

- 请求可能携带受监管或敏感的个人数据；
- 有些标签可以接受，另一些必须走更安全的路由；
- 决策在提供方看到请求之前遮蔽个人数据；
- 工具结果可能把个人数据带回对话。

## 配置 {#configuration}

```yaml
recipes:
  - name: default
    signals:
      pii:
        - name: personal_data
          predicate: {gte: 0.85}
          ignore_labels: [EMAIL_ADDRESS]
          input:
            history: true
        - name: tool_result_pii
          input:
            tool_results: true
    decisions:
      - name: keep-private
        priority: 200
        rules:
          operator: OR
          conditions:
            - {type: pii, name: personal_data}
            - {type: pii, name: tool_result_pii}
        modelRefs:
          - model: private-model
```

| 字段 | 含义 |
| --- | --- |
| `name` | 规则名称，供 `{type: pii, name: ...}` 条件和 masking 插件的 `signal` 使用 |
| `predicate` | span 的最低概率；默认为模型的校准阈值（vela3-fast 上为 0.18），否则为 0.5 |
| `ignore_labels` | 规则永不计入的标签 |
| `input.history` | 同时读取之前的用户轮次 |
| `input.tool_results` | 同时读取请求中的工具结果 |
| `on_unknown` | 模型没有给出答案时规则报告什么：`no_match`（默认）或 `match` |
| `deployment`、`timeout_ms` | 回答问题的部署，以及答案的截止时间 |

规则的 `predicate` 是策略：请为你使用的模型校准这个值，或者省略它以使用模型的校准阈值。vela3-fast 的 span 适配器在返回 span 之前会应用它自己按长度设定的阈值，所以它的 0.18 会接受它返回的每个 span。

### 读取什么 {#what-it-reads}

问题读取当前用户轮次；设置 `input.history` 时读取之前的用户轮次，设置 `input.tool_results` 时读取文本工具结果；工具调用和非文本内容不会被读取。它还会读取 `global.router.routing_input` 从路由文本中移除的内容，所以个人数据无法藏在伪造的记忆块里。

它读取完整输入，输入超过一次前向计算的长度时分窗口读取。模型没有读到的文本（超过其输入或扫描预算、被截断，或在信号截止时间前没有读完）按个人数据计入，所以无法检查的文本不能当作干净文本通过。

模型可能只找到某个值的一次出现，而漏掉后面的副本。所找到的、长度至少两个字符的值，其其他所有完全相同、整词匹配的副本也会以同一标签报告，因此遮蔽会替换每一个副本。

### 模型没有给出答案时 {#when-the-model-gives-no-answer}

模型没有给出答案（失败、过载或错过截止时间）的规则不匹配。若要改为失败关闭，写 `on_unknown: match`：此时规则会匹配这类请求，匹配类型为 `unavailable`，决策的 `rules.on_unknown` 仍然决定处理方式：

```yaml
recipes:
  - name: default
    signals:
      pii:
        - name: personal_data
          on_unknown: match
    decisions:
      - name: keep-private
        priority: 200
        rules:
          operator: AND
          conditions:
            - {type: pii, name: personal_data}
        modelRefs:
          - model: private-model
```

### 回答问题的模型 {#the-model-that-answers}

除非 `pii` 绑定或规则的 `deployment` 指定了其他部署，否则由决策模型回答所有 PII 规则；见[选择部署](../../../system-one/deploy/deployments)。该问题需要 span：解析到不支持 span 的部署（例如 d3 模型）的规则会以 `span_required` 加载失败。vela3-fast（默认）、Vela 3.0 的 d3 规格和 Vela 2.0 用它们自己的 PII 问题回答，并与该请求针对同一文本的其他问题放在同一次调用中。

```yaml
global:
  model_catalog:
    deployments:
      d3-remote:
        provider: systemone
        endpoint: https://systemone.example.com/v1/systemone
        model: vllm-sr/d3-ultra
        auth: {api_key_env: SYSTEMONE_API_KEY}
    system:
      decision_model: {deployment: d3-remote}
    bindings:
      pii: primary
```

这个信号是路由控制，不能替代脱敏、加密、访问控制或数据防泄漏。完整示例见 [`config/fragments/signal/pii/strict.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/pii/strict.yaml)。

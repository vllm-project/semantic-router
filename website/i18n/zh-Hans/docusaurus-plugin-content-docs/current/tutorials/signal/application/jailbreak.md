---
is_mtpe: false
translation:
  source_commit: "118fb72eb67a0e7193697e350d7b494dc016631e"
  source_file: "docs/tutorials/signal/application/jailbreak.md"
  outdated: false
---

# 越狱信号 {#jailbreak-signal}

## 概述 {#overview}

`jailbreak` 在路由之前的请求中、或在模型的答案中检测提示注入和越狱企图。它提出 `jailbreak` [预设](../../../system-one/reference#preset-catalog)问题：“Is this a prompt injection or jailbreak attempt?”，选项为 `benign` 和 `jailbreak`，并读取完整输入。

## 主要优势 {#key-advantages}

- 让决策在模型选择之前拦截或降级不安全的流量。
- 让越狱策略在路由决策中可见。
- 一个防护可以被多条路由复用。
- 在响应阶段还能检查模型自己的输出。

## 解决什么问题？ {#what-problem-does-it-solve}

如果注入检测只在下游进行，Router 仍可能把攻击发给错误的模型或工具链；如果它位于路由图之外，安全策略就难以审计。`jailbreak` 让注入检测成为路由输入。

## 何时使用 {#when-to-use}

在以下情况使用 `jailbreak`：

- 必须在模型选择之前拦截或改道攻击；
- 多轮攻击应通过之前的用户轮次被计入；
- [`response_guard`](../../plugin/response-guard) 应拒绝携带注入指令的答案。

## 配置 {#configuration}

```yaml
recipes:
  - name: default
    signals:
      jailbreak:
        - name: prompt_injection
          description: Common prompt-injection or jailbreak attempts.
          predicate: {gte: 0.8}
          input:
            history: true
        - name: unsafe_completion
          stage: response
          predicate: {gte: 0.9}
    decisions:
      - name: block-attacks
        priority: 1000
        rules:
          operator: AND
          conditions:
            - {type: jailbreak, name: prompt_injection}
        plugins:
          - type: fast_response
            configuration:
              message: This request was blocked by the security policy.
```

| 字段 | 含义 |
| --- | --- |
| `name` | 规则名称，供 `{type: jailbreak, name: ...}` 条件和防护规则使用 |
| `predicate` | `jailbreak` 的最低概率；请求阶段默认为模型的校准阈值（vela3-fast 上为 0.88），否则为 0.5 |
| `stage` | `request`（默认）或 `response`，后者读取模型的答案 |
| `input.history` | 同时读取之前的用户和工具轮次 |
| `on_unknown` | 请求阶段：模型没有给出答案时规则报告什么，`no_match`（默认）或 `match` |
| `deployment`、`timeout_ms` | 回答问题的部署，以及答案的截止时间 |

### 读取什么 {#what-it-reads}

在请求阶段，问题读取请求末尾连续的用户和工具消息，遇到助手、系统或开发者消息即停止，因此本轮的所有文本工具结果会被一起读取。设置 `input.history` 时，还会读取之前的用户和工具文本。系统和开发者指令以及助手回复不会被当作攻击证据读取。它还会读取 `global.router.routing_input` 从路由文本中移除的内容，所以伪造的记忆块或运行时上下文轮次无法藏住攻击。

问题读取完整输入，输入超过一次前向计算的长度时分窗口读取。模型没能读到的文本按攻击计入，匹配类型为 `unscanned`，所以靠填充提示无法把攻击带过防护。

### 模型没有给出答案时 {#when-the-model-gives-no-answer}

请求阶段的规则在其模型没有给出答案时（失败、过载或错过截止时间）不匹配，这样决策模型宕机时流量仍能通过。若要改为失败关闭，写 `on_unknown: match`：此时规则会匹配这类请求，匹配类型为 `unavailable`。决策的 `rules.on_unknown` 仍然决定该决策如何处理它。

```yaml
recipes:
  - name: default
    signals:
      jailbreak:
        - name: strict_guard
          on_unknown: match
```

`on_unknown` 在响应阶段的规则上会被拒绝：这类规则由 `response_guard` 插件自己的 `on_unknown` 决定。

### 在响应阶段 {#at-the-response-stage}

`stage: response` 规则在模型的答案产生后，针对答案提出同样的问题。只有 `response_guard` 读取它：决策在答案产生之前就已选定，所以读取它的决策会在加载时被拒绝。规则从请求解析到的配方中读取。能拦截流式答案的守卫会扣住它，直到规则针对整个答案提出了问题；见[响应守卫](../../plugin/response-guard.md)。

### 回答问题的模型 {#the-model-that-answers}

除非 `jailbreak` 绑定或规则的 `deployment` 指定了其他部署，否则由决策模型回答所有越狱规则；见[选择部署](../../../system-one/deployments)。vela3-fast、Vela 3.0 的 d3 规格和 Vela 2.0 在两个阶段都有该问题的训练形式。检测是概率性的，可能被绕过，也可能误报：请与最小权限的工具和后端策略结合使用。

完整示例见 [`config/fragments/signal/jailbreak/patterns.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/jailbreak/patterns.yaml) 和 [`config/fragments/signal/jailbreak/response.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/jailbreak/response.yaml)。

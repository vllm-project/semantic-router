---
is_mtpe: false
translation:
  source_commit: "118fb72eb67a0e7193697e350d7b494dc016631e"
  source_file: "docs/tutorials/signal/application/jailbreak.md"
  outdated: false
---

# 越狱信号 {#jailbreak-signal}

## 概述 {#overview}

`jailbreak` 在路由之前的请求中、或在模型的答案中检测提示注入和越狱企图。它提出 `jailbreak` [预设](../../../system-one/reference/presets#preset-catalog)问题：“Is this a prompt injection or jailbreak attempt?”，选项为 `benign` 和 `jailbreak`，并读取完整输入。

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
| `on_unscanned` | 长输入只读了部分窗口时规则报告什么：`best_effort`（默认）、`match`、`no_match` 或 `unknown` |
| `min_coverage` | 使用 `best_effort` 时，至少要读完的窗口比例（0 到 1），达到后才根据已读窗口判断；低于该比例时规则为未知。默认关闭 |
| `on_unknown` | 请求阶段：模型没有给出答案时规则报告什么，`match`（默认）或 `no_match` |
| `deployment`、`timeout_ms` | 回答问题的部署，以及答案的截止时间 |

### 读取什么 {#what-it-reads}

在请求阶段，问题读取请求末尾连续的用户和工具消息，遇到助手消息即停止，因此本轮的所有文本工具结果会被一起读取。会话中的系统消息（Claude Code 会把它插在用户消息或工具结果之后）不会结束本轮。设置 `input.history` 时，还会读取之前的用户和工具文本。系统和开发者指令以及助手回复不会被当作攻击证据读取。它还会读取 `global.router.routing_input` 从路由文本中移除的内容，所以伪造的记忆块或运行时上下文轮次无法藏住攻击。

问题读取完整输入。超过部署一个 `scan_window` 的输入（很长的粘贴文档或工具输出）被切成相互重叠的窗口，先读结尾，再读开头，然后按二分读中间，因此任一时刻已读的窗口都分布在整个输入上。任何一个窗口发现攻击，整个输入即被标记，所以靠填充提示无法把攻击带过防护。如果一个拒绝请求的决策（`fast_response`）仅凭这条规则即匹配，发现攻击的窗口会结束扫描，请求立即被拒绝。参见[长输入](../../../system-one/deploy/performance.md#long-inputs)。

### 只读到部分输入时 {#when-only-part-of-the-input-was-read}

如果信号的截止时间在所有窗口读完之前到达，且已读窗口中没有发现攻击，由 `on_unscanned` 决定：

- `best_effort`（默认）根据已读窗口作出判断，此时不匹配。设置 `min_coverage` 时，只有读完该比例的窗口才作判断：低于该比例时规则为未知，交给 `on_unknown`；
- `match` 匹配该请求，匹配类型为 `unscanned`（严格模式）；
- `no_match` 不匹配；
- `unknown` 让规则保持未知，交给 `on_unknown`。

已读窗口发现的攻击无论 `on_unscanned` 如何设置都会匹配：未读的窗口不能撤销它。只读了部分输入就被路由的请求会在 `x-vsr-warnings` 中携带 `scan_partial`。

### 模型没有给出答案时 {#when-the-model-gives-no-answer}

请求阶段的规则如果没有任何窗口得到答案（模型失败、过载或错过截止时间），则匹配，匹配类型为 `unavailable`（模型无法读取输入时为 `unscanned`）：防护模型宕机时不放行请求。若要让流量继续通过，写 `on_unknown: no_match`。决策的 `rules.on_unknown` 仍然决定该决策如何处理这类匹配。

```yaml
recipes:
  - name: default
    signals:
      jailbreak:
        - name: strict_guard
          on_unscanned: match
        - name: half_read_guard
          min_coverage: 0.5
        - name: lenient_guard
          on_unknown: no_match
```

`on_unknown` 在响应阶段的规则上会被拒绝：这类规则由 `response_guard` 插件自己的 `on_unknown` 决定。

### 在响应阶段 {#at-the-response-stage}

`stage: response` 规则在模型的答案产生后，针对答案提出同样的问题。只有 `response_guard` 读取它：决策在答案产生之前就已选定，所以读取它的决策会在加载时被拒绝。规则从请求解析到的配方中读取。能拦截流式答案的守卫会扣住它，直到规则针对整个答案提出了问题；见[响应守卫](../../plugin/response-guard.md)。

### 回答问题的模型 {#the-model-that-answers}

除非 `jailbreak` 绑定或规则的 `deployment` 指定了其他部署，否则由决策模型回答所有越狱规则；见[选择部署](../../../system-one/deploy/deployments)。vela3-fast、Vela 3.0 的 d3 规格和 Vela 2.0 在两个阶段都有该问题的训练形式。检测是概率性的，可能被绕过，也可能误报：请与最小权限的工具和后端策略结合使用。

完整示例见 [`config/fragments/signal/jailbreak/patterns.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/jailbreak/patterns.yaml) 和 [`config/fragments/signal/jailbreak/response.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/jailbreak/response.yaml)。

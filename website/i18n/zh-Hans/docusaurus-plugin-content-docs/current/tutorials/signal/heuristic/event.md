---
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/tutorials/signal/heuristic/event.md"
  outdated: false
---

# 事件信号 {#event-signal}

## 概览 {#overview}

`event` 按事件类型、严重级别、紧急程度或领域特定动作码，为结构化、类事件请求路由。

在 `routing.signals.events` 下定义事件规则。

## 主要优势 {#key-advantages}

- 使用基于正则的匹配，无需模型推理。
- 将企业事件驱动载荷（错误告警、审计日志、事故报告）路由到专用模型池，无需领域分类器。
- 置信度与匹配条件数量成正比，为决策引擎提供分级信号。
- 时间紧急检测（`urgent`、`immediate`、`asap`、`deadline`、`time-sensitive`、`now`、`critical.window`）可独立于事件类型路由时间敏感事件。

## 解决什么问题？ {#what-problem-does-it-solve}

keyword 与 embedding 信号面向自然语言查询。结构化事件载荷——JSON 片段、告警消息、交易错误码——包含定义明确的字段，关键词匹配无法干净建模。`event` 为每类事件提供命名、可组合的信号，而不强迫运维编写脆弱的正则关键词规则。

## 何时使用 {#when-to-use}

在以下情况使用 `event`：

- 请求包含机器生成的事件载荷（错误告警、审计日志、交易失败）
- 希望按严重级别档独立于事件类型路由
- 领域特定动作码（例如 `TXN_DECLINE`、`AUTH_FAIL`）应确定性选择模型池
- 时间敏感事件需要绕过标准的延迟容忍队列

## 配置 {#configuration}

```yaml
routing:
  signals:
    events:
      - name: critical_payment_event
        description: Critical payment or transaction events that need incident-grade routing.
        event_types:
          - payment_failed
          - transaction_declined
        severities:
          - critical
          - high
        action_codes:
          - TXN_DECLINE
        temporal: true
```

### 字段 {#fields}

| 字段 | 类型 | 说明 |
|-------|------|-------------|
| `name` | string | 在 `routing.decisions[].rules` 中引用的规则名 |
| `description` | string | 可选的人类可读说明，解释规则何时应匹配 |
| `event_types` | 字符串列表 | 要匹配的事件类型模式（不区分大小写的词边界） |
| `severities` | 字符串列表 | 严重级别关键词：`critical`、`high`、`medium`、`low` |
| `action_codes` | 字符串列表 | 领域特定动作码（不区分大小写的词边界） |
| `temporal` | bool | 为 `true` 时匹配紧急标记：`urgent`、`immediate`、`asap`、`deadline`、`time-sensitive`、`now`、`critical.window` |

至少满足一个已配置条件时规则匹配。**置信度**为 `0.5 + 0.5 × (matched_criteria / total_criteria)`。最多四个条件（`event_types`、`severities`、`action_codes`、`temporal`）时，四条件规则上的单条件匹配得到 `0.625`；完全匹配始终得到 `1.0`。只配置一个条件且匹配的规则也得到 `1.0`。

## 决策示例 {#example-decision}

```yaml
routing:
  decisions:
    - name: route_critical_event
      description: Route critical payment events to the fast response model.
      priority: 200
      rules:
        type: event
        name: critical_payment_event
      modelRefs:
        - model: fast-response-model
```

## 依赖与限制 {#dependencies-and-limitations}

事件匹配读取请求文本；它不解析或校验权威事件 schema。正则匹配可被调用方伪造，因此不要单独用该信号做授权或事故严重级别判定。完整示例见：
[`config/fragments/signal/event/payment-critical.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/event/payment-critical.yaml)。

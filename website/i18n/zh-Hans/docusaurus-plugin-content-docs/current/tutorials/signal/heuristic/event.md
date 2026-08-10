---
translation:
  source_commit: "a0694610"
  source_file: "docs/tutorials/signal/heuristic/event.md"
  outdated: false
---

# Event 信号

## 概览

`event` 是启发式路由信号族，用于处理从请求文本中提取的**结构化事件元数据**：事件类型、严重级别、时间紧迫性和领域特定操作代码。

它映射到 `config/fragments/signal/event/`，并在 `routing.signals.events` 下声明。

## 主要优势

- 零 ML 推理：所有匹配均基于正则表达式，运行时间不到一毫秒。
- 无需领域分类器，即可将企业事件驱动型载荷（错误警报、审计日志、事件报告）路由到专用模型池。
- 置信度与匹配条件的数量成正比，为决策引擎提供有梯度的信号。
- 时间紧迫性检测（`urgent`、`immediate`、`asap`、`deadline`、`time-sensitive`、`now`、`critical.window`）可独立于事件类型路由时效敏感事件。

## 解决什么问题？

关键词和嵌入信号针对自然语言查询进行了调优。结构化事件载荷（JSON 片段、警报消息、事务错误代码）包含定义明确的字段，关键词匹配无法清晰地对此建模。`event` 为每类事件提供命名且可组合的信号，而不强迫运维人员编写脆弱的正则表达式关键词规则。

## 何时使用

在以下情况使用 `event`：

- 请求包含机器生成的事件载荷（错误警报、审计日志、事务失败）
- 希望独立于事件类型按严重性等级进行路由
- 领域特定操作代码（例如 `TXN_DECLINE`、`AUTH_FAIL`）应以确定性方式选择模型池
- 时效敏感事件需要绕过标准的延迟容忍队列

## 配置

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

### 字段

| 字段 | 类型 | 描述 |
| --- | --- | --- |
| `name` | 字符串 | `routing.decisions[].rules` 中引用的规则名称 |
| `description` | 字符串 | 规则应在何时匹配的可选易读说明 |
| `event_types` | 字符串列表 | 要匹配的事件类型模式（不区分大小写的单词边界） |
| `severities` | 字符串列表 | 严重性关键词：`critical`、`high`、`medium`、`low` |
| `action_codes` | 字符串列表 | 领域特定操作代码（不区分大小写的单词边界） |
| `temporal` | 布尔值 | 为 `true` 时，匹配紧迫性标记：`urgent`、`immediate`、`asap`、`deadline`、`time-sensitive`、`now`、`critical.window` |

至少满足一个已配置条件时，规则即匹配。**置信度**为 `0.5 + 0.5 × (matched_criteria / total_criteria)`。最多可配置四个条件（`event_types`、`severities`、`action_codes`、`temporal`）；对于配置了四个条件的规则，仅匹配一个条件时置信度为 `0.625`，完全匹配时始终为 `1.0`。仅配置一个条件的规则在该条件匹配时，置信度也为 `1.0`。

## 决策示例

```yaml
routing:
  decisions:
    - name: route_critical_event
      rules:
        type: event
        name: critical_payment_event
      modelRefs:
        - model: fast-response-model
```

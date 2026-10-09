---
translation:
  source_commit: "9156d5bc1ed9edff626b95a2b8260a77cb1712c5"
  source_file: "docs/tutorials/signal/learned/decision.md"
  outdated: false
---

# Decision 信号

`routing.signals.decision` 向判断模型提出具名问题，答案可供决策和投影使用。任务支持来自运行时报告的原生题型，不由模型家族名或是否针对路由训练决定。

## 配置

```yaml
routing:
  signals:
    decision:
      - name: task
        question:
          type: choice
          instructions: What kind of task does the request ask for?
          choices:
            - key: code
              description: Writing or reviewing code
            - key: other
              description: Other requests
  decisions:
    - name: coding
      rules:
        operator: AND
        conditions:
          - type: decision
            name: task
            label: code
      modelRefs:
        - model: code-model
```

先在 `providers.models` 声明后端。没有 `deployment` 的问题使用 `global.model_catalog.system.decision_model`，默认 Vela 2.0 0.3B；显式 `deployment` 覆盖该选择。

## 题型与任务 {#set-and-span-questions}

| 题型 | 结果 | 用途 |
| --- | --- | --- |
| `choice` | 单个选项与概率 | 类别、偏好、选模 |
| `score` | 等级之间的分数 | 复杂度 |
| `noul` | 判断概率 | 存在性、安全性 |
| `set` | 多个独立标签 | 需求、风险类别 |
| `span` | 原文位置与标签 | 实体定位 |

Router 的任务编译器在模型没有原生 set 但支持 noul 时，可以用每标签一个 noul 问题实现 set，标为 `composed_noul`。span 需要原生支持，不能从存在性答案虚构位置。System One 原生 API 仍按原生题型校验；Router 组合任务不扩展它的原生模型能力。任务可用与质量评测是两件事。

## 批处理与输入预算

相同部署、相同阶段的兼容问题可以合并；不同输入保留独立状态。选模与响应检查发生在后续阶段，可能再次调用同一部署。一份模型资源、一次 API 调用和一次 forward 是不同概念。

一般路由问题可以按配置截取长输入供判断，发往 Chat 后端的请求不会因此被截断。需要完整扫描时使用相应的 PII、Guard 或 Reask 任务与覆盖策略。输入超过预算、失败或超时会产生未知结果，按 `on_error` 与决策 `rules.on_unknown` 处理；后者可拒绝请求。调用截止时间不保证底层正在运行的推理立即停止。

完整字段与示例见 [Decision 模型指南](../../../model-runtime/guides/decisions)和 [System One API](../../../api/router.md)。

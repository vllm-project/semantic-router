---
translation:
  source_commit: "9716b569"
  source_file: "docs/tutorials/algorithm/selection/prompt.md"
  outdated: false
---

# Prompt 选择

## 概览

`prompt` 使用一个具体的辅助模型，从已匹配决策的 `modelRefs` 中恰好选择一个模型。候选列表、结构化响应 schema、确定性生成设置和回退行为均由运行时负责。

## 主要优势

- 使用直白的指令表达模型选择策略
- 将辅助模型限制在已声明的决策候选模型范围内
- 复用模型卡描述和现有的选择注册表
- 辅助调用失败时回退到第一个有效候选模型

## 解决什么问题？

对于定性的路由策略，语义选择器和基于指标的选择器并不总是最简单的编码方式。Prompt 选择允许小型模型在有明确边界的候选集合中进行选择，而无需将请求资格或安全逻辑移出信号和决策层。

## 何时使用

当一个决策有多个有效候选模型，且选择取决于定性的任务要求时，请使用 Prompt 选择。元数据、授权、隐私及其他确定性门控仍应保留在信号和决策层中。

## 配置

```yaml
routing:
  decisions:
    - name: adaptive-model-choice
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: general-small
          use_reasoning: false
        - model: reasoning-large
          use_reasoning: true
      algorithm:
        type: prompt
        on_error: fallback
        prompt:
          model: router-small
          instructions: >-
            Use general-small for ordinary requests. Use reasoning-large for
            hard reasoning, coding, debugging, or multi-step analysis.
          timeout_seconds: 5
```

`model` 必须是 `routing.modelCards` 中声明并由 `providers.models` 支持的具体模型，而且必须使用 OpenAI 兼容的 API 格式。候选基础模型的名称必须唯一；当 LoRA 变体或推理变体共享同一个基础模型时，应使用不同的决策。候选模型名称和可用的模型卡描述由运行时添加。选择器接收当前用户轮次，并返回一个固定的 JSON 对象，其中包含精确的候选模型名称和简短的选择理由。
内部辅助调用使用 `global.integrations.looper.endpoint`，该端点必须指向路由器的 OpenAI 兼容聊天端点。

模型生成的选择理由文本不会按原文记录到日志或持久化。Replay 存储有明确上限的结果/回退原因代码，指标则公开选择器耗时和回退次数，但不包含请求内容。

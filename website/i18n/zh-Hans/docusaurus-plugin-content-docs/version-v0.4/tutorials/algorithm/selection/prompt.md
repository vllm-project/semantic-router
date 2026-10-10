---
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/tutorials/algorithm/selection/prompt.md"
  outdated: false
---

# 提示词选择

## 概览

`prompt` 使用具体的辅助模型，从匹配决策的 `modelRefs` 中恰好选择一个模型。运行时拥有候选列表、结构化响应 schema、确定性生成设置和回退行为。

## 主要优势

- 用普通指令表达模型选择策略
- 将辅助模型约束在已声明的决策候选内
- 复用模型卡片描述和现有选择注册表
- 辅助调用失败时回退到第一个有效候选

## 解决什么问题？

语义和基于指标的选择器并不总是编码定性路由策略的最简单方式。提示词选择让小模型在有界候选集中选择，同时不把请求资格或安全逻辑移出信号和决策。

## 何时使用

当决策有多个有效候选，且选择取决于定性任务要求时，使用提示词选择。把元数据、授权、隐私和其他确定性门控留在信号和决策中。

## 配置

```yaml
routing:
  decisions:
    - name: adaptive-model-choice
      description: Let a helper model choose the best eligible candidate.
      priority: 100
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

`model` 必须是在 `routing.modelCards` 中声明、并由 `providers.models` 支持的具体模型，且必须使用 OpenAI 兼容 API 格式。候选基础模型名必须唯一；当 LoRA 或推理变体共享同一基础模型时，使用单独的决策。候选名称和可用的模型卡片描述由运行时添加。选择器接收当前用户回合，并返回包含精确候选名称和简短理由的固定 JSON 对象。
内部辅助调用使用 `global.integrations.looper.endpoint`，它必须指向路由器的 OpenAI 兼容 chat 端点。

模型生成的理由文本不会按原文记录或持久化。回放存储有界的结果/回退原因码，指标暴露选择器耗时和回退次数，不含请求内容。

辅助模型会收到当前用户回合和候选描述，因此其提供商必须被路由的数据策略允许。它的选择被限制在已声明候选内，但仍是模型生成的；确定性的隐私、授权和安全门控属于信号和决策。完整示例见：
[`config/fragments/algorithm/selection/prompt.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/selection/prompt.yaml)。

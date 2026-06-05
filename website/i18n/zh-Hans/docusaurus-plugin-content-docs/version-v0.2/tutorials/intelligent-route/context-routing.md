---
sidebar_position: 8
sidebar_label: 上下文路由教程
translation:
  source_file: "docs/tutorials/intelligent-route/context-routing.md"
  outdated: false
---

# 上下文路由教程

本教程将向你展示如何使用 **上下文信号**（Token 数量）根据请求长度进行路由。

这在以下场景中非常有用：

- 将短查询路由到更快、更小的模型
- 将长文档/长提示词路由到支持长上下文窗口的模型
- 对短任务使用更便宜的模型，从而优化成本

## 场景说明

我们的目标是：

1. 将短请求（< 4K tokens）路由到快速模型（`llama-3-8b`）
2. 将中等长度请求（4K - 32K tokens）路由到标准模型（`llama-3-70b`）
3. 将长请求（32K - 128K tokens）路由到长上下文模型（`claude-3-opus`）

## 第一步：定义上下文信号

在 `signals` 配置中添加 `context_rules`：

```yaml
signals:
  context:
    - name: "short_context"
      min_tokens: "0"
      max_tokens: "4K"
      description: "Short queries suitable for fast models"

    - name: "medium_context"
      min_tokens: "4K"
      max_tokens: "32K"
      description: "Medium length context"

    - name: "long_context"
      min_tokens: "32K"
      max_tokens: "128K"
      description: "Long context requiring specialized handling"
```

## 第二步：定义决策规则

创建基于这些上下文信号触发的路由决策：

```yaml
decisions:
  - name: "fast_route"
    priority: 10
    rules:
      operator: "AND"
      conditions:
        - type: "context"
          name: "short_context"
    modelRefs:
      - model: "llama-3-8b"

  - name: "standard_route"
    priority: 10
    rules:
      operator: "AND"
      conditions:
        - type: "context"
          name: "medium_context"
    modelRefs:
      - model: "llama-3-70b"

  - name: "long_context_route"
    priority: 10
    rules:
      operator: "AND"
      conditions:
        - type: "context"
          name: "long_context"
    modelRefs:
      - model: "claude-3-opus"
```

## 第三步：组合逻辑（高级）

你可以将上下文信号与其他信号（例如领域或关键词）组合使用。

**示例**：将大型 **编程** 任务路由到专门支持长上下文的代码模型：

```yaml
decisions:
  - name: "long_code_analysis"
    priority: 20  # 更高优先级
    rules:
      operator: "AND"
      conditions:
        - type: "context"
          name: "long_context"
        - type: "domain"
          name: "computer_science"
    modelRefs:
      - model: "deepseek-coder-v2"
```

## Token 计数机制说明

- 路由器会在 **做出路由决策之前** 统计 token 数量。
- 使用与大多数 LLM 兼容的快速分词器。
- 为了提升可读性，支持使用 `"K"`（1000）和 `"M"`（1,000,000）这样的后缀。
- 如果一个请求同时匹配多个范围（例如规则存在重叠），则所有匹配的上下文信号都会被激活。

## 监控

你可以通过 Prometheus 指标监控 token 的分布情况：
`llm_context_token_count`

这有助于你根据实际的流量模式，微调上下文区间的划分。

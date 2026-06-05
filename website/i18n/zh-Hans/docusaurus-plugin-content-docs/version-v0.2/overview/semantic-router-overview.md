---
translation:
  source_commit: "bac2743"
  source_file: "docs/overview/semantic-router-overview.md"
  outdated: false
sidebar_position: 2
---

# 什么是 Semantic Router？

**Semantic Router** 是一个智能路由层，它根据从请求中提取的多种 signal，动态地为每个查询选择最合适的语言模型。

## 问题

传统的 LLM 部署对所有任务使用单一模型：

```text
用户查询 → 单一 LLM → 响应
```

**问题**：

- 简单查询的成本高
- 专业任务的性能不佳
- 没有安全或合规控制
- 资源利用率低

## 解决方案

Semantic Router 使用**Signal-Driven Decision**来智能地路由查询：

```text
用户查询 → Signal Extraction → Decision Engine → 最佳模型 → 响应
```

**优势**：

- 具有成本效益的路由（简单任务使用较小的模型）
- 更好的质量（利用专业模型的优势）
- 内置安全性（jailbreak 检测、PII 过滤）
- 灵活且可扩展（Plugin 架构）

## 工作原理

### 1. 信号提取

Router 从每个请求中提取多种类型的 signal：

| 信号类型 | 检测内容 | 示例 |
|------------|----------------|---------|
| **keyword** | 特定术语和模式 | "calculate", "prove", "debug" |
| **embedding** | 语义含义 | 数学意图、代码意图、创意意图 |
| **domain** | 知识领域 | 数学、计算机科学、历史 |
| **fact_check** | 验证需求 | 事实主张、医疗建议 |
| **user_feedback** | 用户满意度 | "That's wrong", "try again" |
| **preference** | 路由偏好 | 复杂意图匹配 |

### 2. 决策制定

使用逻辑规则组合 signal 以做出路由决策：

```yaml
decisions:
  - name: math_routing
    rules:
      operator: "AND"
      conditions:
        - type: "keyword"
          name: "math_keywords"
        - type: "domain"
          name: "mathematics"
    modelRefs:
      - model: qwen-math
        weight: 1.0
```

**工作原理**：如果查询包含数学关键词 **AND (并且)** 被归类为数学领域，则路由到数学模型。

### 3. 模型选择

根据 decision，Router 选择最佳模型：

- **数学查询** → 数学专用模型 (例如 Qwen-Math)
- **代码查询** → 代码专用模型 (例如 DeepSeek-Coder)
- **创意查询** → 创意模型 (例如 Claude)
- **简单查询** → 轻量级模型 (例如 Llama-3-8B)

### 4. Plugin Chain

在模型执行之前和之后，plugin 处理请求/响应：

```yaml
plugins:
  - type: "semantic-cache"    # 首先检查缓存
  - type: "jailbreak"         # 检测对抗性 prompt
  - type: "pii"               # 过滤敏感数据
  - type: "system_prompt"     # 添加上下文
  - type: "hallucination"     # 验证事实
```

## 关键概念

### Mixture of Models (MoM)

与在单一模型内运行的 Mixture of Experts (MoE) 不同，Mixture of Models 在**系统级别**运行：

| 方面 | Mixture of Experts (MoE) | Mixture of Models (MoM) |
|--------|-------------------------|------------------------|
| **范围** | 在单一模型内 | 跨多个模型 |
| **路由** | 内部门控网络 | 外部Semantic Router  |
| **模型** | 共享架构 | 独立模型 |
| **灵活性** | 训练时固定 | 运行时动态 |
| **用例** | 模型效率 | 系统智能 |

### Signal-Driven Decision

传统路由使用简单的规则：

```yaml
# 传统：简单的关键词匹配
if "math" in query:
    route_to_math_model()
```

Signal-Driven routing 使用多种 signal：

```yaml
# 信号驱动：多种信号组合
if (has_math_keywords AND is_math_domain) OR has_high_math_embedding:
    route_to_math_model()
```

**优势**：

- 更准确的路由
- 更好地处理边缘情况
- 适应上下文
- 减少误报

## 真实世界示例

**用户查询**："证明 2 的平方根是无理数"

**信号提取**：

- keyword: ["prove", "square root", "irrational"] ✓
- embedding: 与数学查询的相似度 0.89 ✓
- domain: "mathematics" (数学) ✓

**决策**：路由到 `qwen-math` (所有数学信号一致)

**应用插件**：

- semantic-cache: 缓存未命中，继续
- jailbreak: 无对抗性模式
- system_prompt: 添加了 "Provide rigorous mathematical proof" (提供严格的数学证明)
- hallucination: 启用以进行验证

**结果**：来自专业模型的高质量数学证明

## 下一步

- [什么是 Collective Intelligence？](collective-intelligence.md) - signal 如何创造系统智能
- [什么是 Signal-Driven Decision？](signal-driven-decisions.md) - 深入解 Decision Engine
- [配置指南](../installation/configuration.md) - 设置您的 Semantic Router

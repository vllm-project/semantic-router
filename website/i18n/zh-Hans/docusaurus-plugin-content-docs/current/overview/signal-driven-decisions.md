---
translation:
  source_commit: "bac2743"
  source_file: "docs/overview/signal-driven-decisions.md"
  outdated: true
sidebar_position: 4
---

# 什么是 Signal-Driven Decision？

**Signal-Driven Decision** 是核心架构，它通过从请求中提取多种 signal 并将它们结合起来做出更好的路由决策，从而实现智能路由。

## 核心理念

传统路由使用单一信号：

```yaml
# 传统：单一分类模型
if classifier(query) == "math":
    route_to_math_model()
```

Signal-Driven routing 使用多种 signal：

```yaml
# 信号驱动：多种信号组合
if (keyword_match AND domain_match) OR high_embedding_similarity:
    route_to_math_model()
```

**为什么这很重要**：多个 signal 共同投票比任何单一 signal 做出更准确的决策。

## 7 种 Signal 类型

### 1. Keyword Signal

- **内容**：使用 AND/OR 运算符的快速模式匹配
- **延迟**：小于 1ms
- **用例**：确定性路由、合规性、安全性

```yaml
signals:
  keywords:
    - name: "math_keywords"
      operator: "OR"
      keywords: ["calculate", "equation", "solve", "derivative"]
```

**示例**："Calculate the derivative of x^2" → 匹配 "calculate" 和 "derivative"

### 2. Embedding Signal

- **内容**：使用 embedding 的语义相似度
- **延迟**：10-50ms
- **用例**：意图检测、释义处理

```yaml
signals:
  embeddings:
    - name: "code_debug"
      threshold: 0.70
      candidates:
        - "My code isn't working, how do I fix it?"
        - "Help me debug this function"
```

**示例**："Need help debugging this function" → 0.78 相似度 → 匹配！

### 3. Domain Signal

- **内容**：MMLU 领域分类（14 个类别）
- **延迟**：50-100ms
- **用例**：学术和专业领域路由

```yaml
signals:
  domains:
    - name: "mathematics"
      mmlu_categories: ["abstract_algebra", "college_mathematics"]
```

**示例**："Prove that the square root of 2 is irrational" → Mathematics (数学) 领域

### 4. Fact Check Signal

- **内容**：基于机器学习的需要事实验证的查询检测
- **延迟**：50-100ms
- **用例**：医疗保健、金融服务、教育

```yaml
signals:
  fact_checks:
    - name: "factual_queries"
      threshold: 0.75
```

**示例**："What is the capital of France?" → 需要事实核查

### 5. User Feedback Signal

- **内容**：用户反馈和更正的分类
- **延迟**：50-100ms
- **用例**：客户支持、自适应学习

```yaml
signals:
  user_feedbacks:
    - name: "negative_feedback"
      feedback_types: ["correction", "dissatisfaction"]
```

**示例**："That's wrong, try again" → 检测到负面反馈

### 6. Preference Signal

- **内容**：基于 LLM 的路由偏好匹配
- **延迟**：200-500ms
- **用例**：复杂意图分析

```yaml
signals:
  preferences:
    - name: "creative_writing"
      llm_endpoint: "http://localhost:8000/v1"
      model: "gpt-4"
      routes:
        - name: "creative"
          description: "Creative writing, storytelling, poetry"
```

**示例**："Write a story about dragons" → 偏好创意路由

### 7. Language Signal

- **内容**：多语言检测（100 多种本地化语言）
- **延迟**：小于 1ms
- **用例**：路由查询特定语言的模型或采用特定语言的策略

```yaml
signals:
  language:
    - name: "en"
      description: "English language queries"
    - name: "es"
      description: "Spanish language queries"
    - name: "zh"
      description: "Chinese language queries"
    - name: "ru"
      description: "Russian language queries"
```

- **示例 1**："Hola, ¿cómo estás?" → Spanish (es) → Spanish model
- **示例 2**："你好，世界" → Chinese (zh) → Chinese model

## Signal 如何组合

### AND 运算符 - 必须全部匹配

```yaml
decisions:
  - name: "advanced_math"
    rules:
      operator: "AND"
      conditions:
        - type: "keyword"
          name: "math_keywords"
        - type: "domain"
          name: "mathematics"
```

- **逻辑**：**仅当**关键词 AND (并且) 领域都匹配时，路由到 advanced_math
- **用例**：高置信度路由（减少误报）

### OR 运算符 - 任意匹配

```yaml
decisions:
  - name: "code_help"
    rules:
      operator: "OR"
      conditions:
        - type: "keyword"
          name: "code_keywords"
        - type: "embedding"
          name: "code_debug"
```

- **逻辑**：**如果**关键词 OR (或者) 嵌入匹配，路由到 code_help
- **用例**：广泛覆盖（减少漏报）

### 嵌套逻辑 - 复杂规则

```yaml
decisions:
  - name: "verified_math"
    rules:
      operator: "AND"
      conditions:
        - type: "domain"
          name: "mathematics"
        - operator: "OR"
          conditions:
            - type: "keyword"
              name: "proof_keywords"
            - type: "fact_check"
              name: "factual_queries"
```

- **逻辑**：如果 (数学领域) AND (证明关键词 OR 需要事实核查) 则路由
- **用例**：复杂路由场景

## 真实世界示例

### 用户查询

```text
"Prove that the square root of 2 is irrational"
```

### 信号提取

```yaml
signals_detected:
  keyword: true          # "prove", "square root", "irrational"
  embedding: 0.89        # 与数学查询的高度相似性
  domain: "mathematics"  # MMLU 分类
  fact_check: true       # 证明需要验证
```

### 决策过程

```yaml
decision: "advanced_math"
reason: "All math signals agree (keyword + embedding + domain + fact_check)" # 所有数学信号一致
confidence: 0.95
selected_model: "qwen-math"
```

### 为什么这有效

- **多个信号一致**：高置信度
- **启用了事实核查**：质量保证
- **专业模型**：最适合数学证明

## 下一步

- [配置指南](../installation/configuration.md) - 配置 signal 和 decision
- [Keyword Routing 教程](../tutorials/intelligent-route/keyword-routing.md) - 学习 keyword signal
- [Embedding Routing 教程](../tutorials/intelligent-route/embedding-routing.md) - 学习 embedding signal
- [Domain Routing 教程](../tutorials/intelligent-route/domain-routing.md) - 学习 domain signal

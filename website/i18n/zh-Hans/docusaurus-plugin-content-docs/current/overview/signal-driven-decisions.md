---
translation:
  source_commit: "dd5c06f"
  source_file: "docs/overview/signal-driven-decisions.md"
  outdated: true
is_mtpe: true
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

## 10 种 Signal 类型

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

### 8. 延迟信号 — 基于百分位的路由

- **内容**：使用 TPOT（Time Per Output Token，每个输出 Token 的耗时）和
  TTFT（Time To First Token，首 Token 延迟）的百分位数对模型延迟进行评估。
- **延迟**：通常为 2–5ms（针对 10 个模型，异步运行）。百分位计算的时间复杂度为
  O(n log n)，其中 n 为每个模型的观测样本数（通常 10–100，最大 1000）。
- **用例**：基于自适应的百分位阈值，将对延迟敏感的请求路由到更快的模型。

```yaml
signals:
  latency:
    - name: "low_latency_comprehensive"
      tpot_percentile: 10  # TPOT 的第 10 百分位（最快的前 10% Token 生成速度）
      ttft_percentile: 10  # TTFT 的第 10 百分位（最快的前 10% 首 Token）
      description: "适用于实时应用——启动快、生成快"
    - name: "balanced_latency"
      tpot_percentile: 50  # TPOT 的中位数
      ttft_percentile: 10  # TTFT 的前 10%（优先保证快速启动）
      description: "优先快速启动，接受中等的生成速度"
```

**示例**：
实时聊天请求 → `low_latency_comprehensive` 信号 → 路由到同时满足 TPOT 和 TTFT 百分位阈值的模型。

**工作原理**：

- TPOT 和 TTFT 会从每一次模型响应中自动采集和统计
- 基于百分位的阈值会自适应每个模型的实际性能分布
- 支持任意数量的观测样本：
  - 1–2 个样本时使用平均值
  - 3 个及以上样本时使用百分位计算
- 当同时设置 TPOT 和 TTFT 百分位时，模型必须**同时满足两个条件**（AND 逻辑）
- **推荐做法**：同时使用 TPOT 和 TTFT 百分位，以获得更全面的延迟评估

### 9. Context Signal

- **内容**：基于 token 计数的短/长请求处理路由
- **延迟**：1ms（处理过程中计算）
- **用例**：将长上下文请求路由到具有更大上下文窗口的模型
- **指标**：使用 `llm_context_token_count` 直方图跟踪输入 token 计数

```yaml
signals:
  context_rules:
    - name: "low_token_count"
      min_tokens: "0"
      max_tokens: "1K"
      description: "短请求"
    - name: "high_token_count"
      min_tokens: "1K"
      max_tokens: "128K"
      description: "需要大上下文窗口的长请求"
```

**示例**：一个包含 5,000 个 token 的请求 → 匹配 "high_token_count" → 路由到 `claude-3-opus`

### 10. Complexity Signal

- **内容**：基于 embedding 的查询复杂度分类（困难/简单/中等）
- **延迟**：50-100ms（embedding 计算）
- **用例**：将复杂查询路由到强大模型，简单查询路由到高效模型
- **逻辑**：两步分类：
  1. 通过将查询与规则描述进行比较，找到最匹配的规则
  2. 使用困难/简单候选 embedding 在该规则内分类难度

```yaml
signals:
  complexity:
    - name: "code_complexity"
      threshold: 0.1
      description: "检测代码复杂度级别"
      hard:
        candidates:
          - "design distributed system"
          - "implement consensus algorithm"
          - "optimize for scale"
      easy:
        candidates:
          - "print hello world"
          - "loop through array"
          - "read file"
```

**示例**："How do I implement a distributed consensus algorithm?" → 匹配 "code_complexity" 规则 → 与困难候选高度相似 → 返回 "code_complexity:hard"

**工作原理**：

1. 将查询 embedding 与每个规则的描述进行比较
2. 选择最匹配的规则（描述相似度最高）
3. 在该规则内，将查询与困难和简单候选进行比较
4. 难度信号 = max_hard_similarity - max_easy_similarity
5. 如果信号 > 阈值："hard"，如果信号 < -阈值："easy"，否则："medium"

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

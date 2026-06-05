---
translation:
  source_commit: "bac2743"
  source_file: "docs/tutorials/intelligent-route/preference-routing.md"
  outdated: false
---

# 偏好信号路由 (Preference Signal Routing)

本指南向您展示如何使用基于 LLM 的偏好匹配来路由请求。偏好信号使用外部 LLM 分析复杂的意图，并做出细致的路由决策。

## 关键优势

- **复杂意图分析**：使用 LLM 推理进行细致的路由决策
- **灵活的逻辑**：使用自然语言定义路由偏好
- **高准确度**：对复杂意图的检测准确率达 90-98%
- **可扩展性**：无需重新训练模型即可添加新的偏好

## 它解决了什么问题？

有些路由决策对于简单的模式匹配或分类来说过于复杂：

- **细微的意图**："解释量子力学的哲学意义"
- **多方面查询**："比较和对比功利主义与义务论"
- **上下文相关**："解决这个问题的最佳方法是什么？"

偏好信号使用外部 LLM 来分析这些复杂查询，并将其与路由偏好进行匹配，使您能够：

1. 处理其他信号会漏掉的复杂意图
2. 根据 LLM 推理做出细致的路由决策
3. 使用自然语言定义路由逻辑
4. 无需重新训练即可适应新的用例

## 配置

### 基础配置

在您的 `config.yaml` 中定义偏好信号：

```yaml
signals:
  preferences:
    - name: "code_generation"
      description: "生成新的代码片段、编写函数、创建类"

    - name: "bug_fixing"
      description: "识别并修复错误、调试问题、排除故障"

    - name: "code_review"
      description: "审查代码质量、提供改进建议、最佳实践"

    - name: "other"
      description: "无关的查询或已满足的请求"
```

### 外部 LLM 配置

在 `router-defaults.yaml` 中配置用于偏好匹配的外部 LLM：

```yaml
# 外部模型配置
# 用于高级路由信号，如通过外部 LLM 进行的基于偏好的路由
external_models:
  - llm_provider: "vllm"
    model_role: "preference"
    llm_endpoint:
      address: "127.0.0.1"
      port: 8000
    llm_model_name: "openai/gpt-oss-120b"
    llm_timeout_seconds: 30
    parser_type: "json"
    access_key: ""  # 可选：用于 Authorization Header (Bearer token)
```

### 在决策规则中使用

```yaml
decisions:
  - name: preference_code_generation
    description: "根据 LLM 偏好匹配路由代码生成请求"
    priority: 200
    rules:
      operator: "AND"
      conditions:
        - type: "preference"
          name: "code_generation"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: false
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "你是一位代码生成专家。请编写简洁、高效且文档齐全的代码。"

  - name: preference_bug_fixing
    description: "根据 LLM 偏好匹配路由错误修复请求"
    priority: 200
    rules:
      operator: "AND"
      conditions:
        - type: "preference"
          name: "bug_fixing"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: true
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "你是一位调试专家。请仔细分析问题，找出根本原因，并提供清晰的修复方案和解释。请逐步思考并在回答前验证你的推理。"
```

## 工作原理

### 1. 查询分析

外部 LLM 分析查询：

```
查询: "解释量子力学的哲学意义"

LLM 分析:
- 是否需要深度推理: 是
- 复杂度级别: 高
- 领域: 哲学 + 物理
- 推理类型: 分析性、概念性
```

### 2. 偏好匹配

LLM 将查询与定义的偏好进行匹配：

```yaml
preferences:
  - name: "complex_reasoning"
    description: "需要深度推理和分析"
    # LLM 评估：此查询是否需要深度推理？
    # 结果：是 (置信度: 0.95)
```

### 3. 路由决策

根据匹配结果路由查询：

```
偏好匹配成功: complex_reasoning (0.95)
决策: deep_reasoning (深度推理)
模型: reasoning-specialist (推理专家模型)
```

## 用例

### 1. 学术研究 - 复杂分析

**问题**：研究类查询需要深度推理和分析

```yaml
signals:
  preferences:
    - name: "research_analysis"
      description: "需要深度分析和批判性思维的学术研究"

  domains:
    - name: "philosophy"
      description: "哲学查询"
      mmlu_categories: ["philosophy", "formal_logic"]

decisions:
  - name: academic_research
    description: "路由学术研究查询"
    priority: 200
    rules:
      operator: "AND"
      conditions:
        - type: "domain"
          name: "philosophy"
        - type: "preference"
          name: "research_analysis"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: true
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "你是一位学术研究专家，在批判性分析和哲学推理方面拥有专业知识。"
```

**示例查询**：

- "分析康德《批判》的认识论意义" → ✅ 复杂分析
- "什么是哲学？" → ❌ 简单定义

### 2. 商业战略 - 决策制定

**问题**：战略性查询需要细致的分析

```yaml
signals:
  preferences:
    - name: "strategic_thinking"
      description: "需要多维度分析的商业战略"

  keywords:
    - name: "business_keywords"
      operator: "OR"
      keywords: ["战略", "市场", "竞争", "增长", "strategy"]
      case_sensitive: false

decisions:
  - name: strategic_analysis
    description: "路由战略商业查询"
    priority: 200
    rules:
      operator: "AND"
      conditions:
        - type: "keyword"
          name: "business_keywords"
        - type: "preference"
          name: "strategic_thinking"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: true
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "你是一位资深商业战略家，在市场分析和竞争战略方面拥有专业知识。"
```

**示例查询**：

- "分析我们的竞争地位并推荐增长策略" → ✅ 战略性
- "我们的收入是多少？" → ❌ 简单查询

### 3. 技术架构 - 设计决策

**问题**：架构决策需要深度的技术推理

```yaml
signals:
  preferences:
    - name: "architecture_design"
      description: "需要设计思维和权衡分析的技术架构"

  keywords:
    - name: "architecture_keywords"
      operator: "OR"
      keywords: ["架构", "设计", "扩展性", "性能", "architecture"]
      case_sensitive: false

decisions:
  - name: architecture_analysis
    description: "路由架构设计查询"
    priority: 200
    rules:
      operator: "AND"
      conditions:
        - type: "keyword"
          name: "architecture_keywords"
        - type: "preference"
          name: "architecture_design"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: true
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "你是一位技术架构专家，在系统设计、可扩展性和性能优化方面拥有专业知识。"
```

**示例查询**：

- "设计一个可扩展的微服务架构并进行权衡分析" → ✅ 设计思维
- "什么是微服务？" → ❌ 简单定义

## 性能特征

| 维度 | 值 |
|--------|-------|
| 延迟 | 100-500ms (取决于 LLM) |
| 准确度 | 90-98% |
| 成本 | 较高 (调用外部 LLM) |
| 可扩展性 | 受限于 LLM 端点 |

## 最佳实践

### 1. 作为最后手段使用

偏好信号成本昂贵。请优先使用其他信号：

```yaml
decisions:
  - name: simple_math
    priority: 10
    rules:
      operator: "OR"
      conditions:
        - type: "keyword"
          name: "math_keywords"  # 快速、廉价
    
  - name: complex_reasoning
    priority: 5
    rules:
      operator: "OR"
      conditions:
        - type: "preference"
          name: "complex_reasoning"  # 慢、昂贵
```

### 2. 结合其他信号

使用 AND 运算符以减少误报：

```yaml
rules:
  operator: "AND"
  conditions:
    - type: "domain"
      name: "philosophy"  # 快速预过滤
    - type: "preference"
      name: "complex_reasoning"  # 昂贵的验证
```

### 3. 缓存 LLM 响应

启用缓存以降低延迟和成本：

```yaml
preferences:
  - name: "complex_reasoning"
    description: "需要深度推理"
    llm_endpoint: "http://localhost:11434"
    cache_enabled: true
    cache_ttl: 3600  # 1 小时
```

### 4. 设置适当的超时

防止缓慢的 LLM 调用造成阻塞：

```yaml
preferences:
  - name: "complex_reasoning"
    description: "需要深度推理"
    llm_endpoint: "http://localhost:11434"
    timeout: 2000  # 2 秒
    fallback_on_timeout: false  # 超时时不匹配
```

### 5. 监控性能

跟踪 LLM 调用的延迟和准确度：

```yaml
logging:
  level: info
  preference_signals: true
  llm_latency: true
```

## 高级配置

### 多个 LLM 端点

针对不同的偏好使用不同的 LLM：

```yaml
signals:
  preferences:
    - name: "complex_reasoning"
      description: "深度推理"
      llm_endpoint: "http://localhost:11434"
      model: "llama3-70b"  # 复杂推理使用大模型
    
    - name: "simple_classification"
      description: "简单的意图分类"
      llm_endpoint: "http://localhost:11435"
      model: "llama3-8b"  # 简单任务使用小模型
```

### 自定义提示词

自定义 LLM 提示词以提高准确度：

```yaml
preferences:
  - name: "complex_reasoning"
    description: "需要深度推理"
    llm_endpoint: "http://localhost:11434"
    prompt_template: |
      分析以下查询，并判断它是否需要深度的推理和分析。
      查询：{query}
      回答 YES 或 NO，并说明理由。
```

## 参考

完整信号架构请参见 [信号驱动决策架构](../../overview/signal-driven-decisions.md)。

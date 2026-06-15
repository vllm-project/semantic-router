---
translation:
  source_commit: "bac2743"
  source_file: "docs/tutorials/intelligent-route/fact-check-routing.md"
  outdated: false
---

# 事实核查信号路由 (Fact Check Signal Routing)

本指南向您展示如何根据请求是否需要事实验证来进行路由。`fact_check` 信号有助于识别需要幻觉检测或事实核查的事实性查询。

## 关键优势

- **自动检测**：基于机器学习检测事实性查询与创意/代码类查询
- **幻觉防御**：将事实性查询路由到具有验证功能的模型
- **资源优化**：仅在需要时应用昂贵的事实核查
- **合规性**：确保受监管行业的事实准确性

## 它解决了什么问题？

并非所有查询都需要事实验证：

- **事实性查询**："法国的首都是哪里？" → 需要验证
- **创意性查询**："写一个关于龙的故事" → 不需要验证
- **代码类查询**："写一个 Python 函数" → 不需要验证

`fact_check` 信号会自动识别哪些查询需要事实验证，使您能够：

1. 将事实性查询路由到具有幻觉检测的模型
2. 仅为事实性查询启用事实核查插件
3. 通过避免不必要的验证来优化成本

## 配置

### 基础配置

在您的 `config.yaml` 中定义事实核查信号：

```yaml
signals:
  fact_check:
    - name: needs_fact_check
      description: "查询包含应根据上下文进行验证的事实陈述"

    - name: no_fact_check_needed
      description: "查询是创意、代码相关或基于观点的 - 无需事实验证"
```

### 在决策规则中使用

```yaml
decisions:
  - name: factual_queries
    description: "路由带验证的事实性查询"
    priority: 150
    rules:
      operator: "AND"
      conditions:
        - type: "fact_check"
          name: "needs_fact_check"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: true
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "你是一位事实信息专家。请提供准确、可验证的信息，并在可能时提供来源。"
      - type: "hallucination"
        configuration:
          enabled: true
          threshold: 0.7
```

## 用例

### 1. 医疗保健 - 医疗信息

**问题**：医疗查询必须事实准确，以避免造成伤害

```yaml
signals:
  fact_check:
    - name: needs_fact_check
      description: "查询包含应验证的事实陈述"

  domains:
    - name: "health"
      description: "医疗和健康查询"
      mmlu_categories: ["health"]

decisions:
  - name: verified_medical
    description: "带事实验证的医疗查询"
    priority: 200
    rules:
      operator: "AND"
      conditions:
        - type: "domain"
          name: "health"
        - type: "fact_check"
          name: "needs_fact_check"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: true
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "你是一位医疗信息专家。请提供准确的、基于证据的健康信息。"
      - type: "hallucination"
        configuration:
          enabled: true
          threshold: 0.8  # 医疗类设置高阈值
```

**示例查询**：

- "糖尿病的症状有哪些？" → ✅ 路由并验证
- "写一个关于医生的故事" → ❌ 创意类，不验证

### 2. 金融服务 - 投资信息

**问题**：财务建议必须准确，以符合监管要求

```yaml
signals:
  fact_check:
    - name: needs_fact_check
      description: "查询包含应验证的事实陈述"

  keywords:
    - name: "financial_keywords"
      operator: "OR"
      keywords: ["股票", "投资", "投资组合", "股息"]
      case_sensitive: false

decisions:
  - name: verified_financial
    description: "带验证的金融查询"
    priority: 200
    rules:
      operator: "AND"
      conditions:
        - type: "keyword"
          name: "financial_keywords"
        - type: "fact_check"
          name: "needs_fact_check"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: true
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "你是一位金融信息专家。请提供准确的金融信息，并附带适当的免责声明。"
      - type: "hallucination"
        configuration:
          enabled: true
          threshold: 0.8
```

**示例查询**：

- "苹果公司当前的市盈率是多少？" → ✅ 事实性，已验证
- "解释投资策略" → ❌ 一般建议，不验证

### 3. 教育 - 历史事实

**问题**：教育内容必须事实准确

```yaml
signals:
  fact_check:
    - name: needs_fact_check
      description: "查询包含应验证的事实陈述"

  domains:
    - name: "history"
      description: "历史查询"
      mmlu_categories: ["history"]

decisions:
  - name: verified_history
    description: "带验证的历史查询"
    priority: 150
    rules:
      operator: "AND"
      conditions:
        - type: "domain"
          name: "history"
        - type: "fact_check"
          name: "needs_fact_check"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: true
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "你是一位历史专家。请提供准确的、带有恰当背景的历史信息。"
      - type: "hallucination"
        configuration:
          enabled: true
          threshold: 0.7
```

**示例查询**：

- "第二次世界大战什么时候结束的？" → ✅ 事实性，已验证
- "写一个历史小说故事" → ❌ 创意类，不验证

## 性能特征

| 维度 | 值 |
|--------|-------|
| 延迟 | 20-50ms |
| 准确度 | 80-90% |
| 误报率 (False Positives) | 5-10% (创意类被标记为事实性) |
| 漏报率 (False Negatives) | 5-10% (事实性被标记为创意类) |

## 最佳实践

### 1. 结合领域信号

同时使用 `fact_check` 和领域信号以提高准确性：

```yaml
rules:
  operator: "AND"
  conditions:
    - type: "domain"
      name: "science"
    - type: "fact_check"
      name: "needs_verification"
```

### 2. 设置合适的优先级

事实性查询应具有更高的优先级：

```yaml
decisions:
  - name: verified_factual
    priority: 100  # 高优先级
    rules:
      operator: "AND"
      conditions:
        - type: "fact_check"
          name: "needs_verification"
```

### 3. 启用幻觉检测

始终为事实性查询启用幻觉插件：

```yaml
plugins:
  - type: "hallucination"
    configuration:
      enabled: true
      threshold: 0.7
```

### 4. 监控误报/漏报

跟踪被误分类的查询：

```yaml
logging:
  level: debug
  fact_check: true
```

## 参考

完整信号架构请参见 [信号驱动决策架构](../../overview/signal-driven-decisions.md)。

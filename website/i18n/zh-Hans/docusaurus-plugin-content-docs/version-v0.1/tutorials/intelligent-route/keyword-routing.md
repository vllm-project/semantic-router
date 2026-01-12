---
translation:
  source_commit: "bac2743"
  source_file: "docs/tutorials/intelligent-route/keyword-routing.md"
  outdated: false
---

# Keyword Based Routing

本指南向您展示如何使用显式 keyword rule 和 regex 模式来路由请求。Keyword routing 提供透明、可审计的路由 decision，这对于 compliance、security 和需要可解释 AI 的场景至关重要。

## 关键优势

- **透明**：路由 decision 完全可解释且可审计
- **合规**：确定性行为满足监管要求（GDPR, HIPAA, SOC2）
- **快速**：亚毫秒级 latency，无 ML 推理开销
- **可解释**：清晰的 rule 使 debug 和验证变得简单直接

## 它解决了什么问题

基于 ML 的 classification 是一个难以审计和解释的黑箱。Keyword routing 提供：

- **可解释的 decision**：确切知道为什么一个查询被路由到特定 category
- **监管 compliance**：审计人员可以验证路由逻辑是否符合要求
- **确定性行为**：相同的 input 始终产生相同的 output
- **零 latency**：无需 model 推理，瞬时 classification
- **精确控制**：针对 security、compliance 和业务逻辑的显式 rule

## 何时使用

- 需要 audit trail 的**受监管行业**（金融、医疗、法律）
- 需要确定性 PII 检测的**security/compliance**场景
- 亚毫秒级 latency 至关重要的**高吞吐量系统**
- 具有清晰 keyword 指示的**紧急/优先级路由**
- 匹配 regex 模式的**结构化数据**（email、ID、文件路径）

## 配置

在您的 `config.yaml` 中添加 keyword signal：

```yaml
# 定义 keyword signal
signals:
  keywords:
    - name: "urgent_keywords"
      operator: "OR"  # 匹配任意关键词
      keywords: ["urgent", "immediate", "asap", "emergency", "紧急"]
      case_sensitive: false

    - name: "sensitive_data_keywords"
      operator: "OR"
      keywords: ["SSN", "social security", "credit card", "password", "身份证"]
      case_sensitive: false

    - name: "spam_keywords"
      operator: "OR"
      keywords: ["buy now", "free money", "click here", "免费领取"]
      case_sensitive: false

# 使用 keyword signal 定义 decision
decisions:
  - name: urgent_request
    description: "路由紧急请求"
    priority: 100  # 高优先级
    rules:
      operator: "OR"
      conditions:
        - type: "keyword"
          name: "urgent_keywords"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: false
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "你是一位响应迅速的助手，专门处理紧急请求。"

  - name: sensitive_data
    description: "路由敏感数据查询"
    priority: 90
    rules:
      operator: "OR"
      conditions:
        - type: "keyword"
          name: "sensitive_data_keywords"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: false
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "你是一位具有安全意识的助手，专门处理敏感数据。"

  - name: filter_spam
    description: "拦截垃圾查询"
    priority: 95
    rules:
      operator: "OR"
      conditions:
        - type: "keyword"
          name: "spam_keywords"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: false
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "此查询似乎是垃圾信息。请提供礼貌的回复。"
```

## 运算符

- **OR**：如果找到任意一个 keyword 则匹配
- **AND**：仅当找到所有 keyword 时才匹配
- **NOR**：仅当未找到任何 keyword 时才匹配（排除）

## 请求示例

```bash
# 紧急请求（匹配 "urgent"）
curl -X POST http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MoM",
    "messages": [{"role": "user", "content": "I need urgent help with my account"}]
  }'

# 敏感数据（匹配所有关键词）
curl -X POST http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MoM",
    "messages": [{"role": "user", "content": "My SSN and credit card were stolen"}]
  }'
```

## 真实世界用例

### 1. 金融服务（透明合规）

**问题**：监管机构要求为 audit trail 提供可解释的路由 decision
**解决方案**：Keyword rule 为每个路由 decision 提供清晰的 "原因"（例如，"SSN" keyword → 安全处理器）
**影响**：通过了 SOC2 审计，实现了 decision 的完全透明

### 2. 医疗平台（合规的 PII 检测）

**问题**：HIPAA 要求确定性的、可审计的 PII 检测
**解决方案**：AND operator 通过记录的 rule 检测多个 PII 指示符
**影响**：100% 确定性，提供符合 compliance 要求的完整 audit trail

### 3. 高频交易（亚毫秒级路由）

**问题**：实时市场数据路由需要 \<1ms 的 classification latency
**解决方案**：Keyword matching 提供即时 classification，无 ML 开销
**影响**：0.1ms latency，可处理 100K+ 次请求/秒

### 4. 政府服务（可解释的规则）

**问题**：公民需要了解为什么请求被路由或拒绝
**解决方案**：清晰的 keyword rule 可以用通俗易懂的语言进行解释
**影响**：减少了投诉，实现了 decision 制定的透明化

### 5. 企业安全（透明的威胁检测）

**问题**：Security team 需要了解为什么查询被 flag
**解决方案**：针对威胁模式使用显式的 keyword/regex rule，并附带清晰的文档
**影响**：Security team 可以自信地验证和更新 rule

## 性能优势

- **亚毫秒级 latency**：无 ML 推理开销
- **高吞吐量**：单核可处理 100K+ 次请求/秒
- **成本可预测**：无需 GPU 或 embedding model
- **零冷启动**：第一个请求即可实现即时 classification

## 参考

完整配置请参见 [keyword.yaml](https://github.com/vllm-project/semantic-router/blob/main/config/intelligent-routing/in-tree/keyword.yaml)。

---
translation:
  source_commit: "bac2743"
  source_file: "docs/tutorials/intelligent-route/user-feedback-routing.md"
  outdated: false
---

# 用户反馈信号路由 (User Feedback Signal Routing)

本指南向您展示如何根据用户反馈和满意度信号来路由请求。`user_feedback` 信号有助于识别后续消息、更正和满意度水平。

## 关键优势

- **自适应路由**：检测用户何时不满意，并路由到更强大的模型
- **更正处理**：自动处理 "错了"、"再试一次" 等消息
- **满意度分析**：识别正面与负面反馈
- **改进的用户体验 (UX)**：当用户表示不满意时提供更好的响应

## 它解决了什么问题？

用户经常在后续消息中提供反馈：

- **更正**："不对"、"不，我不是那个意思"
- **满意**："谢谢"、"很有帮助"、"完美"
- **澄清**："你能多解释一下吗？"、"我不明白"
- **重试**："再试一次"、"给我另一个答案"

`user_feedback` 信号会自动识别这些模式，使您能够：

1. 将更正请求路由到更强大的模型
2. 检测满意度水平以供监控
3. 适当地处理后续问题
4. 根据反馈提高响应质量

## 配置

### 基础配置

在您的 `config.yaml` 中定义用户反馈信号：

```yaml
signals:
  user_feedbacks:
    - name: "wrong_answer"
      description: "用户表示之前的回答不正确"

    - name: "satisfied"
      description: "用户对回答表示满意"

    - name: "need_clarification"
      description: "用户需要对回答进行更多澄清"

    - name: "want_different"
      description: "用户想要其他不同的答案"
```

### 在决策规则中使用

```yaml
decisions:
  - name: wrong_answer_route
    description: "处理表示错误答案的用户反馈 - 重新思考并提供正确的响应"
    priority: 150
    rules:
      operator: "AND"
      conditions:
        - type: "user_feedback"
          name: "wrong_answer"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: true
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "用户已表示之前的回答不正确。请仔细重新考虑问题，找出之前响应中可能存在的错误，并提供经过更正且准确的答案。请逐步思考并在回答前验证你的推理。"

  - name: retry_with_different_approach
    description: "路由寻求不同方法的请求"
    priority: 100
    rules:
      operator: "AND"
      conditions:
        - type: "user_feedback"
          name: "want_different"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: true
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "用户想要不同的方法或视角。请提供一个与之前响应不同的替代方案或解释。"
```

## 反馈类型

### 1. 错误回答 (Wrong Answer)

**模式**："错了"、"不对"、"不正确"、"再试一次"

```yaml
signals:
  user_feedbacks:
    - name: "wrong_answer"
      description: "用户表示之前的回答不正确"

decisions:
  - name: wrong_answer_route
    description: "处理表示错误回答的用户反馈"
    priority: 150
    rules:
      operator: "AND"
      conditions:
        - type: "user_feedback"
          name: "wrong_answer"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: true
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "用户已表示之前的回答不正确。请仔细重新考虑问题并提供更正后的答案。"
```

**示例查询**：

- "错了，答案是 42" → ✅ 检测到更正
- "不，我不是那个意思" → ✅ 检测到更正
- "换个方法再试一次" → ✅ 检测到更正

### 2. 满意 (Satisfied)

**模式**："谢谢"、"完美"、"很有帮助"、"太棒了"

```yaml
signals:
  user_feedbacks:
    - name: "satisfied"
      description: "用户对回答表示满意"

decisions:
  - name: track_satisfaction
    description: "跟踪满意的用户"
    priority: 50
    rules:
      operator: "AND"
      conditions:
        - type: "user_feedback"
          name: "satisfied"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: false
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "用户感到满意。请继续提供有用的协助。"
```

**示例查询**：

- "谢谢，这正是我需要的" → ✅ 检测到满意
- "完美，这很有帮助" → ✅ 检测到满意
- "解释得太棒了" → ✅ 检测到满意

### 3. 需要澄清 (Need Clarification)

**模式**："你能多解释一下吗？"、"我不明白"、"你是什么意思？"

```yaml
signals:
  user_feedbacks:
    - name: "need_clarification"
      description: "用户需要对回答进行更多澄清"

decisions:
  - name: provide_clarification
    description: "提供更详细的解释"
    priority: 100
    rules:
      operator: "AND"
      conditions:
        - type: "user_feedback"
          name: "need_clarification"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: true
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "用户需要更多澄清。请提供更详细的、带有示例的逐步解释。"
```

**示例查询**：

- "你能用更简单的术语解释一下吗？" → ✅ 需要澄清
- "最后一部分我没听懂" → ✅ 需要澄清
- "你这话是什么意思？" → ✅ 需要澄清

### 4. 想要不同的方法 (Want Different Approach)

**模式**："换个答案"、"换种方式"、"给我看看其他选择"

```yaml
signals:
  user_feedbacks:
    - name: "want_different"
      description: "用户想要其他不同的答案"

decisions:
  - name: retry_with_different_approach
    description: "提供替代方案"
    priority: 100
    rules:
      operator: "AND"
      conditions:
        - type: "user_feedback"
          name: "want_different"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: true
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "用户想要不同的方法或视角。请提供一个与之前响应不同的替代方案或解释。"
```

**示例查询**：

- "再给我一个解决这个问题的方法" → ✅ 想要替代方案
- "给我展示一个不同的方法" → ✅ 想要替代方案
- "你能试试别的办法吗？" → ✅ 想要替代方案

## 用例

### 1. 客户支持 - 升级

**问题**：不满意的客户需要更好的响应

```yaml
signals:
  user_feedbacks:
    - name: "wrong_answer"
      description: "客户表示之前的回答不正确"

decisions:
  - name: escalate_to_premium
    description: "升级到高级模型"
    priority: 150
    rules:
      operator: "AND"
      conditions:
        - type: "user_feedback"
          name: "wrong_answer"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: true
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "客户对之前的回答不满意。请提供一个更好、更准确的响应。"
```

### 2. 教育 - 自适应学习

**问题**：学生在困惑时需要不同的解释

```yaml
signals:
  user_feedbacks:
    - name: "need_clarification"
      description: "学生需要对回答进行更多澄清"

decisions:
  - name: detailed_explanation
    description: "提供详细解释"
    priority: 100
    rules:
      operator: "AND"
      conditions:
        - type: "user_feedback"
          name: "need_clarification"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: true
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "学生需要更多澄清。请提供详细的、带有示例的逐步解释。"
```

## 最佳实践

### 1. 结合上下文

使用对话历史记录来改进检测：

```yaml
# 跟踪对话状态
context:
  previous_response: true
  conversation_history: 3  # 最近 3 条消息
```

### 2. 设置升级优先级

更正请求应具有高优先级：

```yaml
decisions:
  - name: handle_correction
    priority: 100  # 更正请求的高优先级
```

### 3. 监控满意度

跟踪反馈模式：

```yaml
logging:
  level: info
  user_feedback: true
  satisfaction_metrics: true
```

### 4. 使用合适的模型

- **更正**：路由到能力更强/更昂贵的模型
- **澄清**：路由到擅长解释的模型
- **满意**：继续使用当前模型

## 参考

完整信号架构请参见 [信号驱动决策架构](../../overview/signal-driven-decisions.md)。

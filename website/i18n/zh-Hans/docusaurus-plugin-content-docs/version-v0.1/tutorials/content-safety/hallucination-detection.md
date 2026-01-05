---
translation:
  source_commit: "4e2810a"
  source_file: "docs/tutorials/content-safety/hallucination-detection.md"
  outdated: false
---

# 幻觉检测 (Hallucination Detection)

Semantic Router 提供了先进的幻觉检测功能，以验证 LLM 的响应是否基于提供的上下文。系统使用经过微调的 ModernBERT token 分类器来识别未被检索结果或工具输出支持的陈述。

## 概览

幻觉检测系统：

- **验证** LLM 响应是否基于提供的上下文（RAG 结果、工具输出）
- **识别** token 级别的未支持陈述
- **提供** 使用 NLI（自然语言推理）的详细解释
- **警告或拦截** 当检测到幻觉时
- **无缝集成** 到 RAG 和工具调用 (tool-calling) 工作流中

## 工作原理

幻觉检测在三阶段管道中运行：

1. **事实核查分类 (Fact-Check Classification)**：确定查询是否需要事实验证（事实性问题 vs. 创意/观点类问题）
2. **Token 级检测**：分析 LLM 响应以识别未支持的陈述
3. **NLI 解释**（可选）：为幻觉片段提供详细的推理

## 配置

### 全局模型配置

首先，在 `router-defaults.yaml` 中配置幻觉检测模型：

```yaml
# router-defaults.yaml
# 幻觉缓解配置
# 默认禁用 - 在决策中通过 hallucination 插件启用
hallucination_mitigation:
  enabled: false

  # 事实核查分类器：确定提示词是否需要事实验证
  fact_check_model:
    model_id: "models/mom-halugate-sentinel"
    threshold: 0.6
    use_cpu: true

  # 幻觉检测器：验证 LLM 响应是否基于上下文
  hallucination_model:
    model_id: "models/mom-halugate-detector"
    threshold: 0.8
    use_cpu: true

  # NLI 模型：为幻觉片段提供解释
  nli_model:
    model_id: "models/mom-halugate-explainer"
    threshold: 0.9
    use_cpu: true
```

### 在决策中启用幻觉检测

使用 `hallucination` 插件为每个决策启用幻觉检测：

```yaml
# config.yaml
decisions:
  - name: "general_decision"
    description: "带有事实核查的通用问题"
    priority: 100
    rules:
      operator: "OR"
      conditions:
        - type: "domain"
          name: "general"
        - type: "fact_check"
          name: "needs_fact_check"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: false
    plugins:
      - type: "hallucination"
        configuration:
          enabled: true
          use_nli: true  # 启用 NLI 以获得详细解释
          # 检测到幻觉时的操作: "header", "body", "block", 或 "none"
          hallucination_action: "header"
          # 需要事实核查但没有工具上下文时的操作: "header", "body", 或 "none"
          unverified_factual_action: "header"
          # 在正文警告中包含详细信息（置信度、片段）
          include_hallucination_details: true
```

### 插件配置选项

| 选项 | 值 | 描述 |
|---------------------------------|-------------------------------------|----------------------------------------------------------|
| `enabled` | `true`, `false` | 为此决策启用/禁用幻觉检测 |
| `use_nli` | `true`, `false` | 使用 NLI 模型进行详细解释 |
| `hallucination_action` | `header`, `body`, `block`, `none` | 检测到幻觉时的操作 |
| `unverified_factual_action` | `header`, `body`, `none` | 需要事实核查但无可用上下文时的操作 |
| `include_hallucination_details` | `true`, `false` | 在响应正文中包含置信度和片段 |

### 操作模式

| 操作 | 行为 | 用例 |
|----------|-----------------------------------------------|---------------------------------------|
| `header` | 添加警告 Header，允许响应 | 开发、监控 |
| `body` | 在响应正文中添加警告，允许响应 | 面向用户的警告 |
| `block` | 返回错误，拦截响应 | 生产环境、高风险应用 |
| `none` | 无操作，仅记录日志 | 静默监控 |

## 幻觉检测如何工作

处理请求时：

1. **事实核查分类**：哨兵模型 (sentinel model) 确定查询是否需要事实验证
2. **上下文提取**：从 LLM 响应中捕获工具结果或 RAG 上下文
3. **幻觉检测**：如果上下文可用，检测器将分析响应
4. **操作**：根据配置，系统添加 Header、修改正文或拦截响应

检测到幻觉时的响应 Header：

```http
X-Hallucination-Detected: true
X-Hallucination-Confidence: 0.85
X-Unsupported-Spans: "Paris was founded in 1492"
```

## 用例

### RAG (检索增强生成)

验证 LLM 响应是否基于检索到的文档：

```yaml
plugins:
  - type: "hallucination"
    configuration:
      enabled: true
      use_nli: false
      hallucination_action: "header"
      unverified_factual_action: "header"
```

**示例**：客户支持机器人检索文档并生成答案。幻觉检测确保响应不包含文档中不存在的信息。

### 工具调用工作流

验证 LLM 响应是否准确反映了工具输出：

```yaml
plugins:
  - type: "hallucination"
    configuration:
      enabled: true
      use_nli: true
      hallucination_action: "block"
      unverified_factual_action: "header"
      include_hallucination_details: true
```

**示例**：AI 智能体调用数据库查询工具。幻觉检测防止智能体捏造查询未返回的数据。

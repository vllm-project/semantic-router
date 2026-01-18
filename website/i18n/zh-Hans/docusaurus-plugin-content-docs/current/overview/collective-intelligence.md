---
translation:
  source_commit: "bac2743"
  source_file: "docs/overview/collective-intelligence.md"
  outdated: true
sidebar_position: 3
---

# 什么是 Collective Intelligence？

**Collective Intelligence** 是指当多个模型、signal 和 decision 过程作为一个统一系统协同工作时产生的涌现智能。

## 核心理念

就像专家团队比任何个人专家都能更好地解决问题一样，专门的 LLM 系统比任何单一模型都能提供更好的结果。

### 传统方法：单一模型

```
用户查询 → 单一 LLM → 响应
```

**局限性**：

- 一个模型试图擅长所有事情
- 没有专业化或优化
- 简单和复杂的任务使用相同的模型
- 不从模式中学习

### 集体智能方法：模型系统

```
用户查询 → Signal Extraction → Decision Engine → 最佳模型 → 响应
              ↓          ↓            ↓
           7 种 signal 类型   AND/OR 规则   专业化模型
              ↓          ↓            ↓
           上下文分析     智能选择      Plugin Chain
```

**优势**：

- 每个模型专注于它最擅长的事情
- 系统从所有交互的模式中学习
- 基于多种信号的自适应路由
- 信号融合带来的涌现智能

## 集体智能如何涌现

### 1. Signal 多样性

不同的 signal 捕获智能的不同方面：

| 信号类型 | 智能方面 |
|------------|-------------------|
| **keyword** | 模式识别 |
| **embedding** | 语义理解 |
| **domain** | 知识分类 |
| **fact_check** | 真理验证需求 |
| **user_feedback** | 用户满意度 |
| **preference** | 意图匹配 |
| **language** | 多语言检测 |

**Collective 效益**：signal 的组合提供了比任何单一 signal 更丰富的理解。

### 2. Decision 融合

Signal 使用逻辑运算符进行组合：

```yaml
# 示例：具有多种信号的数学路由
decisions:
  - name: advanced_math
    rules:
      operator: "AND"
      conditions:
        - type: "keyword"
          name: "math_keywords"
        - type: "domain"
          name: "mathematics"
        - type: "embedding"
          name: "math_intent"
```

**Collective 效益**：多个 signal 共同投票比任何单一 signal 做出更准确的决策。

### 3. 模型专业化

不同的模型贡献其优势：

```yaml
modelRefs:
  - model: qwen-math      # 最擅长数学推理
    weight: 1.0
  - model: deepseek-coder # 最擅长代码生成
    weight: 1.0
  - model: claude-creative # 最擅长创意写作
    weight: 1.0
```

**Collective 效益**：系统级智能通过路由到正确的专家而涌现。

### 4. Plugin 协作

Plugin 协同工作以增强响应：

```yaml
plugins:
  - type: "semantic-cache"    # 速度优化
  - type: "jailbreak"         # 安全层
  - type: "pii"               # 隐私保护
  - type: "system_prompt"     # 上下文注入
  - type: "hallucination"     # 质量保证
```

**Collective 效益**：多层处理创建了一个更健壮和安全的系统。

## 真实世界示例

让我们看看实际中的 Collective Intelligence：

### 用户查询

```
"证明 2 的平方根是无理数"
```

### 信号提取

```yaml
signals_detected:
  keyword: ["prove", "square root", "irrational"]  # 检测到数学关键词
  embedding: 0.89                                   # 与数学查询的高度相似性
  domain: "mathematics"                             # MMLU 分类
  fact_check: true                                  # 证明需要验证
```

### 决策过程

```yaml
decision_made: "advanced_math"
reason: "All math signals agree (keyword + embedding + domain)" # 所有数学信号一致
confidence: 0.95
```

### 模型选择

```yaml
selected_model: "qwen-math"
reason: "Specialized in mathematical proofs" # 专注于数学证明
```

### 插件链

```yaml
plugins_applied:
  - semantic-cache: "Cache miss, proceeding" # 缓存未命中，继续
  - jailbreak: "No adversarial patterns detected" # 未检测到对抗性模式
  - system_prompt: "Added: 'Provide rigorous mathematical proof'" # 已添加：'提供严格的数学证明'
  - hallucination: "Enabled for fact verification" # 启用以进行事实验证
```

### 结果

- **准确**：路由到数学专家
- **快速**：首先检查缓存
- **安全**：验证无越狱尝试
- **高质量**：启用了幻觉检测

**这就是 Collective Intelligence**：没有任何单一组件做出决策。智能从 signal、rule、model 和 plugin 的协作中涌现。

## Collective Intelligence 的优势

### 1. 更好的准确性

- 多种 signal 减少误报
- 专业化模型在其领域表现更好
- Signal 融合捕获边缘情况

### 2. 提高鲁棒性

- 即使一个 signal 失败，系统仍能继续工作
- 多层安全提供纵深防御
- Fallback 机制确保可靠性

### 3. 持续学习

- 系统从所有交互的模式中学习
- Feedback signal 改进未来的路由
- Collective knowledge 随时间增长

### 4. 涌现能力

- 系统可以处理单一组件未设计处理的情况
- 新模式从 signal 组合中涌现
- 智能随系统复杂性扩展

## 下一步

- [什么是 Signal-Driven Decision？](signal-driven-decisions.md) - 深入解 Decision Engine
- [配置指南](../installation/configuration.md) - 设置您自己的 Collective Intelligence 系统
- [智能路由教程](../tutorials/intelligent-route/keyword-routing.md) - 学习配置 signal

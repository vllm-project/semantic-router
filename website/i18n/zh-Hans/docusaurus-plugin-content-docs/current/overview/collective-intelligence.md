---
translation:
  source_commit: "ab2aa160"
  source_file: "docs/overview/collective-intelligence.md"
  outdated: true
sidebar_position: 3
---

# 什么是集体智能？

**集体智能**是指多个模型、信号与决策过程作为统一系统协同工作时，所涌现出的智能。

## 核心思想

正如专家团队能比任何单一专家更好地解决问题，一组专用 LLM 也能比单一模型提供更优结果。

### 传统做法：单模型

```
用户查询 → 单一 LLM → 响应
```

**局限**：

- 一个模型试图面面俱到
- 缺少专精与优化
- 简单与复杂任务用同一模型
- 难以从模式中持续学习

### 集体智能：模型系统

```
用户查询 → 信号提取 → 投影协调 → 决策引擎 → 插件 + 模型分发 → 响应
              ↓                    ↓                         ↓                         ↓
        16 类信号族           分区 / 打分 / 映射           布尔策略               专用模型
```

**优势**：

- 各模型专注所长
- 系统从全量交互中学习模式
- 基于多信号的自适应路由
- 信号融合带来涌现能力

## 集体智能如何产生

### 1. 信号多样性

不同信号刻画智能的不同侧面：

| 信号族分组 | 智能侧面 |
| ---------- | -------- |
| **启发式**（`authz`、`context`、`keyword`、`language`、`structure`） | 快速的请求形态、区域与策略门禁 |
| **学习型**（ `complexity`、`domain`、`embedding`、`kb`、`modality`、`fact-check`、`jailbreak`、`pii`、`preference`、`reask`、`user-feedback`） | 语义、安全、模态与偏好理解 |

**集体收益**：多信号组合比单一信号理解更丰富。

### 2. 投影协调

当路由器把信号协调成可复用的中间事实时，价值更大：

```yaml
projections:
  partitions:
    - name: balance_domain_partition
      semantics: exclusive
      members: [mathematics, coding, creative]
      default: creative
  scores:
    - name: reasoning_pressure
      method: weighted_sum
      inputs:
        - type: complexity
          name: hard
          weight: 0.6
        - type: embedding
          name: math_intent
          weight: 0.4
  mappings:
    - name: reasoning_band
      source: reasoning_pressure
      method: threshold_bands
      outputs:
        - name: balance_reasoning
          gte: 0.5
```

**集体收益**：投影将多条较弱或竞争性的信号转化为可复用的命名路由事实，供多条决策共用。

### 3. 决策融合

信号通过逻辑运算符组合：

```yaml
# 示例：多信号数学路由
decisions:
  - name: advanced_math
    rules:
      operator: "AND"
      conditions:
        - type: "domain"
          name: "mathematics"
        - type: "projection"
          name: "balance_reasoning"
```

**集体收益**：多信号「共同投票」比单一信号更准确。

### 4. 模型专精

不同模型贡献各自强项：

```yaml
modelRefs:
  - model: qwen-math # 数学推理
    weight: 1.0
  - model: deepseek-coder # 代码生成
    weight: 1.0
  - model: claude-creative # 创意写作
    weight: 1.0
```

**集体收益**：系统级智能来自把查询交给合适的专家。

### 5. 插件协作

插件协同增强响应：

```yaml
routing:
  decisions:
    - name: "protected-route"
      plugins:
        - type: "semantic-cache" # 加速
        - type: "response_jailbreak" # 响应筛查
        - type: "system_prompt" # 上下文
        - type: "hallucination" # 质量
```

**集体收益**：多层处理使系统更稳健、更安全。

## 实例

### 用户查询

```
"Prove that the square root of 2 is irrational"
```

### 信号提取

```yaml
signals_detected:
  keyword: ["prove", "square root", "irrational"] # 数学关键词
  embedding: 0.89 # 与数学查询高相似
  domain: "mathematics" # MMLU 分类
  fact_check: true # 证明需要核验
```

### 投影协调

```yaml
projection_outputs:
  balance_domain_partition: "mathematics"
  balance_reasoning: true
```

### 决策

```yaml
decision_made: "advanced_math"
reason: "数学领域且投影指示推理压力"
confidence: 0.95
```

### 模型选择

```yaml
selected_model: "qwen-math"
reason: "擅长数学证明"
```

### 插件链

```yaml
plugins_applied:
  - semantic-cache: "未命中缓存，继续"
  - response_jailbreak: "持续检查输出是否越界"
  - system_prompt: "追加：请给出严格数学证明"
  - hallucination: "启用事实核验"
```

### 结果

- **准确**：路由到数学专家
- **快**：先查缓存
- **安全**：持续筛查输出风险
- **高质量**：启用幻觉检测

**这就是集体智能**：并非由单一组件拍板，而是信号、投影、规则、模型与插件协作涌现出的决策。

## 集体智能的收益

### 1. 更高准确率

- 多信号降低误报
- 专用模型在各自领域表现更好
- 信号融合覆盖更多边界情况

### 2. 更强鲁棒性

- 某一信号失效时系统仍可工作
- 多层安全形成纵深防御
- 回退机制保障可用性

### 3. 持续学习

- 从全量交互中学习模式
- 反馈信号改进后续路由
- 集体知识随时间增长

### 4. 涌现能力

- 可处理并非为单点设计的场景
- 新模式从信号组合中涌现
- 智能随系统复杂度扩展

## 下一步

- [什么是信号驱动决策？](signal-driven-decisions) — 深入决策引擎
- [配置指南](../installation/configuration) — 搭建自己的集体智能系统
- [信号教程](../tutorials/signal/overview) — 学习配置信号与决策

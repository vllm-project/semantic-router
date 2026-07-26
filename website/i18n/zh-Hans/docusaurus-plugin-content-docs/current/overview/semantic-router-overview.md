---
description: 语义路由器概览，说明多信号模型选择如何在 LLM 系统中提升成本、质量、安全与灵活性。
translation:
  source_commit: "ab2aa160"
  source_file: "docs/overview/semantic-router-overview.md"
  outdated: true
sidebar_position: 2
---

# 什么是语义路由器（Semantic Router）？

**语义路由器（Semantic Router）** 是一层智能路由：根据从请求中提取的多种信号，为每次查询动态选择最合适的语言模型。

## 问题

传统 LLM 部署往往对所有任务使用单一模型：

```text
用户查询 → 单一 LLM → 响应
```

**弊端**：

- 简单查询成本过高
- 专项任务表现不佳
- 缺少安全与合规控制
- 资源利用率差

## 方案

语义路由器采用**信号驱动决策**，智能路由查询：

```text
用户查询 → 信号提取 → 投影协调 → 决策引擎 → 插件 + 模型分发 → 响应
```

**收益**：

- 成本更优（简单任务用小模型）
- 质量更好（强项任务用专用模型）
- 内置安全（越狱检测、PII 过滤等）
- 灵活可扩展（投影 + 插件架构）

## 工作流程

### 1. 信号提取

路由器从每次请求中提取 **16 类维护中的信号族**：

| 信号族分组 | 族 | 示例作用 |
| ---------- | -- | -------- |
| **启发式** | `authz`、`context`、`keyword`、`language`、`structure` | 低成本策略、请求形态与区域门禁 |
| **学习型** | `complexity`、`domain`、`embedding`、`kb`、`modality`、`fact-check`、`jailbreak`、`pii`、`preference`、`reask`、`user-feedback` | 语义、安全与响应质量理解 |

### 2. 投影协调

投影将原始信号匹配协调为可复用的路由事实：

```yaml
routing:
  projections:
    partitions:
      - name: support_intents
        semantics: exclusive
        members: [technical_support, account_management]
        default: technical_support
    scores:
      - name: request_difficulty
        method: weighted_sum
        inputs:
          - type: complexity
            name: hard
            weight: 0.4
    mappings:
      - name: difficulty_band
        source: request_difficulty
        method: threshold_bands
        outputs:
          - name: balance_reasoning
            gte: 0.6
```

### 3. 决策

信号与投影输出通过逻辑规则组合，形成路由决策：

```yaml
decisions:
  - name: math_routing
    rules:
      operator: "AND"
      conditions:
        - type: "domain"
          name: "mathematics"
        - type: "projection"
          name: "balance_reasoning"
    modelRefs:
      - model: qwen-math
        weight: 1.0
```

**含义**：若查询被归类为数学**且**投影层标记为偏重推理，则路由到数学模型。

### 4. 模型选择

根据决策选择最合适的模型：

- **数学查询** → 数学专用模型（如 Qwen-Math）
- **代码查询** → 代码专用模型（如 DeepSeek-Coder）
- **创意查询** → 创意向模型（如 Claude）
- **简单查询** → 轻量模型（如 Llama-3-8B）

### 5. 插件链

在模型执行前后，插件处理请求/响应：

```yaml
routing:
  decisions:
    - name: "guarded-route"
      plugins:
        - type: "semantic-cache" # 先查缓存
        - type: "response_jailbreak" # 响应侧风险筛查
        - type: "system_prompt" # 添加上下文
        - type: "hallucination" # 事实核验
```

## 关键概念

### 模型混合（Mixture of Models, MoM）

与在**单个模型内部**工作的 Mixture of Experts（MoE）不同，MoM 在**系统层面**运作：

| 方面 | Mixture of Experts（MoE） | Mixture of Models（MoM） |
| ---- | ------------------------- | ------------------------ |
| **范围** | 单模型内部 | 跨多个模型 |
| **路由** | 内部门控网络 | 外部语义路由器 |
| **模型** | 共享架构 | 彼此独立 |
| **灵活性** | 训练时固定 | 运行时动态 |
| **场景** | 模型效率 | 系统级智能 |

### 信号驱动决策

传统路由常用简单规则：

```yaml
# 传统：简单关键词
if "math" in query: route_to_math_model()
```

信号驱动路由组合多种信号：

```yaml
# 信号驱动：多信号组合
if (has_math_keywords AND is_math_domain) OR has_high_math_embedding: route_to_math_model()
```

**收益**：

- 路由更准确
- 更好处理边界情况
- 能适应上下文
- 降低误报

## 实例

**用户查询**：「证明根号 2 是无理数」

**信号提取**：

- keyword：["prove", "square root", "irrational"] ✓
- embedding：与数学查询相似度 0.89 ✓
- domain："mathematics" ✓

**决策**：路由到 `qwen-math`（数学相关信号一致）

**插件**：semantic-cache 未命中；response_jailbreak 持续监测输出风险；system_prompt 增加「给出严格数学证明」；hallucination 用于核验

**结果**：由专用数学模型给出高质量证明

## 下一步

- [什么是集体智能？](collective-intelligence) — 信号如何形成系统智能
- [什么是信号驱动决策？](signal-driven-decisions) — 深入决策引擎
- [配置指南](../installation/configuration) — 部署语义路由器

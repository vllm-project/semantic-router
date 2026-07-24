---
translation:
  source_commit: "49e7c766"
  source_file: "docs/tutorials/projection/scores.md"
  outdated: false
sidebar_position: 3
---

# 分数（Scores）

## 概览

`routing.projections.scores` 将匹配信号证据合成为**一个连续数值**。

在以下情况使用分数：

- 一条路由依赖多个弱信号而非单一决定性检测器
- 学习型与启发式证据应贡献同一路由结果
- 希望数值聚合留在决策层之外

## 主要优势

- 将多个弱信号聚合成单一连续数值供路由使用。
- 加权混合逻辑集中在一处，便于审计。
- 支持二值、置信度与原始数值三种值源。
- 负权重可在信号匹配时主动拉低分数（例如明显简单请求）。

## 解决什么问题？

决策适合可读布尔逻辑，不适合表达「从上下文长度取一点、从推理标记取一点、对极简单请求减权，再判断属于哪一档」。

分数在信号与决策策略之间提供显式数值层。

在 `balance` 配方中例如：

- `difficulty_score` 混合简洁性、上下文长度、结构、推理标记、嵌入与复杂度等
- `verification_pressure` 混合事实核查需求、引用请求、高风险领域、纠正反馈与上下文长度

这样权重故事集中在一处，而不是散落在多条决策中。

## 运行时行为

当前实现仅支持 `method: weighted_sum`。

每个输入贡献：

`weight * input_value`

`input_value` 取决于 `value_source`：

- 省略或 `binary`：信号匹配用 `match`，未匹配用 `miss`
- `confidence`：使用匹配置信度，未匹配为 `0`
- `raw`：使用 `SignalValues` 中的原始数值（如计数或度量值），缺失时为 `0`

当前默认：

- `match` 默认为 `1.0`
- `miss` 默认为 `0.0`

校验器要求每个输入引用 `routing.signals` 中已声明的信号。

当前支持的输入类型包括：

- `keyword`
- `embedding`
- `domain`
- `fact_check`
- `user_feedback`
- `reask`
- `preference`
- `language`
- `context`
- `structure`
- `complexity`
- `modality`
- `authz`
- `jailbreak`
- `pii`

分数是内部投影状态；决策不直接引用分数名；下一步由映射消费。

## 规范 YAML

```yaml
routing:
  projections:
    scores:
      - name: difficulty_score
        method: weighted_sum
        inputs:
          - type: keyword
            name: simple_request_markers
            weight: -0.28
          - type: context
            name: long_context
            weight: 0.18
          - type: keyword
            name: reasoning_request_markers
            weight: 0.22
            value_source: confidence
          - type: embedding
            name: agentic_workflows
            weight: 0.18
            value_source: confidence
          - type: complexity
            name: general_reasoning:hard
            weight: 0.22
```

### 原始值源

当信号族通过 `SignalValues` 暴露数值度量（计数、距离、token 总数）时，使用 `value_source: raw` 将这些值直接送入加权和，而不是将其简化为二值或置信度标量。

```yaml
routing:
  projections:
    scores:
      - name: workload_pressure
        method: weighted_sum
        inputs:
          - type: structure
            name: many_questions
            weight: 0.2
            value_source: raw
          - type: structure
            name: nested_depth
            weight: 0.4
            value_source: raw
```

不同信号族的原始值可能采用不同量纲。请谨慎选择权重，或使用与预期数值范围相符的阈值区间。

## DSL

```dsl
PROJECTION score difficulty_score {
  method: "weighted_sum"
  inputs: [
    { type: "keyword", name: "simple_request_markers", weight: -0.28 },
    { type: "context", name: "long_context", weight: 0.18 },
    { type: "keyword", name: "reasoning_request_markers", weight: 0.22, value_source: "confidence" },
    { type: "embedding", name: "agentic_workflows", weight: 0.18, value_source: "confidence" },
    { type: "complexity", name: "general_reasoning:hard", weight: 0.22 }
  ]
}
```

## 配置字段

| 字段 | 含义 |
|------|------|
| `name` | 分数标识 |
| `method` | 当前为 `weighted_sum` |
| `inputs[].type` | 读取的信号族；也可以是 `projection`，用于引用先前的分数或映射输出 |
| `inputs[].name` | 已声明的信号名；当 `type: projection` 且 `value_source: score`（默认）时为分数名，当 `value_source: confidence` 时为映射输出名 |
| `inputs[].weight` | 贡献系数；负权重降低分数 |
| `inputs[].value_source` | `binary`、`confidence`、`raw` 或 `score`（用于投影输入）；投影输入使用 `confidence` 时读取映射输出的校准置信度 |
| `inputs[].match` / `inputs[].miss` | 二值模式下的显式取值 |

## 配置

分数位于 `routing.projections.scores`。每个分数需要 `name`、`method`（当前为 `weighted_sum`）以及引用已声明信号的 `inputs` 列表。完整说明见 [规范 YAML](#规范-yaml) 与 [配置字段](#配置字段)。

## 何时使用

在以下情况使用分数：

- 多个弱指标应合成为单一难度或升级信号
- 同一加权故事要在多条路由间复用
- 希望在一处集中调节路由敏感度

## 何时不用

在以下情况不要使用分数：

- 单一原始信号已能干净决定路由
- 规则用普通布尔逻辑即可保持可读
- 需要立即可见的决策输出名——分数仍需映射

## 设计说明

- 保持分数名稳定，因为 `routing.projections.mappings[*].source` 依赖它们。
- 记录每个权重的理由，尤其在混合置信型学习信号与启发式信号时。
- 数值聚合优先用分数，让 `routing.decisions` 专注可读布尔组合。
- 当匹配信号应主动降低档位时（如 `balance` 对明显简单请求），使用负权重。

## 层级组合

分数可以通过 `type: projection` 引用先前的投影分数或映射输出置信度。这样便可构建由一个分数叠加另一个分数的分层路由结构。

### 分数引用分数

使用 `value_source: score`（或省略 `value_source`）读取先前计算的分数值：

```yaml
routing:
  projections:
    scores:
      - name: difficulty_score
        method: weighted_sum
        inputs:
          - type: keyword
            name: reasoning_request_markers
            weight: 0.6
            value_source: confidence

      - name: verification_pressure
        method: weighted_sum
        inputs:
          - type: projection
            name: difficulty_score
            value_source: score
            weight: 0.8
          - type: fact_check
            name: needs_fact_check
            weight: 0.4

    mappings:
      - name: verification_band
        source: verification_pressure
        method: threshold_bands
        outputs:
          - name: needs_deep_verify
            gte: 0.7
          - name: standard_verify
            lt: 0.7
```

### 引用置信度

使用 `value_source: confidence` 读取映射输出区间的校准置信度：

```yaml
- type: projection
  name: needs_deep_verify
  value_source: confidence
  weight: 0.5
```

### 依赖顺序

分数可以按任意顺序声明。运行时会按拓扑顺序计算，确保始终先解析依赖项，再计算依赖它们的分数。配置校验会拒绝循环依赖。

### 投影输入配置字段

| 字段 | 含义 |
|------|------|
| `type` | `projection` |
| `name` | 已声明的分数名（用于 `value_source: score`）或映射输出名（用于 `value_source: confidence`） |
| `value_source` | `score`（读取原始分数值）或 `confidence`（读取映射输出置信度） |
| `weight` | 贡献系数 |

## 下一步

- 阅读[映射](./mappings)，将分数转为决策可用的命名档位。
- 阅读[概览](./overview)，了解完整投影工作流及与信号、决策的关系。

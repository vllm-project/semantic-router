---
translation:
  source_commit: "45bfd49e"
  source_file: "docs/tutorials/projection/scores.md"
  outdated: true
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

`keyword`、`embedding`、`domain`、`fact_check`、`user_feedback`、`preference`、`language`、`context`、`structure`、`complexity`、`modality`、`authz`、`jailbreak`、`pii`

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
| `inputs[].type` | 读取的信号族 |
| `inputs[].name` | 已声明的信号名 |
| `inputs[].weight` | 贡献系数；负权重降低分数 |
| `inputs[].value_source` | `binary`、`confidence` 或 `raw` 行为 |
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

## 下一步

- 阅读 [Mappings](./mappings)，将分数转为决策可用的命名档位。
- 阅读 [Overview](./overview) 了解完整投影工作流及与信号、决策的关系。

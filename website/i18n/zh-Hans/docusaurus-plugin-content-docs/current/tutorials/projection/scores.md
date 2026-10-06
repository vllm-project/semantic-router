---
sidebar_position: 3
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/tutorials/projection/scores.md"
  outdated: false
---

# 分数

## 概览

`routing.projections.scores` 将已匹配的信号证据组合成一个连续数值。

## 解决什么问题？

决策适合可读的布尔逻辑。它们不适合表达“从上下文长度取一点证据，从推理标记再取一些，对非常简单的请求减去一些权重，然后再决定属于哪一档”。

分数通过在信号和决策策略之间提供一层显式数值来解决该问题。

## 分数在运行时如何表现 {#how-scores-behave-at-runtime}

只支持 `method: weighted_sum`。

每个输入贡献：

`weight * input_value`

`input_value` 的计算取决于 `value_source`：

- 省略或 `binary`：信号匹配时使用 `match`，未匹配时使用 `miss`
- `confidence`：使用已匹配信号的置信度，信号未匹配时为 `0`
- `raw`：使用来自 `SignalValues` 的原始数值（例如计数或测量值），缺失时为 `0`

默认值：

- `match` 默认为 `1.0`
- `miss` 默认为 `0.0`

大多数输入引用 `routing.signals` 下已声明的信号。`kb_metric` 和 `projection` 类型改为引用如下所述的派生运行时状态。

支持的输入类型包括：

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
- `kb`
- `conversation`
- `event`
- `kb_metric`
- `projection`

对于 `kb_metric`，`kb` 标识已配置的知识库，`metric` 选择 `best_score`、`best_matched_score` 或该知识库声明的指标，`value_source` 为 `score`。对于 `projection`，`name` 标识更早的分数或映射输出。

分数是内部投影状态。决策不直接引用分数名；接下来由映射消费它们。

## 配置

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

### 原始值来源 {#raw-value-source}

当某个信号家族通过 `SignalValues` 暴露数值测量（计数、距离、token 总量）时，使用 `value_source: raw` 把它们直接送入加权和，而不是先降成二元或置信度标量。

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

原始值在不同信号家族间可能尺度不同。请谨慎选择权重，或使用考虑预期数值范围的阈值档。

## 配置字段 {#config-fields}

| 字段 | 含义 |
|-------|---------|
| `name` | 分数标识符 |
| `method` | 当前为 `weighted_sum` |
| `inputs[].type` | 支持的信号家族、`kb_metric` 或 `projection` |
| `inputs[].name` | 已声明的信号名，或 `projection` 的更早分数/映射输出 |
| `inputs[].kb` / `inputs[].metric` | `kb_metric` 的知识库名和数值指标 |
| `inputs[].weight` | 贡献乘数；负权重会降低分数 |
| `inputs[].value_source` | `binary`、`confidence`、`raw`，或投影输入的 `score`；投影输入上的 `confidence` 读取映射输出的校准置信度 |
| `inputs[].match` / `inputs[].miss` | 二元模式的显式值 |

## 何时使用

在以下情况使用分数：

- 若干弱指标应组合成一条难度或升级信号
- 同一加权故事应由多条路由复用
- 希望在一处集中调优路由灵敏度

## 何时不使用 {#when-not-to-use}

不要在以下情况使用分数：

- 一条原始信号已经干净地决定路由
- 规则可以保持为普通布尔逻辑
- 需要立即得到决策可见的输出名；分数仍需要映射

## 分层组合 {#hierarchical-composition}

分数可以用 `type: projection` 引用更早的投影分数或映射输出置信度。这支持一层分数建立在另一层之上的分层路由构造。

### 分数到分数引用 {#score-to-score-reference}

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

### 置信度引用 {#confidence-reference}

使用 `value_source: confidence` 读取映射输出档的校准置信度：

```yaml
- type: projection
  name: needs_deep_verify
  value_source: confidence
  weight: 0.5
```

### 依赖顺序 {#dependency-ordering}

分数可以按任意顺序声明。运行时按拓扑顺序评估它们，以便依赖总是在被依赖者之前解析。环会在配置校验时被拒绝。

分数不做额外模型调用，也不会自行持久化内容；它们组合请求上已有的信号结果。权重不会自动校准异构输入，因此请在带标签的流量上评估完整分数和映射。完整示例见
[`config/recipes/balance/config.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/recipes/balance/config.yaml)。

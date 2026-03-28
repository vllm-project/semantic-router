---
translation:
  source_commit: "45bfd49e"
  source_file: "docs/tutorials/projection/mappings.md"
  outdated: false
sidebar_position: 4
---

# 映射（Mappings）

## 概览

`routing.projections.mappings` 将投影分数转为决策可消费的**命名路由档位**。

在以下情况使用映射：

- 决策应读取语义标签（如 `balance_reasoning`、`verification_required`），而非原始阈值
- 一个分数应通过可复用命名输出供给多条决策
- 路由策略审查应在命名档位上进行，而非内联数值比较

## 主要优势

- 将数值分数转为决策可读的策略名。
- 阈值策略集中在一处，避免在路由间重复。
- 一个分数可通过可复用命名输出服务多条决策。
- 可选通过 `sigmoid_distance` 做置信度校准。

## 解决什么问题？

分数作为内部信号有用，但决策规则不应依赖每个人记住「0.82 表示推理档」或「0.35 表示需核验」。

映射将数值阈值转为可复用策略名。

带来两点收益：

- 一个分数可供给多条决策
- 阈值策略集中在一处，而非在路由间重复

这也是投影对决策**可见**的环节。当前实现中，决策只能引用 `mapping.outputs[*].name`，不能引用分数名或分区名。

## 运行时行为

当前实现仅支持 `method: threshold_bands`。

每个输出用以下边界声明一条或多条：

- `lt`
- `lte`
- `gt`
- `gte`

重要运行时细节：

- 按顺序检查输出
- **首个**匹配的输出胜出
- 若无输出匹配，映射不产生任何结果
- 可选 `calibration` 为匹配的投影输出计算置信度

当前支持的校准方法为 `sigmoid_distance`，根据分数与最近阈值边界的距离推导置信度。

## 规范 YAML

```yaml
routing:
  projections:
    mappings:
      - name: difficulty_band
        source: difficulty_score
        method: threshold_bands
        calibration:
          method: sigmoid_distance
          slope: 10.0
        outputs:
          - name: balance_simple
            lt: 0.18
          - name: balance_medium
            gte: 0.18
            lt: 0.48
          - name: balance_complex
            gte: 0.48
            lt: 0.82
          - name: balance_reasoning
            gte: 0.82

  decisions:
    - name: reasoning_math
      priority: 250
      rules:
        operator: AND
        conditions:
          - type: domain
            name: math
          - type: projection
            name: balance_reasoning
```

## DSL

```dsl
PROJECTION mapping difficulty_band {
  source: "difficulty_score"
  method: "threshold_bands"
  calibration: { method: "sigmoid_distance", slope: 10 }
  outputs: [
    { name: "balance_simple", lt: 0.18 },
    { name: "balance_medium", gte: 0.18, lt: 0.48 },
    { name: "balance_complex", gte: 0.48, lt: 0.82 },
    { name: "balance_reasoning", gte: 0.82 }
  ]
}

ROUTE reasoning_math {
  PRIORITY 250
  WHEN domain("math") AND projection("balance_reasoning")
  MODEL "openai/gpt5.4"
}
```

## 配置字段

| 字段 | 含义 |
|------|------|
| `name` | 映射标识 |
| `source` | 读取的分数名 |
| `method` | 当前为 `threshold_bands` |
| `calibration` | 可选的匹配输出置信度模型 |
| `outputs[].name` | 决策可见的投影名 |
| `outputs[].lt/lte/gt/gte` | 该输出的阈值边界 |

## 控制台

- `Config -> Projections` 以规范配置编辑映射
- `Config -> Decisions` 可用条件类型 `projection` 引用映射输出

## 配置

映射位于 `routing.projections.mappings`。每个映射需要 `name`、源分数 `source`、`method`（当前为 `threshold_bands`）以及带阈值边界的 `outputs` 列表。完整说明见 [规范 YAML](#规范-yaml) 与 [配置字段](#配置字段)。

## 何时使用

在以下情况使用映射：

- 多条路由应共享同一套档位名
- 希望决策规则可读，如 `projection("verification_required")`
- 阈值策略应集中、可审计

## 何时不用

在以下情况不要使用映射：

- 决策应直接引用原始信号
- 分数仅作诊断，不参与路由策略
- 尚未定义映射应读取的分数

## 设计说明

- 输出名应面向策略，使决策读起来像路由意图，而非阈值算术。
- 保持阈值档位单调、有序、易审计。
- 决策用 `type: projection` 消费命名输出，避免在多处重复数值阈值。
- 除非有意依赖顺序行为，否则避免重叠档位；运行时返回**首个**匹配输出。

## 下一步

- 需要为映射构建数值源时，阅读 [Scores](./scores)。
- 完整 `routing.projections` 工作流及与信号、决策的关系见 [Overview](./overview)。

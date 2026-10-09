---
sidebar_position: 4
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/tutorials/projection/mappings.md"
  outdated: false
---

# 映射

## 概览

`routing.projections.mappings` 将投影分数转成决策可以消费的命名路由档位。

## 解决什么问题？

分数是有用的内部信号，但决策规则不应依赖每个人都记得“0.82 表示推理档”或“0.35 表示需要核验”。

映射通过把数值阈值转成可复用的策略名来解决该问题。

这也是投影变成决策可见的节点。决策引用 `mapping.outputs[*].name`，而不是分数名或分区名。

## 映射在运行时如何表现 {#how-mappings-behave-at-runtime}

支持两种映射方法：

- `threshold_bands`（默认，未设置 `method` 时也使用）— 发出**第一个**匹配的输出档。
- `multi_emit` — 发出**每一个**匹配的输出档，因此一个映射可以从同一分数设置若干正交策略标签。至少需要两个输出。

每个输出使用以下一个或多个边界声明：

- `lt`
- `lte`
- `gt`
- `gte`

重要的运行时细节：

- 按声明顺序检查输出
- 使用 `threshold_bands` 时，第一个匹配的输出胜出
- 使用 `multi_emit` 时，发出每个匹配的输出（按声明顺序）
- 若没有输出匹配，映射不发出任何内容
- 可选的 `calibration` 为每个发出的投影输出计算置信度

当前支持的校准方法是 `sigmoid_distance`，它根据分数离最近阈值边界的距离推导置信度。

## 配置

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
    - name: reasoning_deep
      description: Use the reasoning model for the highest difficulty band.
      priority: 250
      rules:
        operator: AND
        conditions:
          - type: domain
            name: math
          - type: projection
            name: balance_reasoning
      modelRefs:
        - model: google/gemini-3.1-pro
```

## DSL {#dsl}

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

ROUTE reasoning_deep {
  PRIORITY 250
  WHEN domain("math") AND projection("balance_reasoning")
  MODEL "google/gemini-3.1-pro"
}
```

## 配置字段 {#config-fields}

| 字段 | 含义 |
|-------|---------|
| `name` | 映射标识符 |
| `source` | 要读取的分数名 |
| `method` | `threshold_bands`（默认）或 `multi_emit` |
| `calibration` | 匹配输出的可选置信度模型 |
| `outputs[].name` | 决策可见的投影名 |
| `outputs[].lt/lte/gt/gte` | 该输出的阈值边界 |

## 控制面板 {#dashboard}

- `Config -> Projections` 以规范配置形式编辑映射
- `Config -> Decisions` 可以用条件类型 `projection` 引用映射输出

## 何时使用

在以下情况使用映射：

- 若干路由应共享同一档位名称
- 希望有可读的决策规则，例如 `projection("verification_required")`
- 阈值策略应集中且可审计

## 何时不使用 {#when-not-to-use}

不要在以下情况使用映射：

- 决策应直接引用原始信号
- 分数只用于诊断，不是路由策略的一部分
- 尚未先定义该映射应读取的分数

映射不做模型或存储调用；它们转换请求上已计算的分数。阈值仍继承其输入信号的不确定性和校准。完整端到端示例见
[`config/recipes/balance/config.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/recipes/balance/config.yaml)。

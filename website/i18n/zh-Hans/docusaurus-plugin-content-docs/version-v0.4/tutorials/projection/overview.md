---
sidebar_position: 1
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/tutorials/projection/overview.md"
  outdated: false
---

# 投影

## 概览

投影位于信号提取和决策匹配之间。它们解析信号间的竞争，将多个信号合成一个分数，并把该分数转成决策可以引用的命名输出。

路由管道是：

1. `routing.signals` 提取事实。
2. `routing.projections` 协调或派生事实。
3. `routing.decisions` 匹配策略并选择候选模型。

## 主要优势

- 在多个决策间复用同一协调或阈值策略。
- 把数值聚合从布尔决策树中分离出来。
- 为回放和调试保留命名、可解释的输出。

## 解决什么问题？

单个信号刻意保持狭窄。真实路由策略常常需要从一组竞争意图中选出一个胜者，或从若干弱指标得到一条可复用的难度分数。没有投影时，该逻辑会在决策间重复，数值阈值也难以审计。

## 何时使用

在以下情况使用投影：

- 领域或嵌入组中只应保留一个成员处于活动状态
- 若干信号应贡献到一条连续分数
- 若干决策应共享同一套命名阈值档

当一条原始信号已清楚表达路由条件，或多个匹配应保持独立可见时，跳过投影。

## 投影类型 {#projection-types}

| 类型 | 目标 | 决策可见？ | 指南 |
|---|---|---|---|
| `partitions` | 从竞争的领域或嵌入信号中保留一个胜者 | 否；决策仍引用获胜的原始信号 | [分区](./partitions) |
| `scores` | 用 `weighted_sum` 组合信号值 | 否；分数供给映射或其他分数 | [分数](./scores) |
| `mappings` | 将分数转成命名阈值输出 | 是，通过 `type: projection` | [映射](./mappings) |
| trace | 在路由回放中解释分区、分数和映射结果 | 仅运维 | [投影 traces](./traces) |

当前方法是：

- 分区语义：`exclusive` 和 `softmax_exclusive`
- 分数方法：`weighted_sum`
- 映射方法：`threshold_bands`（第一个匹配输出）和 `multi_emit`（每个匹配输出；至少需要两个输出）
- 可选映射校准：`sigmoid_distance`

## 配置

```yaml
routing:
  signals:
    embeddings:
      - name: technical-support
        threshold: 0.75
        candidates: [installation help, troubleshooting]
      - name: account-management
        threshold: 0.72
        candidates: [billing issue, subscription change]
    context:
      - name: long-context
        min_tokens: 4K
        max_tokens: 200K

  projections:
    partitions:
      - name: support-intents
        semantics: exclusive
        members: [technical-support, account-management]
        default: technical-support
    scores:
      - name: request-difficulty
        method: weighted_sum
        inputs:
          - type: embedding
            name: technical-support
            value_source: confidence
            weight: 0.5
          - type: context
            name: long-context
            weight: 0.5
    mappings:
      - name: difficulty-band
        source: request-difficulty
        method: threshold_bands
        outputs:
          - name: support-fast
            lt: 0.5
          - name: support-escalated
            gte: 0.5

  decisions:
    - name: escalated-support
      description: Route difficult support requests to the larger model.
      priority: 150
      rules:
        operator: AND
        conditions:
          - type: projection
            name: support-escalated
      modelRefs:
        - model: support-large
```

只有映射输出名用 `type: projection` 引用。决策不直接引用分区名或分数名。

DSL 通过 `PROJECTION partition`、`PROJECTION score` 和 `PROJECTION mapping` 块暴露同样的三个概念。控制面板在 **Config > 投影** 下暴露它们。

## 依赖与限制 {#dependencies-and-limitations}

- 投影不做额外的模型或存储调用；它们消费请求上已计算的信号结果。
- 分区默认值是回退，不是其成员已匹配的证据。
- 加权和不自动校准来自不同信号家族的输入。请在带标签的流量上一起评估权重和映射档。
- 派生分数之间的环会在校验时被拒绝。
- 端到端示例见 [`balance` 配方](https://github.com/vllm-project/semantic-router/blob/main/config/recipes/balance/config.yaml)。

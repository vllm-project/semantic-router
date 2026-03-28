---
translation:
  source_commit: "45bfd49e"
  source_file: "docs/tutorials/projection/overview.md"
  outdated: false
sidebar_position: 1
---

# 投影（Projections）

## 概览

`routing.projections` 位于原始信号检测与最终决策匹配之间的**协调层**。

信号回答「匹配了什么？」决策回答「哪条路由应胜出？」投影填补中间：**如何把若干信号结果协调成可复用的路由事实？**

在需要以下行为时使用：

- 在竞争性的领域或嵌入通道中解析出唯一胜者
- 将多个弱信号合成为一条连续路由分数
- 将阈值策略集中定义一次，并在多条决策中复用命名结果

## 主要优势

- 将信号协调与决策逻辑分离，两层都可读。
- 一条加权或阈值策略可被多条决策复用。
- 提供命名路由档位（如 `balance_simple`、`verification_required`），避免散落数值比较。
- 在竞争的领域或嵌入通道间显式选出胜者。

## 解决什么问题？

信号刻意保持狭窄：关键词、嵌入、领域分类或上下文检测各自给出局部事实。决策刻意是布尔性的：把这些事实组合成路由规则。

中间存在实际缺口：

- 多个领域或嵌入信号可能同时匹配，但路由只需要一个胜者
- 多个弱信号合起来可能表示「请求很难」或「回答需核验」
- 同一套阈值故事要在多条决策中复用，而不能到处复制数值逻辑

没有投影时，这种协调逻辑会被塞进决策、在路由间重复，或混回检测层。`routing.projections` 把协调作为独立一层。

## 与信号、决策的关系

把流水线想成三层：

1. `routing.signals` 从请求中提取可复用事实。
2. `routing.projections` 协调或聚合这些事实。
3. `routing.decisions` 用布尔策略匹配并选择路由。

更具体地：

- `partitions` 协调已有的 `domain` 或 `embedding` 匹配，保留一个胜者
- `scores` 将匹配信号聚合成一个数值
- `mappings` 将该数值变成命名的投影输出
- 决策仍用原生类型引用原始信号，如 `domain`、`embedding`、`keyword`
- 决策仅通过 `type: projection` 引用投影输出

当前实现的两条重要边界：

- 决策不直接引用分区名
- 决策不直接引用分数名

只有 `mapping.outputs[*].name` 会成为决策可见的 `projection(...)` 目标。

## 运行时流程

当前运行时中，投影在信号提取之后、决策评估之前：

1. 基础信号在 `routing.signals` 下运行
2. `routing.projections.partitions` 归约竞争的 `domain` 或 `embedding` 匹配
3. `routing.projections.scores` 根据匹配信号与置信度计算数值
4. `routing.projections.mappings` 产生如 `balance_reasoning`、`verification_required` 等命名输出
5. 决策组合原始信号与上述命名输出

因此分区更「靠近信号」，映射更「靠近决策」。

## 当前约定

仓库在编写与运行时使用统一的投影命名：

- DSL 使用 `PROJECTION partition`、`PROJECTION score`、`PROJECTION mapping`
- 规范运行时配置将同一约定存于 `routing.projections.partitions`、`scores`、`mappings`

当前实现支持：

- `exclusive` 或 `softmax_exclusive` 的分区
- `method: weighted_sum` 的分数
- `method: threshold_bands` 的映射
- 可选的 `method: sigmoid_distance` 映射校准

## 工作流

1. 在 `routing.signals` 下定义可复用检测器。
2. 当一个领域或嵌入族在决策读取前应归一为胜者时，使用 [Partitions](./partitions)。
3. 将匹配信号合成为加权分数时，使用 [Scores](./scores)。
4. 将分数转为命名路由档位时，使用 [Mappings](./mappings)。
5. 在 `routing.decisions[*].rules.conditions[*]` 中用 `type: projection` 引用这些档位。

## Balance 配方示例

维护中的 [`deploy/recipes/balance.yaml`](https://github.com/vllm-project/semantic-router/blob/main/deploy/recipes/balance.yaml) 展示了预期模式：

- `balance_domain_partition` 在维护的路由域中解析唯一领域胜者
- `balance_intent_partition` 在嵌入意图通道中解析唯一胜者
- `difficulty_score` 混合上下文、结构、关键词、嵌入与复杂度等证据
- `difficulty_band` 将分数映射为 `balance_simple`、`balance_medium`、`balance_complex`、`balance_reasoning`
- `verification_pressure` 与 `verification_band` 产生如 `verification_required` 等可复用核验输出
- `premium_legal`、`reasoning_math` 等决策将原始 `domain` 匹配与投影输出组合

该配方重要之处在于：投影在多条路由间复用，而非单一玩具示例。

## 规范形态

```yaml
routing:
  signals:
    embeddings:
      - name: technical_support
        threshold: 0.75
        candidates: ["installation guide", "troubleshooting"]
      - name: account_management
        threshold: 0.72
        candidates: ["billing issue", "subscription change"]
    context:
      - name: long_context
        min_tokens: "4000"
        max_tokens: "200000"

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
          - type: embedding
            name: technical_support
            weight: 0.18
            value_source: confidence
          - type: context
            name: long_context
            weight: 0.18

    mappings:
      - name: request_band
        source: request_difficulty
        method: threshold_bands
        outputs:
          - name: support_fast
            lt: 0.25
          - name: support_escalated
            gte: 0.25
```

```dsl
PROJECTION partition support_intents {
  semantics: "exclusive"
  members: ["technical_support", "account_management"]
  default: "technical_support"
}

PROJECTION score request_difficulty {
  method: "weighted_sum"
  inputs: [
    { type: "embedding", name: "technical_support", weight: 0.18, value_source: "confidence" },
    { type: "context", name: "long_context", weight: 0.18 }
  ]
}

PROJECTION mapping request_band {
  source: "request_difficulty"
  method: "threshold_bands"
  outputs: [
    { name: "support_fast", lt: 0.25 },
    { name: "support_escalated", gte: 0.25 }
  ]
}
```

## 配置

完整投影约定在 `routing.projections` 下，含三个子节：

- `partitions`：将竞争的领域或嵌入匹配协调为单一胜者
- `scores`：将匹配证据聚合为连续数值
- `mappings`：将分数转为决策可引用的命名档位（`type: projection`）

完整 YAML 与 DSL 示例见上文 [规范形态](#规范形态)；字段级说明见各子教程。

## 何时使用

在以下情况使用投影：

- 竞争的领域或嵌入通道应收敛到单一胜者
- 路由难度或核验压力分散在多个弱信号上
- 多条决策应共享同一套 `simple / medium / complex` 等分层逻辑
- 希望阈值策略集中在一处，而不是在路由规则中复制粘贴

## 何时不用

在以下情况可跳过投影：

- 单一原始信号已清晰表达路由条件
- 多个匹配应对决策独立可见
- 决策用普通布尔组合即可保持可读，且不需要共享加权逻辑

## 控制台

控制台直接暴露同一投影约定：

- `Config -> Projections` 以规范配置形式管理分区、分数与映射
- `Config -> Decisions` 可用条件类型 `projection` 引用映射输出
- `DSL -> Visual` 将 `Projection Partitions`、`Projection Scores`、`Projection Mappings` 与信号、路由、模型、插件并列展示为可编辑实体

原始导入/导出仍可将当前路由器 YAML 反编译为仅路由 DSL，并将编辑后的 DSL 再编译回规范 YAML。

## 下一步

- 阅读 [Partitions](./partitions)：独占领域或嵌入胜者选择。
- 阅读 [Scores](./scores)：对匹配信号加权聚合。
- 阅读 [Mappings](./mappings)：命名路由档位与 `type: projection` 的决策引用。

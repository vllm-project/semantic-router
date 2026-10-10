---
sidebar_position: 5
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/tutorials/projection/traces.md"
  outdated: false
---

# 投影追踪与回放

## 概览

当 [路由回放](../../installation/configuration) 采集路由记录时，每条记录除了 `projections`（匹配的输出名）和 `projection_scores`（聚合数值分数）外，还可以包含结构化的 **`projection_trace`** 字段（JSON）。

该 trace 解释该请求上分区归约、加权分数和映射阈值*如何*表现，因此运维人员和 dashboard 用户可以调试路由，而不必只从标量分数推断内部过程。

## 主要优势

- 回放记录保持自描述：同一持久化路径同时携带聚合分数和结构化可解释性 JSON。
- 分区竞争者列表、softmax 胜者、映射边界距离和每个输入的分数贡献出现在一个对象中。
- 载荷中的版本 `1` 为增量字段留出空间，而无需改写较旧的消费者。

## 解决什么问题？

匹配的投影名（`projections`）和数值摘要（`projection_scores`）回答**选了什么**，但它们不保留分区**为何**选出胜者，或映射离下一阈值档有多近。

`projection_trace` 为审计、支持和 insights 视图补上这一缺口，而无需在查询时额外推断。

## 何时使用

- 你运行 **路由回放**（memory、Redis 或 PostgreSQL），并希望每条记录带有可解释性列。
- 你对由回放支撑的流程使用 **dashboard Insights** 下钻，并需要可折叠的投影详情。
- 你正在构建对照真实流量（而不只是静态配置）校验投影行为的工具。

## 配置

评估投影时会发出可解释性载荷；存储取决于回放后端配置：

- 使用 **[Router 回放配置](../../installation/configuration)** 中描述的持久化设置启用回放。
- 对于 PostgreSQL，确保迁移包含 JSONB 列 **`projection_trace`**，以及 **`projections`** 和 **`projection_scores`**。

没有单独的“trace 开/关”开关——只要投影运行且记录器持久化富化后的 `SignalResults`/`Record`，tracing 就是隐式的。

## Schema 版本 `1` {#schema-version-1}

该 trace 已版本化，供前向兼容的消费者使用。

- **`partitions`**：每个在此请求上运行归约的 `routing.projections.partitions` 组一条。记录带 **`raw_score`** 的 **`contenders`**（语义为 `softmax_exclusive` 时还有 **`normalized_score`**）、所选 **`winner`**、**`winner_score`**（路由器随后存在信号上的值）、**`raw_winner_score`**、**`margin`**（比较分数中第一名减第二名——softmax 用归一化权重，否则用原始置信度），以及分区合成其配置默认成员时的 **`default_used`**。
- **`scores`**：每个已配置投影分数一条，带 **`total`** 和每个输入的 **`contribution`**（`weight * value`），与加权和运行时一致。
- **`mappings`**：每个投影映射一条。按顺序对每个阈值档记录 **`matched`**、**`boundary_distance`**（到最近活动阈值的距离），并对第一个匹配档记录该档使用的 **`selected_output`**、sigmoid **`confidence`** 和 **`boundary_distance`**。

## 在哪里查看 {#where-to-inspect}

- **Dashboard → Insights**：打开一条由回放支撑的记录。**Projection trace** 部分显示分区胜者表（存在时带竞争者分解）、分数输入和映射决策（包括边界距离和每个输出的阈值步骤列表），以及可折叠的原始 JSON。
- **存储**：同一对象通过配置的路由回放后端持久化在回放 `Record` 上。PostgreSQL 将其存在 `projection_trace` JSONB 列中。

Traces 只从投影契约和已评估的信号结果派生——没有不透明的旁路模型。

投影 traces 包含规则名、分数、竞争者和所选输出。即使未启用正文采集，也应把它们视为策略和流量元数据，并相应限制回放访问。Trace schema 实现于
[`projectiontrace/trace.go`](https://github.com/vllm-project/semantic-router/blob/main/src/semantic-router/pkg/projectiontrace/trace.go)。

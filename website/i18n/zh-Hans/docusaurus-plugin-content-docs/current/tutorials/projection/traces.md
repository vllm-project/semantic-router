---
translation:
  source_commit: "cefd4ff0"
  source_file: "docs/tutorials/projection/traces.md"
  outdated: false
sidebar_position: 5
---

# 投影追踪与回放

## 概览

当 [Router Replay](../../installation/configuration) 捕获路由记录时，除了 `projections`（匹配的输出名称）和 `projection_scores`（聚合数值分数）之外，每条记录还可以包含结构化的 **`projection_trace`** 字段（JSON）。

该追踪说明 partition reduction、加权分数和 mapping 阈值在该请求中的*具体行为*，让运维人员和 Dashboard 用户无需根据标量分数推测内部过程即可调试路由。

## 主要优势

- Replay 记录保持自描述：同一持久化路径同时承载聚合分数和结构化可解释性 JSON。
- Partition contender 列表、softmax winner、mapping 边界距离以及逐输入分数贡献集中呈现在一个对象中。
- Payload 中的版本 `1` 为增量字段预留了空间，无需重写旧版 consumer。

## 解决什么问题？

匹配的投影名称（`projections`）和数值摘要（`projection_scores`）能够说明**选择了什么**，但不会保留 partition **为何**选中某个 winner，也无法说明 mapping 距离下一个阈值区间有多近。

`projection_trace` 补上了审计、支持和 Insights 视图所需的这部分信息，而且无需额外的查询时推理。

## 何时使用

- 正在运行 **Router Replay**（内存、Redis 或 PostgreSQL），并希望每条记录都包含可解释性列。
- 使用 **Dashboard → Insights** 深入查看由 replay 支持的流程，并需要可折叠的投影详情。
- 正在构建根据真实流量（而非仅静态配置）校验投影行为的工具。

## 配置

计算投影时会产生可解释性 payload；其存储方式取决于 replay 后端配置：

- 使用 **[Router Replay 配置](../../installation/configuration)**中介绍的持久化设置启用 replay。
- 对于 PostgreSQL，确保 migration 在 **`projections`** 和 **`projection_scores`** 旁包含 **`projection_trace`**（JSONB）列。

系统没有独立的“启用/关闭追踪”开关：只要运行投影，并且 recorder 持久化增强后的 `SignalResults`/`Record`，追踪就会隐式启用。

## Schema 版本 `1`

追踪带有版本号，以支持 consumer 向前兼容。

- **`partitions`**：每个在当前请求中执行 reduction 的 `routing.projections.partitions` 分组对应一个条目。记录包含带 **`raw_score`** 的 **`contenders`**（语义为 `softmax_exclusive` 时还包含 **`normalized_score`**）、选中的 **`winner`**、**`winner_score`**（随后由路由器存储到信号中的值）、**`raw_winner_score`**、**`margin`**（比较分数中第一名减第二名；softmax 使用归一化权重，其他情况使用原始置信度），以及当 partition 合成其配置的默认成员时设置的 **`default_used`**。
- **`scores`**：每个已配置投影分数对应一个条目，包含 **`total`** 和逐输入 **`contribution`**（`weight * value`），与运行时的加权和一致。
- **`mappings`**：每个 projection mapping 对应一个条目。对于按顺序排列的每个阈值区间，追踪会记录 **`matched`**、**`boundary_distance`**（到最近有效阈值的距离）；对于第一个匹配区间，还会记录该区间所使用的 **`selected_output`**、sigmoid **`confidence`** 和 **`boundary_distance`**。

## 在哪里查看

- **Dashboard → Insights**：打开由 replay 支持的记录。**Projection trace** 部分会显示 partition winner（存在 contender 时可展开其明细）、分数输入和 mapping decision（包括边界距离和逐输出阈值步骤列表）的表格，以及可折叠的原始 JSON。
- **存储**：同一对象会持久化到 replay `Record` 中（内存、Redis 或 PostgreSQL 的 `projection_trace` JSONB 列）。

追踪仅根据投影契约和已经计算的信号结果派生，不存在不透明的 sidecar 模型。

---
title: 阅读报告和 Dashboard
translation:
  source_commit: "31fa0fdab6787b3b149894db259af13c8ce42f5a"
  source_file: "docs/benchmarking/sr-bench/results.md"
  outdated: true
---

# 阅读报告和 Dashboard

报告给出完整计划分母、正确/已评分/失败数、每个基准的分数和区间、四类互斥 token、主体与裁判/模拟器成本、TTFT、延迟分位数、请求时长之和及实际 wall time。未知为 null；失败和部分运行不能称为完整评测。

对已终止的真实运行，可执行 `vllm-sr benchmark reconcile-usage RUN_ID`，从保留的 SSE 计量追加幂等、带版本的费用更正，零模型调用。报告、比较和后续导出采用更正后的派生计量；原始调用/结果详情及冻结 manifest 保持不变。报告显示更正哈希、时间、核验/变更条数和新旧已知费用。证据缺失或冲突仍为未知。支持单模型与有明确单次调用凭据的直接 MoM；不能用最终响应流推算多次内部调用的费用。

`cache_neutral_cost_usd` 是反事实 token 等价费用：全部输入按冻结的普通输入单价计价，再加输出费用。比较同时显示它相对同一基线的节省率和实测四类 token 费用，用于识别连续运行的缓存预热影响；它不是账单费用，也不是实测无缓存执行。

完整 sr-bench 权重为 MMLU-Pro 10%、SimpleQA 10%、GPQA 15%、HLE 15%、ARC 10%、LiveCodeBench 10%、SciCode 10%、Terminal-Bench 10%、τ³ 10%。九项都完整时才输出总分；子集宏平均仍是子集结果。

基线是在相同完整题集上按相同汇总规则选出的最强已测单模型，不是逐题选优 oracle。精确加权质量相同时，优先选择主体费用完整且最低的单模型，再按固定目标 ID 排序；报告列出全部并列最强模型。任一并列最强模型费用不完整时，节省率保持未知。节省率为 `100 × (1 − 候选主体成本 / 基线主体成本)`，要求完整兼容的计量。小样本用于判断方向；质量持平需要预先确定的非劣界值和保留集置信区间。自托管 token 等价价格不是 GPU 账单节省。

配对质量差值默认展示基于独立题目差值的保守加权 Hoeffding 区间。即使所有配对结果相同或全部答错，区间也不会退化为零。分层 bootstrap 区间保留为诊断值；小样本产生 `[0, 0]` 不能证明能力持平。两种区间都不包含最强基线选择、调优选择或数据污染带来的不确定性。

Dashboard 默认进入 **Runs**，按名称、模型、状态和模式筛选任务，展示完成分母、失败数、持久化更新时间和目标类型。只读轮询会在断网后恢复，并发现 CLI 新建的任务；关闭或刷新页面不会重启任务。

在 **Create evaluation** 中，先选择 **smoke**、**quick** 或 **standard**，再勾选一个或多个已准备 benchmark，或使用 **Select all benchmarks**。它们对应真实运行 profile，standard 使用 holdout split。来源必须具有相同的 profile、seed 和 split。**Review plan** 只从这些冻结来源组合完整 benchmark 题组，不下载数据、不重新抽样，也不调用模型。完整使用一个来源时保留原数据集身份；选择子集或组合多个来源时生成可复用的冻结数据集。存在冲突时明确拒绝，不静默合并。

采样、预算以及请求和任务限制通过表单控件设置，无需编辑 JSON。已注册目标的固定参数覆盖运行默认值，并保持只读。**Route preview** 还可填写可选的会话和对话上下文，以检查依赖会话状态的路由。启动前先审阅冻结计划；审阅计划不生成模型答案。

**Datasets** 支持搜索、按 profile/benchmark 筛选和分页。点击数据集可查看题目、benchmark 覆盖和学科分组。题目每页 25 条，可按 benchmark、学科和文本搜索；打开题目可阅读任务说明与选项，固定来源可用时也能展示代码和 agent 任务的完整输入。参考答案、隐藏测试和工具凭据不会返回。来源信息和哈希默认折叠，点击 **Evaluate dataset** 可复用所选数据。能浏览公开题目不代表题目从未被见过；不要用 standard 题目调优。

各 profile 的总题量是其已准备题集的题数之和。题集可能重叠，因此该数值不是去重题数，也不是所选运行的实际分母。

运行详情分为 **Results**、**Questions**、**Calls**、**Evidence** 和 **Recipe**，先看汇总结果，再按需查看逐题响应、计量和冻结配置。

运行期间，即使没有新题完成，已用时间也会继续更新。正在执行的调用展示当前阶段、已用时间、最近记录的响应活动和接收字节数，帮助区分长响应与停止接收数据。流活动不能证明答案质量，也不是计费 token 数；token 和费用仍需完整的用量回执。
CLI 可通过 `vllm-sr benchmark show RUN_ID --calls --active` 读取相同活动，并用 `--after`、`--limit` 分页。

**Compare iterations** 分两步：先选择存在兼容结果的 live 单模型基线，再勾选任意数量的可比较候选。没有兼容候选的基线不会出现在选项中。搜索缩小候选范围，**Select all** 会跨页选择当前搜索匹配的可用运行。更换基线会清空候选；更改选择后，旧比较结果会隐藏，直到再次点击 **Compare runs**。服务端仍会逐一核验成对结果。

候选按创建时间排序，选择保存在 URL 中，不限制为两轮优化。质量/成本图和迭代图配合成对置信区间、节省率、token、延迟和 wall time 展示。结果卡片分页，图表和 CSV、JSON 导出保留全部选中比较。点估计为正但区间跨零时，不能认定已经提升。

MoM 目标可由运维人员在注册信息中设置 `capture_recipe: true`，并固定 `config_hash` 和规范的 `preview_url`。worker 仅在捕获前后生成配置和生效运行时哈希均匹配目标、源配置 ETag 保持不变时，捕获脱敏 recipe；**Frozen recipes** 支持查看、下载及核对采集时间、投影哈希。真实调用独立确认实际配置哈希。下载省略部署连接信息和凭据，是 recipe 产物，不是完整可部署配置。旧运行没有快照时明确显示不可用，不借用后续配置。

题目结果和调用列表每次最多读取 100 条、每页展示 25 条，搜索仅作用于已加载记录，完整调用内容按需读取；汇总指标始终来自完整报告，不受详情条数影响。恢复候选、排除原因和子任务也有分页。

**Run events** 按时间从早到晚显示可读事件，支持类型筛选，每页展示 25 条。打开详情最多读取 1,000 条，后续记录需点击 **Load more events** 显式加载。接口不提供总数，因此满页时只标注已加载数量；筛选也仅覆盖已加载事件。事件是快照，**Refresh evidence** 重新读取快照，运行进度仍独立轮询更新。读取失败会保留游标和已有记录。重评分和训练矩阵导出复用已保存证据，不产生模型调用。

## 显式恢复未完成任务

worker 停止或响应丢失后，先核对持久化派发与调用记录。在已终止任务中使用 **Review recovery plan**：

- **Continue undispatched cases** 只允许继续从未派发模型调用的单元格。
- **Retry known failed cases as new attempts** 需要明确选择单元格，并确认新尝试及额外费用。调用状态或费用不明确、已有完整答案、已被其他恢复任务领取的单元格均排除。

恢复创建独立子任务，保留原任务。子任务分母、进度和成本只覆盖本次选择的范围；原任务已知费用单独显示在 lineage 中。子任务完成不等于原 benchmark 已全量完成。Dashboard 在当前标签页保存待确认的完整恢复请求，响应丢失时复用同一幂等键；不会自动重试模型生成或重启 worker。

```bash
vllm-sr benchmark recover-plan RUN_ID --mode undispatched --output recovery.json
# 核对可继续/排除的单元格；可用 selected_cells 指定已审核的子集。
vllm-sr benchmark recover RUN_ID --plan recovery.json --idempotency-key recovery-1
```

若使用 `--mode failed`，最后一步还需 `--acknowledge-new-attempt`。对账不确定的提交时，必须复用相同计划、单元格和幂等键。

---
title: 从 Smoke 到冻结验收
translation:
  source_commit: "31fa0fdab6787b3b149894db259af13c8ce42f5a"
  source_file: "docs/benchmarking/sr-bench/iterate.md"
  outdated: true
---

# 从 Smoke 到冻结验收

1. 从 smoke preview 开始，再做有界 live smoke，验证最终答案、评分器、身份、计量、取消和持久化证据。
2. 创建 experiment，在 quick/dev 上保存一次单模型矩阵和当前 MoM 结果，后续兼容迭代复用这份基线。
3. 用 `config validate`、`config plan`、`config apply` 做一次明确的配置调整，确认预期哈希已激活；需重启时使用 `serve --replace-active-config`。将注册 MoM 目标的 `config_hash` 更新为已验证的实际哈希，已有运行保留其冻结定义。
4. 从保存的基线生成候选计划，保留相同题目、采样、评分选项和限制，仅选择已注册的 MoM 目标。先 preview，再检查服务端给出的 Replay 可用组合；不符合条件的路径需要 live 评测。
5. 在同一 dev 集上实测有希望的候选并成对比较。冻结最终策略后，再执行预先约定的 standard/holdout live 验收，不根据验收失败调参。

```bash
vllm-sr benchmark experiment create "Routing quality and cost" --idempotency-key study-1
vllm-sr benchmark experiment attach EXPERIMENT_ID --run BASELINE_ID --role baseline
vllm-sr benchmark candidate-plan BASELINE_ID --target balance --mode preview \
  --experiment EXPERIMENT_ID > candidate-review.json
# 审阅计划后，提取冻结 manifest 再提交。
jq '.manifest' candidate-review.json > preview.json
vllm-sr benchmark preview --manifest preview.json --detach
vllm-sr benchmark replay-options --limit 10
vllm-sr benchmark replay-options BASELINE_ID --limit 10
vllm-sr benchmark replay --baseline BASELINE_ID --preview PREVIEW_ID
vllm-sr benchmark comparison-options --limit 10
vllm-sr benchmark comparison-options BASELINE_ID --limit 10
vllm-sr benchmark compare BASELINE_ID CANDIDATE_ID
vllm-sr benchmark regrade RUN_ID --output regrade.json
vllm-sr benchmark export DEV_RUN_ID --output training-matrix.json
```

启用 Learning 时，preview 在当前学习状态的只读快照上执行模型选择；可解析时返回具体模型，并保留选择状态、原因和 `selection_provenance` 中的配置/状态哈希、采集时间、是否使用本地采样及种子。预览不更新学习状态，后续真实请求也可能因状态或采样变化而选择不同模型。不要通过关闭 Learning 来验证它；那会测到另一套策略。Dashboard 在每题预览详情中显示这些依据，并解释尚需真实执行的选择。

配置哈希不会冻结持续变化的 Learning、session 或遥测状态。Harness 不会自动隔离、重置或重放各候选的 live 学习状态；需要记录预期状态条件，并把未受控的状态差异列为比较限制。即使 Learning 关闭，依赖遥测的选型仍可能返回 state-dependent 快照。

单请求可使用 `vllm-sr route preview --request-file request.json`。它支持 Router 的请求子集：role/content/tool-call 消息、tools、函数选择、response format、输出预算、字符串 metadata 和 preview options/context；不接受任意 Chat Completions 参数，`temperature`、`stream` 等不支持字段会被拒绝。可用 `--session-id`、`--conversation-id`、`--sampling-seed` 指定只读预览上下文；seed 不固定后续 live 随机选择。Benchmark 题目用显式 `request_metadata` 传递请求 metadata，包含来源或参考答案的题目 `metadata` 不会被转发。

**Replay** 和 **Compare** 的第一级只展示至少有一个可用下级的基线，第二级只展示兼容预览或实测候选。查询与提交复用同一权威校验，提交时再次校验。两边必须使用相同的冻结题目和请求协议。Replay 还会核验保存的请求输入、确定性选模、评分协议，以及每个所选 cell 恰好一次完整生成。仅题目顺序不同可以复用，并保存明确回执。不会忽略内容哈希或偷偷调用模型。响应丢失后保留原 baseline、preview 和幂等键，按同一意图核对。

Compare 允许比较 completed 或 failed 的 live 运行，但每个计划 cell 都必须有明确终态。失败按错误计入完整计划分母；缺少结果或已完成但未评分的答案会阻止比较。失败状态和未知费用始终保留。这衡量的是冻结限制下实际交付的质量，包含执行失败。

只读 API 为 `GET /api/sr-bench/v1/replay-options` 和 `GET /api/sr-bench/v1/comparison-options`；不传 `baseline_run_id` 查询基线，传入后查询兼容下级。`limit` 为 1–25，`after` 必须使用上一页返回的不透明游标，CLI `benchmark replay-options` 使用相同游标。空页若仍有 `has_more: true`，表示搜索尚未完成，需点击 **Load more** 或传入下一页游标。`scan_limited: true` 表示部分证据超过单页验证限额，不能据此认定不存在其他兼容组合。可见的可比较证据变化时，游标失效并要求从第一页刷新。以上查询不调用模型。

Experiment 持久关联 baseline、initial、preview、estimate、candidate、validation 和 recovery 运行，不改写原始回执。创建或关联实验、查询 Replay 资格和生成候选计划均不产生模型答案；属于同一 experiment 也不代表两次运行一定可比。已结束的完整 live 基线即使失败，也可提供候选计划的冻结协议；这不会重试生成或改写原证据。恢复子任务始终标为 recovery，不替代完整基线。管理员可以继续 CLI 创建的实验；其他可写用户只能在自己的实验中创建新任务。只读用户可以比较有权访问的已有结果。

可在 Dashboard 实验详情或通过 `vllm-sr benchmark experiment delete EXPERIMENT_ID` 删除已结束的实验。删除仅移除分组和关联，保留全部运行、结果和产物；关联任务仍在运行时会阻止删除。重试同一删除会返回已保存的回执，已删除实验的创建键不能再次创建该实验。

Replay 是答案复用产生的诊断估计，不是实测能力、延迟或节省结果；依赖学习状态的快照不允许 replay。离线 regrade 目前只支持选择题/网格最终答案，零模型调用且不改写原始结果。Export 仅允许明确标记的 dev 数据，拒绝 holdout 和未知 split；它不会启动训练。

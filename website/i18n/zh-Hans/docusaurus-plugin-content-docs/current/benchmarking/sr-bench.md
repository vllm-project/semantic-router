---
title: sr-bench 1.0
description: 在可复用的冻结任务上比较 MoM 和单模型的能力、成本、延迟与 token 消耗。
---

# sr-bench 1.0

sr-bench 使用相同的冻结题目比较 MoM 入口与单模型。CLI 和 **Dashboard → Evaluation** 共用一个持久化服务、运行 ID、结果与报告。先用小规模开发集改进路由，再用不相交的保留集验收。

## 选择题量

下表是**每个目标的完整任务数**，不是模型调用次数。代码、Agent、裁判和用户模拟器可能产生多次调用；辅助调用费用单独报告。

| 基准 ID | 能力 | Smoke | Quick/dev | Standard/holdout |
| --- | --- | ---: | ---: | ---: |
| `mmlu-pro` | 14 个学科的知识 | 14 | 500 | 2,000 |
| `gpqa-diamond` | 科学推理 | 4 | 40 | 158 |
| `hle` | HLE 纯文本推理 | 4 | 40 | 200 |
| `livecodebench` | v6 累积编程题 | 2 | 30 | 150 |
| `scicode` | 科学编程完整主问题 | 1 | 3 | 20 |
| `terminal-bench-2.1` | 隔离终端任务 | 1 | 3 | 15 |
| `simpleqa-verified` | 事实准确性 | 5 | 100 | 500 |
| `arc-agi-2` | 公开评测谜题与精确网格输出 | 2 | 12 | 80 |
| `tau3` | τ³ 三个文本交互领域 | 3 | 12 | 60 |
| **总计** | | **36** | **740** | **3,183** |

用 `vllm-sr benchmark catalog` 检查当前安装版本。允许只选择某个能力切片；500 题 MMLU-Pro 开发集不是上游 12,032 题的全量结果。缺少任一基准时，没有完整 sr-bench 分数。

数据准备固定来源版本、内容哈希、任务 ID、种子和分层算法。Smoke 是 quick 的子集，standard 与 quick 不相交。SciCode 子问题保留在同一个任务中。公开题目不能宣称无污染；已看过标签的 GPQA 结果仍需注明复测。

## 使用同一个服务

`vllm-sr serve` 会独立启动核心评测 worker。存储位置为 `<state-root>/.sr-bench/<stack>/store`，主机 API 仅监听回环地址的 `8090 + 端口偏移`。同一工作目录中的 CLI 自动发现该存储和私有服务 token。Dashboard 或配置重载不会中断 worker；`vllm-sr stop` 停止 worker，但保留结果。

核心容器不挂载 Docker socket 或 GPU，也不包含全部上游运行环境。代码和 Agent 任务应使用准备好的独立 worker 主机：

```bash
vllm-sr benchmark setup --benchmark all
vllm-sr benchmark setup --benchmark all --install
```

默认只检查；`--install` 显式安装固定版本的可选解释器和源码，并获取经过 SHA256 校验的 SciCode 测试数据。`--build-sandbox` 构建离线代码评分镜像，回执保存镜像和基础镜像 digest 及固定依赖。这些操作不调用模型。缓存默认为 `~/.cache/vllm-sr/sr-bench-1.0`，可通过 `SR_BENCH_HOME` 覆盖；SciCode 数据默认位于缓存内的 `assets/scicode/test_data.h5`，也可通过 `SR_BENCH_SCICODE_TEST_DATA` 指定已准备文件。终端任务镜像、数据源权限以及裁判/模拟器仍需满足前置条件。

通过 `SR_BENCH_URL` 选择外部 worker 后，不会再创建本地 worker 容器。地址需要从 Dashboard 容器可达；CLI 的主机地址与 Dashboard 的容器地址可以不同，但必须指向同一服务。

```bash
# 在服务和客户端环境中私下配置相同的 SR_BENCH_TOKEN。
vllm-sr benchmark --store ./data/sr-bench serve --host 127.0.0.1 --port 8090
```

非回环监听必须设置服务 token。浏览器通过已认证的 Dashboard 访问代理，不直接访问 worker。`SR_BENCH_TOKEN_ENV` 可以指定自定义服务凭据变量名；模型使用独立的 `api_key_env`。不要把密钥值写入清单或命令参数。

## 准备数据和目标

Parquet 来源需要在准备主机安装 `vllm-sr[bench]`。在 worker 主机或它的共享存储准备数据；本地路径不会自动上传到远端。

```bash
vllm-sr benchmark --store ./data/sr-bench dataset prepare \
  --benchmark mmlu-pro --profile quick > mmlu-quick.json
vllm-sr benchmark --store ./data/sr-bench dataset prepare \
  --benchmark gpqa-diamond --profile quick > gpqa-quick.json
vllm-sr benchmark --store ./data/sr-bench dataset combine \
  mmlu-quick.json gpqa-quick.json > quick-dataset.json
vllm-sr benchmark --store ./data/sr-bench target register --file targets.json
```

受限来源需要相应访问权限。导入本地任务需提供 `--source-path` 和 `--revision`；实际内容仍会被哈希。`--limit` 产生明确标注的子集，不要原地修改冻结文件。

目标字段包括 `id`、`kind: single|mom`、`base_url`、`model` 和可选的 `api_key_env`。计价运行需按实际模型身份提供四类 USD/百万 token 价格：`input`、`cached_input`、`cache_write`、`output`。MoM 还需固定实际配置 `config_hash`，预览使用 `preview_url`，计价时声明 `max_inference_calls`。当前直接 MoM 适配器要求完整的一次推理计量；不能用最终模型的价格代替未计量的组合调用。

运维可用目标的 `request_params` 固定原生生成参数；它们覆盖运行的 `sampling` 默认值，包括温度、推理选项以及显式设置的输出长度。对比前检查实际生效参数。运行的输出上限必须容纳目标固定的 `max_tokens`；降低上限不会改写模型 profile。需要其他原生参数时，应选择另一个由运维注册的 profile。

HLE/SimpleQA 需固定裁判和 `grader_version: sr-bench-reference-judge-v1`；τ³ 需固定模拟器和 `release: 1.0.1`。运维在 store 的 `benchmark-options.json` 配置这些依赖。外部适配器默认发现 `benchmark setup` 安装的固定环境；可用 `SR_BENCH_{LCB,SCICODE,TERMINAL,TAU3}_PYTHON` 和对应 `_ROOT` 覆盖其位置。源码和沙箱镜像仍须固定版本。预检会在付费派发前报告缺失依赖。

## 冻结并运行

清单使用 `version: sr-bench-1.0`，引用数据的 `path`/`sha256`，匹配其 profile/seed，并固定目标、采样和限额。完整示例见 [英文配置说明](https://vllm-sr.ai/docs/benchmarking/sr-bench#plan-and-run)。

```bash
vllm-sr benchmark plan --manifest candidate.yaml --output frozen.json
vllm-sr benchmark run --manifest frozen.json --detach --idempotency-key loop-1
vllm-sr benchmark runs
vllm-sr benchmark show RUN_ID --calls
vllm-sr benchmark report RUN_ID --output report.json
vllm-sr benchmark cancel RUN_ID
```

重复提交相同幂等键绑定相同计划，不会重发模型请求。`Ctrl-C` 仅停止 CLI 等待；停止实际工作需使用 `cancel`。异常、部分输出和派发记录全部保留，不会自动重试生成或重启停止的 worker。

默认 `cost_policy: require_priced` 要求完整价格；显式选择 `capability_only` 可以在价格未知时测试能力，但不能证明节省。绝对超时、空闲超时、输出/重复保护以及任务和运行时限约束工作。费用预留和实际费用停止不是供应商强制执行的通用硬美元上限；缺失计量必须显示未知，不能当作零。

## 两阶段迭代

1. 在相同 dev 题目上保存单模型和当前 MoM 的真实结果，检查错误、路由、成本和延迟。
2. 用 `config validate`、`config plan`、`config apply` 做一次明确的配置调整，确认预期配置哈希已激活；需重启时使用受支持的 `serve --replace-active-config`。
3. 以新哈希冻结清单，用 `benchmark preview` 检查真实 query 的 decision/model。预览没有能力分数。
4. 直接静态路由可用保存的单模型答案做 replay。插件、Agent、组合执行或缺失答案不隐式调用模型。
5. 在相同 dev 集上做候选 live 评测和成对比较。正式结论使用未用于调优的 standard 保留集。

```bash
vllm-sr benchmark preview --manifest preview.json --detach
vllm-sr benchmark replay --baseline BASELINE_ID --preview PREVIEW_ID
vllm-sr benchmark compare BASELINE_ID CANDIDATE_ID
vllm-sr benchmark regrade RUN_ID --output regrade.json
vllm-sr benchmark export DEV_RUN_ID --output training-matrix.json
```

启用 Learning 时，preview 在当前学习状态的只读快照上执行模型选择；可解析时返回具体模型，并保留选择状态、原因和 `selection_provenance` 中的配置/状态哈希、采集时间、是否使用本地采样及种子。预览不更新学习状态，后续真实请求也可能因状态或采样变化而选择不同模型。不要通过关闭 Learning 来验证它；那会测到另一套策略。Dashboard 在每题预览详情中显示这些依据，并解释尚需真实执行的选择。

Replay 是答案复用产生的诊断估计，不是实测能力、延迟或节省结果；依赖学习状态的快照不允许 replay。离线 regrade 目前只支持选择题/网格最终答案，零模型调用且不改写原始结果。Export 仅允许明确标记的 dev 数据，拒绝 holdout 和未知 split；它不会启动训练。

## 阅读报告和 Dashboard

报告给出完整计划分母、正确/已评分/失败数、每个基准的分数和区间、四类互斥 token、主体与裁判/模拟器成本、TTFT、延迟分位数、请求时长之和及实际 wall time。未知为 null；失败和部分运行不能称为完整评测。

对已终止的真实运行，可执行 `vllm-sr benchmark reconcile-usage RUN_ID`，从保留的 SSE 计量追加幂等、带版本的费用更正，零模型调用。报告、比较和后续导出采用更正后的派生计量；原始调用/结果详情及冻结 manifest 保持不变。报告显示更正哈希、时间、核验/变更条数和新旧已知费用。证据缺失或冲突仍为未知。支持单模型与有明确单次调用凭据的直接 MoM；不能用最终响应流推算多次内部调用的费用。

`cache_neutral_cost_usd` 是反事实 token 等价费用：全部输入按冻结的普通输入单价计价，再加输出费用。比较同时显示它相对同一基线的节省率和实测四类 token 费用，用于识别连续运行的缓存预热影响；它不是账单费用，也不是实测无缓存执行。

完整 sr-bench 权重为 MMLU-Pro 10%、SimpleQA 10%、GPQA 15%、HLE 15%、ARC 10%、LiveCodeBench 10%、SciCode 10%、Terminal-Bench 10%、τ³ 10%。九项都完整时才输出总分；子集宏平均仍是子集结果。

基线是在相同完整题集上按相同汇总规则选出的最强已测单模型，不是逐题选优 oracle。精确加权质量相同时，优先选择主体费用完整且最低的单模型，再按固定目标 ID 排序；报告列出全部并列最强模型。任一并列最强模型费用不完整时，节省率保持未知。节省率为 `100 × (1 − 候选主体成本 / 基线主体成本)`，要求完整兼容的计量。小样本用于判断方向；质量持平需要预先确定的非劣界值和保留集置信区间。自托管 token 等价价格不是 GPU 账单节省。

配对质量差值默认展示基于独立题目差值的保守加权 Hoeffding 区间。即使所有配对结果相同或全部答错，区间也不会退化为零。分层 bootstrap 区间保留为诊断值；小样本产生 `[0, 0]` 不能证明能力持平。两种区间都不包含最强基线选择、调优选择或数据污染带来的不确定性。

Dashboard 默认进入 **Runs**，按名称、模型、状态和模式筛选任务，展示完成分母、失败数、持久化更新时间和目标类型。只读轮询会在断网后恢复，并发现 CLI 新建的任务；关闭或刷新页面不会重启任务。

在 **Create evaluation** 中，先选择 **smoke**、**quick** 或 **standard**，再勾选一个或多个已准备 benchmark，或使用 **Select all benchmarks**。它们对应真实运行 profile，standard 使用 holdout split。来源必须具有相同的 profile、seed 和 split。**Review plan** 只从这些冻结来源组合完整 benchmark 题组，不下载数据、不重新抽样，也不调用模型。完整使用一个来源时保留原数据集身份；选择子集或组合多个来源时生成可复用的冻结数据集。存在冲突时明确拒绝，不静默合并。

**Datasets** 支持搜索、按 profile/benchmark 筛选和分页。点击数据集可查看题目、benchmark 覆盖和学科分组。题目每页 25 条，可按 benchmark、学科和文本搜索；打开题目可阅读任务说明与选项，固定来源可用时也能展示代码和 agent 任务的完整输入。参考答案、隐藏测试和工具凭据不会返回。来源信息和哈希默认折叠，点击 **Evaluate dataset** 可复用所选数据。能浏览公开题目不代表题目从未被见过；不要用 standard 题目调优。

运行详情分为 **Results**、**Questions**、**Calls**、**Evidence** 和 **Recipe**，先看汇总结果，再按需查看逐题响应、计量和冻结配置。

**Compare iterations** 先选择单模型基线，再用复选框选择任意数量的已完成候选运行，不限制为两轮优化。候选按创建时间排序，选择保存在 URL 中。质量/成本图和迭代图配合成对置信区间、节省率、token、延迟和 wall time 展示，并支持 CSV、JSON 比较导出。范围不兼容或未完成的运行不能生成对比结果；点估计为正但区间跨零时，不能认定已经提升。

MoM 目标可由运维人员在注册信息中设置 `capture_recipe: true`，并固定 `config_hash` 和规范的 `preview_url`。worker 仅在捕获前后生成配置和生效运行时哈希均匹配目标、源配置 ETag 保持不变时，捕获脱敏 recipe；**Frozen recipes** 支持查看、下载及核对采集时间、投影哈希。真实调用独立确认实际配置哈希。下载省略部署连接信息和凭据，是 recipe 产物，不是完整可部署配置。旧运行没有快照时明确显示不可用，不借用后续配置。

题目结果和调用列表按 100 条分页，完整调用内容按需读取；汇总指标始终来自完整报告，不受已加载详情条数影响。事件默认展示最近 100 条，可继续读取更早记录。重评分和训练矩阵导出复用已保存证据，不产生模型调用。

### 显式恢复未完成任务

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

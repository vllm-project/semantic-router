---
title: 评测平面
translation:
  source_commit: "e56591a9cb24f073bf159927e87116ba6d278741"
  source_file: "docs/benchmarking/evaluation-plane.md"
  outdated: false
---

# 评测平面 {#evaluation-plane}

Evaluation Plane 测量冻结的路由配方、逻辑模型池、binding、工作负载和运行时环境是否改进系统结果。它不会把评测简化为“Router 是否点名了期望模型？”。有用的决策还必须可行、执行预期臂、保持安全和隐私、落在成本和延迟预算内，并改进最终任务或轨迹结果。

实现是一个全新的控制面子系统，具有一份当前运行契约、一个执行器注册表、一种持久包布局、一种报告形状，以及一个服务器证明修订。CLI、控制面板、比较和 Campaign 工作流消费同一证据模型。

控制面板是可选界面，不是执行依赖。Agent 可以通过 Router 校验并应用 YAML、预览路由、经 Envoy 发送真实路由请求，并完全从 CLI 或 HTTP 运行版本化工作负载。该直接路径见 [Agent 评测循环](agent-evaluation-loop)。

实时评测受主体约束。目录为每个请求可达的 Mixture-of-Models 配方发布一个目标，从不是通用运行时目标。运行在发送任何请求前，会冻结该 Mixture 的入口别名、配方和决策边界、逻辑模型臂、provider 回退、支持模型、价格，以及配方/pool/binding/topology 摘要。

## 系统边界 {#system-boundary}

评测观察完整路径，同时保留运行时所有权：

```text
workload + session/tool/media state + policy constraints
                            |
                            v
              signals -> projections -> decision
                            |
                            v
       logical action: model(s), budget, selector/looper, fallback
                            |
                            v
       serving execution: endpoint, queue, retry, physical replica
                            |
                            v
 quality + success + cost + latency + safety + privacy + preference
```

- Router 拥有逻辑入口解析、配方执行、逻辑模型选择、selector/looper 策略、生成预算和逻辑回退约束。
- Agent 和产品控制面拥有工具、角色、工作区变更和外部副作用。
- Envoy 以及服务/集群层拥有传输、物理放置、副本、队列和容量。
- Evaluation Plane 关联这些事实，而不会把其所有权移入 Router 请求路径。

此所有权描述中的“Fleet”是概念性的。已退役的 Fleet 控制面板不属于 Evaluation：`/fleet-sim` 路由、导航、API 表面和启动 sidecar 依赖已被移除。独立的 `src/fleet-sim` 研究包、其文档和发布工具保持独立，不会由控制面板栈启动。

## 八条评测轨道 {#eight-evaluation-tracks}

每条所选轨道有自己的证据级别、覆盖、指标和门槛。运行级证据级别是最弱的所选轨道，因此一条强轨道不能提升另一条缺少证据的轨道。

| 轨道 | 问题 | 当前指标表面 |
| -------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Routing | 配方是否选择了有用且合格的逻辑臂？ | coverage、accuracy、abstention、fallback、execution success、selected-arm count 和 entropy、route latency p50/p95 |
| Model pool | 池是否包含有用、可学习且可靠的互补性？ | arm count、best single、pool oracle、oracle gain、unique wins、selection coverage/entropy、quality dominance、quality-cost Pareto dominance、per-arm 和 worst-arm reliability、pairwise failure overlap、all-arm failure rate |
| Routing + pool | 路由实现了多少池上限？ | realized quality、oracle regret、normalized regret、oracle-capture ratio、reliability、每次成功的完整运行时成本、latency p95 |
| Agentic | 路由在轨迹上是否仍然有用？ | terminal success、task score、invalid-tool rate、trajectory length、每条轨迹的隐私暴露、每个成功轨迹的完整运行时成本 |
| Multimodal | 媒体能力、路由、执行、grounding 和隐私是否正确？ | 总体和按模态的 support/quality、privacy violations |
| Preference | 合格反馈是否偏向候选？ | agreement、propensity coverage、effective sample size 和 ratio、self-normalized IPS agreement |
| Safety | 硬策略和阻断决策是否正确？ | violations/case、violated-case rate 和单侧 95% 上界、block accuracy、假阴性和假阳性率 |
| Capacity | 观察到的稳定包络在哪里？ | 按级别和聚合吞吐量、p95/p99、success/error、scaling efficiency、observed saturation、稳定并发上界、每次成功成本，以及冻结 SLO 余量 |

成本归约器在账本不完整时失败关闭。例如，缺失的臂或轨迹成本不会产生人为偏低的每次成功成本数字。选择 Capacity 的实时运行必须在创建前声明版本化 SLO，并请求至少 2 的并发，以便测量相邻级别缩放，而不是隐式接受。清单冻结所需并发、最大 p95 延迟、最大错误率、最小吞吐量和最小相邻级别吞吐量缩放。
Worker 将真实负载观察归约为单调合格包络；服务器重新计算配置文件，并拒绝被改动的级别决策、饱和、余量或裁决。回放容量保持诊断性，不携带 SLO。

## 可运行执行源 {#runnable-execution-sources}

目录将“理解这个问题”与“能为该目标回答它”分开。每个目标为其每个模式声明 `accepted_executors`。规划器只接纳该目标接受的套件模式、轨道和执行器，且每次运行恰好包含一个执行器队列。

| 源 | 当前能力 | 科学边界 |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| `evaluation-smoke` | 通过 `fixture-replay.v1` 在全部八条轨道 schema 上做确定性四案例回放 | 仅 E0 垂直切片诊断 |
| `live-mom-core` | 通过 `mom-cohort-replay.v1` 或 `live-runtime.v1` 使用同一不可变 64 案例隐藏标签队列，带路由、稠密 case-by-frozen-arm 矩阵和路由结果 | 回放为 E0；完整的服务器证明实时运行可以密封 routing E3、model-pool E4 和 joint E5（运行级 E3）。这些级别本身不满足 G3，G3 需要服务器控制的成对 |
| `live-agent-tasks` | 完整 `evaluation-agent-task-ledger.v1` 证据，带由外部生产 Agent 运行时观察的 `evaluation-agent-task-attempt.v1` 重复任务轨迹，包括绑定到精确 Mixture 的评分、隐私、成本和真实工具执行回执 | 服务器校验和归约后的 Agentic E5 任务质量证据；评测 worker 不执行工具，`benchmark_parity_claim` 保持 `none`，此方法没有 Campaign 门槛，也永远不会使 G6 合格 |
| `live-fault-recovery` | 完整的经纪精确步故障账本，带成对基线/处理回执、重复种子、状态、副作用、重试和延迟 | 仅在服务器在至少 5 个种子上重新归约至少 20 对之后为 E5；标为 Continuity 的故障转移仅诊断 |
| `live-multimodal` | 通过活动运行时的有界合格非文本请求 | E0 媒体传输和响应诊断 |
| `live-hard-policy` | Router 拥有的策略/配置证明，加上恰好覆盖所需规则/执行点对的动态攻击/阻断观察 | G2 在显式端点提供一个完整密封实时窗口之前依赖数据 |
| `live-production-experiment` | 消费显式配置的外部账本，包含密封的随机化策略臂分配/暴露、发布控制和可选偏好结果 | G8 使用运维安全回执和风险 UCB；G9 额外要求完整倾向合格的目标/参考结果；vLLM-SR 不创建或运营该实验 |
| `live-capacity` | 通过活动入口的短重复闭环负载级别，用于检查负载执行、遥测和报告生成 | 仅 E0 诊断；它不使 G7 合格，也不支持发布容量决策 |
| 已安装规范化套件，回放 | 精确固定的源导出，规范化为类型化私有产物，并由 `normalized-suite-replay.v1` 回放 | 仅在受信任安装器重新派生时，为按源绑定的每轨道证据 |
| 已安装规范化套件，实时 | 由 `normalized-suite-live.v1` 对活动运行时执行的可见案例 | 实时执行不获得回放资格；它必须赢得服务器拥有的实时证据 |

执行器兼容性是注册表数据，不是 target-ID 分支。目标提供者按模式声明已接受的执行器身份；运行清单为每个套件冻结所选身份，暂存在接纳工作前解析该精确身份。多个执行器实现可以共享一个套件类，而不会更改或模糊已有运行。

轨道可用性同样由每个 Mixture 目标宣传的特性和每轨道要求派生。Routing 需要其冻结配方、拓扑、Router 评测 API 和 Envoy。Model-pool 和 joint 需要 Envoy、拓扑和至少两个可执行冻结臂；它们不依赖 Router 诊断 API。Multimodal 额外需要非文本臂。
Agentic 任务质量和故障恢复使用分开的显式账本，套件不能互相替代。Preference 需要生产实验账本，safety 需要硬策略账本，capacity 需要 Envoy。状态模型有意精确：不支持的套件/目标/执行器组合在规划期间被拒绝；格式错误或缺失的必需产物会使执行或密封失败；已执行但无观察的单元格记录为 `unavailable`；有效报告缺少门槛的类型化证明时，该门槛为 `unavailable`；观察到的合格回归为 `fail`；被所选变更配置文件排除的门槛为 `not_applicable`。目录方法只报告 `configured` 或 `data_required`；资格来自密封运行证据，从不来自目录存在。

### 配置生产证据服务 {#configure-production-evidence-services}

`vllm-sr serve` 从主机环境将 Evaluation 配置转发到控制面板容器。配置只包含规范源、有界超时和环境变量名。凭据值作为继承的容器环境条目传递，从不渲染进 Docker 参数、运行清单、目录响应、报告或日志。

| 控制面板环境变量 | 契约 |
| --- | --- |
| `EVALUATION_ROUTER_API_KEY_ENV` | 专用 Router bearer-token 环境变量名。该 token 必须在 `global.services.management_api.auth.tokens` 下声明，其角色必须包含 `classify.invoke`，且不得是控制面板配方管理凭据。 |
| `EVALUATION_ENVOY_API_KEY_ENV` | 仅用于经纪 Envoy 模型发现和聊天调用的凭据名。 |
| `EVALUATION_AGENT_TASK_LEDGER_URL`、`_API_KEY_ENV`、`_TIMEOUT` | 密封的提供者观察 Agent 任务账本的精确规范源、独立凭据引用和 Go duration。 |
| `EVALUATION_FAULT_RECOVERY_LEDGER_URL`、`_API_KEY_ENV`、`_TIMEOUT` | 密封恢复账本的精确规范源、独立凭据引用和 Go duration。 |
| `EVALUATION_HARD_POLICY_LEDGER_URL`、`_API_KEY_ENV`、`_TIMEOUT` | 密封策略账本的精确规范源、独立凭据引用和 Go duration。 |
| `EVALUATION_PRODUCTION_EXPERIMENT_LEDGER_URL`、`_API_KEY_ENV`、`_TIMEOUT` | 密封分配/暴露/结果账本的精确规范源、独立凭据引用和 Go duration。 |

每个账本要么完全缺失，要么同时配置源和 API 密钥环境引用。配置后超时默认为 `30s`，且最多为 `10m`。Router、Envoy 和账本凭据引用两两不同；每个账本源都不同于 Router、Envoy 和其他账本。源是精确的 `http(s)://host[:port]` 值，不含用户信息、路径、查询、片段、空白或尾部斜杠。无效或部分配置会在 Evaluation 路由可用前失败关闭。

启用 Router bearer 认证时，省略 `EVALUATION_ROUTER_API_KEY_ENV` 仍可通过 Envoy 在满足自身要求的地方保持 model-pool、joint、multimodal 和 capacity 工作可用，但会从目标中移除路由评测。提供专用 token 会恢复 `routing.preview`；Go broker 在服务器端解析其值，并仅向精确冻结的 Router 源添加 `Authorization`。
Python worker 只携带 `SecretRef` 身份，从不解析其环境值或构造授权头。因此，独立的 `vllm-sr benchmark run` 接受未认证目标，但当清单引用凭据而没有控制面板 broker 时会失败关闭。

### 同时寻址基线与候选部署 {#address-baseline-and-candidate-deployments-together}

当一个控制面板 Evaluation 服务必须寻址多个同时运行的 Mixture-of-Models 部署时，设置 `EVALUATION_DEPLOYMENTS_DIR`。规范本地 `vllm-sr serve` 路径仅将此主机目录只读挂载到控制面板容器的 `/app/evaluation-deployments`。Router 和 Envoy 既不接收该挂载，也不接收该环境变量。未设置该变量时，当前单运行时目标及其现有零配置行为不变。

该目录包含严格的 `registry.json` 以及引用的 Router YAML 文件：

```json
{
  "schema_version": "evaluation-deployments.v1",
  "deployments": [
    {
      "id": "baseline",
      "name": "Baseline",
      "description": "Current production deployment",
      "config_file": "baseline/config.yaml",
      "router_origin": "http://baseline-router:8080",
      "envoy_origin": "http://baseline-envoy:8899"
    },
    {
      "id": "candidate",
      "name": "Candidate",
      "description": "Candidate deployment",
      "config_file": "candidate/config.yaml",
      "router_origin": "http://candidate-router:8080",
      "envoy_origin": "http://candidate-envoy:8899"
    }
  ]
}
```

Schema 拒绝未知字段、空部署列表、重复 ID 或结果目标 ID、非规范源、绝对/穿越配置路径，以及注册表、配置或主机挂载路径中的任何符号链接。每个配置通过与默认运行时相同的 Mixture 快照加载器解析。服务器从其精确字节派生配置摘要，并从解析内容派生配方、selector、adaptation、binding、pool 和拓扑身份。`registry.json` 不能包含凭据或账本端点；现有仅全局环境的 SecretRefs 仍是唯一的凭据和类型化账本权威。

目录目标 ID 按部署限定（`<deployment>--<mixture-id>`），而嵌入的 Mixture ID 和配方名保持为共享逻辑实验主体。只有安全的部署名会投影到已认证目录。源、配置路径和 SecretRefs 在冻结运行清单和 broker 中保持私有。这让受控成对绑定一个基线目标和一个候选目标，而不会将其网络地址与配方处理混淆。

## Mixture-of-Models 回放与实时评测 {#mixture-of-models-replay-and-live-evaluation}

Mixture 工作区直接链接到 Evaluation，并带有精确所选入口。浏览器只提交目录 `target_id`。它不能注入源、配方或模型列表。服务器将当前目标解析并冻结到运行和清单中；当配方、pool、binding、拓扑、别名、决策或执行器接纳发生漂移时，启动时校验失败关闭。

在零配置单运行时模式下，目标 ID 仍是 Mixture ID。在部署注册表中，目标 ID 按部署限定，而嵌入的 Mixture ID 和配方名在基线和候选之间保持稳定；不可变摘要区分其修订。服务器派生一份规范、正交的因子图：

| 因子 | 精确可执行身份 |
| --- | --- |
| 配方 | 非分类器信号、决策规则/优先级/插件和路由策略；排除候选、选择器算法、选择器投影和决策本地 adaptation |
| Selector policy | 分类器信号、投影和精确的每决策算法配置 |
| Selector | selector-policy 摘要加上每个仅选择器支持模型的逻辑名、单向 provider-model 身份、provider/config 摘要、可用时声明的运行时修订，以及单向后端拓扑摘要 |
| 自适应 | 精确的每决策在线 adaptation 和保护配置 |
| Binding | 入口/别名/回退加上精确决策和候选迭代模型引用；排除选择器身份 |
| Pool | 排序的候选臂可执行身份、能力、模态、价格、配置摘要和声明的运行时修订 |
| Environment | Router/Envoy 和方法账本源/凭据引用，以及候选池服务拓扑；选择器后端拓扑不在此重复 |

这种分解防止同一变更被标为配方、selector 或在线 adaptation 实验。成对比较强制精确配置文件契约，包括所需的主增量：

| 变更配置文件 | 所需主增量 | 允许的依赖增量 | 冻结因子 |
| --- | --- | --- | --- |
| `schema_adapter` | 源代码修订 | 无 | 配方、selector、adaptation、binding、pool、environment |
| `recipe` | 配方 | 无 | 源代码、selector、adaptation、binding、pool、environment |
| `selector` | selector | 无 | 源代码、配方、adaptation、binding、pool、environment |
| `model_pool` | pool | 候选 binding 和候选池拓扑 | 源代码、配方、selector、adaptation、运行时源、凭据和方法账本 |
| `runtime_capacity` | environment | 无 | 源代码、配方、selector、adaptation、binding、pool |
| `online_adaptation` | adaptation | 无 | 源代码、配方、selector、binding、pool、environment |

`agent_multimodal` 尚没有一个独立的服务器拥有处理摘要。它对单独诊断运行仍然有效，但成对比较会失败关闭，而不是在第二个配置文件名下接受通用配方/binding 漂移。未来的成对契约必须冻结显式轨迹、工具/状态、媒体接纳和模态执行因子，然后才能启用该配置文件。

`live-mom-core` 在回放和实时模式中都有一个不可变修订和一个 64 案例工作负载身份。回放使用 `mom-cohort-replay.v1` 产生确定性冻结目标反事实；实时使用 `live-runtime.v1` 调用目标。两种模式都从同一完整稠密池 oracle 为路由打分。回放保持 E0。带 broker 回执、执行证明和服务器归约的完整实时运行可以密封 routing E3、model-pool E4 和 joint E5，运行级证据为 E3，因为运行报告其最弱所选轨道。仅这些级别不能通过 G3；该门槛额外要求服务器控制的基线/候选成对。回放执行器将每个确定性选择和结果绑定到完整的可见和隐藏评分案例快照，而不仅仅是案例 ID，因此更改的案例内容不能静默复用旧伪结果。

一个队列产生三种互补观察：

1. **配方路由：** `POST /api/v1/routing/preview?trace=true` 评估精确冻结的入口，并记录配方、决策、选择方法、选择状态、所选逻辑臂、回退状态和跟踪摘要。
2. **模型池：** 每个案例直接发送到每个冻结逻辑臂。完整 `case × arm` 矩阵测量每臂质量/可靠性/成本、best single、pool oracle、互补性、unique wins、dominance、Pareto 结构和相关失败。
3. **路由系统：** 每个案例通过冻结入口发送。所选臂、响应质量、由 token 派生的冻结价格、可靠性、延迟、oracle capture 和 regret 测量实现的路由加池系统。

Worker 接收可见提示词，但不接收隐藏标签。其网络沙箱只能为精确清单轨道/案例/尝试请求 `models.list`、`routing.preview`、`arm-chat.completions` 或 `routed-chat.completions`。Go broker 拥有源和凭据，验证每个虚拟别名和配方，将直接调用限制在冻结臂，将路由选择限制在冻结决策边界或显式回退，并为每次普通观察写一份回执。服务器在执行后加入隐藏标签，重新计算响应质量和冻结价格成本，要求稠密池矩阵，并在没有原生路由标签时从完整同队列臂结果派生路由 oracle。缺失矩阵单元格永远不会变成零或伪造的路由失败。

## 基准研究与适配器清单 {#benchmark-research-and-adapter-inventory}

基准注册表从 Intelligent Routing Landscape 固定 13 个描述符。其源清单是 15 个外部检出：13 个代码仓库加上单独的 CodeRouterBench 和 xRouteBench 数据集仓库。研究描述符记录每个基准的原生目标。内置规范化目录单独包含 13 个定义，对应今天能从这些固定安全派生的更小证据表面。

十一个规范化器可执行。RouteJudge/ORBIT 和 RouterEval 描述符仅诊断，因为其固定检出不暴露所需的安全每案例 JSON/CSV 导出；Evaluation Plane 不执行上游代码，也不反序列化其 pickle 产物。

| 基准 | 精确源固定 | 原生设计与重点 | 当前安全规范化 schema | 可执行轨道和严格限制 |
| ------------------------------ | -------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| RouterArena | `fda4c53bcf9a979fd9c6f6bb6b713d6ab08ff43e` | 盲查询到模型预测、任务评分、成本、最优性、扰动鲁棒性、Router 延迟 | `routerarena.predictions-and-robustness.v2` | routing、model-pool、joint；导入/回放保持 E0。解析器验证的完整扰动语料可以对冻结 Mixture 重新实时执行，并仅使 `declared-shift.server-live.v1` routing G4 合格 |
| RouteJudge / ORBIT | `494810de2605f69737e72b55baf6e60c95c6dec0` | 预算条件推荐，随后是匿名偏好投票 | `routejudge-orbit.unavailable.v1` | 仅诊断；该固定既没有 RouteJudge 暴露/投票记录，也没有安全每案例导出 |
| CodeRouterBench | `e43839edb0d5d0a9feec2f7078019406ab4d64bd` 加上数据集 `e567d89bdd569c9c74ffc7c7118e50d15e46b886` | 有序编码流、稠密臂结果、已验证历史 adaptation、Agentic OOD | `coderouterbench.id-results.v1` | routing、model-pool、joint；仅 ID 任务身份和稠密结果，排除沙箱 OOD Agent 流 |
| LLMRouterBench | `c77cb0506949d8f959e97967d2fefca0e8ff1b05` | 大型稠密查询按模型矩阵、预算收益、oracle gap、Pareto 距离 | `llmrouterbench.result-documents.v1` | model-pool；仅对齐结果文档，没有已学习 Router 决策或原生前沿归约器 |
| RouterEval | `bf94b49cc9f8b37181715a7309e1b70ff5308942` | 从小型到研究规模池的模型池缩放、相对引用和熵 | `routereval.unavailable.v1` | 仅诊断；每案例池证据在该固定中仅通过 pickle 加载 |
| RouterBench | `cc67d1008bd8f3cf1e8040cc3ba4034d31b93c0c` | 稠密结果、模型/级联/过度生成动作、Zero Router 凸包和 AIQ | `routerbench.wide-csv.v1` | model-pool；仅转换后的宽矩阵，不是级联、过度生成、AIQ 或 hull 对等 |
| xRouteBench / LLMRouter | `da3430baaea672743c3957457b0c76faba19876e` 加上数据集 `ea4b6e1b29d9a734f55f0a637baf326bad6aa681` | 单轮、会话、个性化、先字幕多模态、多智能体场景 | `xroutebench.standardized-csv.v1` | model-pool；一次一个标准化场景 CSV，没有推断的会话、偏好、媒体或隐藏调用账本 |
| TwinRouterBench | `7cbb0deac8f697b5faa8489c309560e53d2ef088` | 在 Agent 轨迹前缀/步骤路由，静态层级标签加动态 SWE 执行 | `twinrouterbench.static-summary.v1` | agentic；仅静态前缀摘要，明确不是动态 SWE 沙箱结果 |
| MMR-Bench | `83c8308427a3597213fdba298c098da887b8b01b` | 稠密多模态模型结果和质量规范化成本曲线 | `mmrbench.merged-csv.v1` | model-pool、multimodal；媒体字节被哈希，原生规范化成本不报告为 USD，原生 AUC/能力掩码主张被排除 |
| AceBench | `9a17bc2c7ee3fab9ca023036b82a81898512a001` | 可执行边/云 Agent 任务、任务效用、成本、隐私边界 | `acebench.run-summary.v1` | agentic；仅终端摘要，没有完整工具、出口、副作用或隐私暴露账本 |
| continuity-bench | `5b7e7f82027c5b983435057ddc4d7115b7e9a97b` | 确定性故障转移协议、上下文保留和延迟开销 | `continuitybench.labeled-failover.v3` | agentic 诊断；标注失败观察不是真实超时、429/5xx、retry-after、部分流、网络或 provider 故障，不能决定 G6 |
| FusionFactory / LLMFusionBench | `ef62645a48b9e2167201047da047854415e2bc89` | 选择模型子集、协作拓扑、推理思考和综合 | `fusionfactory.aligned-csv.v1` | model-pool；仅对齐的基础/推理结果，没有图拓扑、综合或隐藏评判调用 |
| R2-Router / R2-Bench | `b0b2291aeee08feb4bedbd199ab014ec60d0004f` | 在质量曲线上的联合模型和输出预算动作 | `r2bench.model-budget-csv.v1` | model-pool；固定 15 预算安全长格式，不加载预测器，也不主张部署曲线/容量对等 |

这些限制是适配器契约的一部分。规范化字段仅在被解析为类型化记录并由当前归约器消费时才合格。研究元数据中列出的论文指标不是可执行证据。

### 基准集对 vLLM-SR 的启示 {#what-the-benchmark-set-teaches-vllm-sr}

- RouterArena 使盲预测文件、隐藏评分、扰动对、价格快照和 Router 开销显式。
- RouteJudge 将偏好收集、暴露、分配和倾向与正确性和安全分开。
- CodeRouterBench 表明自适应路由必须冻结流顺序、反馈时机、记忆状态和无未来泄漏规则。
- LLMRouterBench 和 RouterEval 说明为什么池大小、池构造、划分和种子必须是析因处理轴，而不是隐藏常量。
- RouterBench 要求在将价值归于查询感知路由之前，先有 best-single、无信息前沿和 oracle 基线。
- xRouteBench 表明会话、个性化、模态和隐藏调用需要显式状态和成本记录；先字幕路由不是原生多模态执行。
- TwinRouterBench 将决策点做成轨迹前缀，并说明为什么静态步骤标签和动态任务完成必须保持分开。
- MMR-Bench 要求媒体谱系、模态能力掩码和模态专用质量/成本切片。
- AceBench 使隐私、出口、工具和副作用成为轨迹级硬约束。
- continuity-bench 将可用性与连续性分开，并推动真实故障注入、重复种子和状态转移证据。
- FusionFactory 将动作从单臂扩展到子集/拓扑/综合图，并要求账本中的每个隐藏调用。
- R2-Router 将动作扩展到模型加预算，并要求观察到的输出长度和预算合规，而不仅仅是配置上限。

## 安装固定基准导入 {#installing-pinned-benchmark-imports}

外部检出和原生导出留在被忽略的私有目录中。解析器验证的导入遵循一条可复现路径：

```bash
vllm-sr benchmark benchmarks
vllm-sr benchmark normalizers
vllm-sr benchmark verify-source \
  --adapter <adapter-id> \
  --source-root <ignored-source-root>
vllm-sr benchmark suite-normalize \
  --adapter <adapter-id> \
  --suite-id <suite-id> \
  --source-root <ignored-source-root> \
  --export-root <frozen-native-export> \
  --output <new-normalized-output>
vllm-sr benchmark suite-install \
  --request <new-normalized-output>/request.json \
  --bundle <new-normalized-output>/bundle \
  --source-root <ignored-source-root> \
  --export-root <frozen-native-export> \
  --suite-store <private-suite-store>
```

安装器验证精确干净代码和可选数据集固定，对提供的导出重跑已注册解析器，要求请求和产物字节匹配，校验可见/评分分离和记录覆盖，并密封源、产物集、清单和回放执行器摘要。这证明所提供字节的解析器确定性。它**不**证明上游基准、数据集生成器或原生评分命令产生了这些字节。

因此每个规范化导入都是显式探索性 E0 证据。用户提供的规范化包会做 schema 校验，但不能主张解析器验证。两种导入来源都不能声明资格期望或自行发布门槛。当已注册解析器证明完整、唯一的源/目标扰动语料时，`normalized-suite-live.v1` 可以通过服务器 broker 执行这些精确案例。只有完整成功回执加上服务器证明和独立成对/切片归约，才能随后以 E4 发布已注册的 `declared-shift.server-live.v1` 证据源。该狭窄 G4 陈述关乎精确固定语料上的当前 Mixture；它不主张上游代码已运行，或上游排行榜已被复现。

## 安装第三方 Benchmark Pack {#installing-a-third-party-benchmark-pack}

已经发出 Evaluation Plane 规范化记录的基准不需要源代码插件，也不需要进入内置规范化目录。用此固定布局发布一个仅数据的 Git 仓库：

```text
benchmark.yaml
bundle/
  visible/cases.jsonl
  grading/cases.jsonl
  metadata/licenses.json
  grading/...                 # optional normalized evidence roles
  metadata/media.jsonl        # required for live multimodal cases
```

最小清单描述工作负载，而不是要运行的命令：

```yaml
schema_version: evaluation-benchmark-pack.v1
id: acme-routing-v1
benchmark_id: acme.routing
name: Acme routing benchmark
decision_unit: request
action_space: one model
track_ids: [routing]
split_protocol: fixed hidden test split
arm_ids: [small, large]
data_classification: restricted
redistribution: metadata_only
limitations:
  - Covers the declared routing domain only.
```

从仓库根目录安装，私有套件存储放在该检出之外：

```bash
vllm-sr benchmark benchmark-install \
  --pack <clean-benchmark-pack-checkout> \
  --suite-store <private-suite-store>
```

安装器要求一个干净 Git 检出根，并记录其精确提交。`benchmark.yaml` 和 `bundle/` 下的每个文件必须由该提交跟踪。包有封闭的文件/角色清单：别名、符号链接、未知文件、可执行 runner、任意媒体类型、重复 YAML 键、格式错误的 JSONL、不匹配的可见/评分计数，以及不完整的轨道计划都会被拒绝。安装只复制已校验的内容寻址数据；它从不导入模块、启动 shell，或执行来自 pack 的代码。

每个已安装 pack 都可以作为探索性 E0 证据回放。当数据足够时，pack 也可以对所选冻结 Mixture-of-Models 驱动现有 `normalized-suite-live.v1` 执行器：

- routing 案例需要标识冻结 Mixture 臂（按 arm ID 或模型名）的 `expected_route` 隐藏标签，并产生最高到 E3 的经纪绑定路由诊断；
- model-pool 和 joint 案例需要 `expected_answer` 隐藏标签，并产生精确答案评分、最高到 E4 的完整直接臂矩阵，以及最高到 E5 的路由系统证据；
- multimodal 案例每个案例需要一张内联图像、精确媒体清单和隐藏答案，并产生最高到 E4 的服务器评分证据；
- capacity 案例使用平台拥有的闭环负载协议，仅在服务器校验完整冻结 SLO 证明时才能达到 E5。

Agentic、在线偏好和硬策略主张需要更丰富的托管执行或生产账本。在 pack 中声明这些轨道会保留其规范化回放价值，但不会暴露误导性的通用实时方法。添加真正新的执行语义仍是经过评审的平台变更；添加新数据集则不是。

## 证据级别 {#evidence-levels}

| 级别 | 观察深度 |
| ----- | ----------------------------------------------------------------------------------------------------------------- |
| E0 | 契约、身份、引用、确定性摘要、执行可达性和诊断管道 |
| E1 | 信号区分、缺失/错误行为、延迟、有意义处的校准，以及成对不变性 |
| E2 | 投影纯度、覆盖、重叠、下游相关、边界扰动和贡献完整性 |
| E3 | 期望决策、默认/优先级/碰撞行为、硬策略决策和仅 Router 延迟 |
| E4 | 可行 oracle、实现效用、regret、池使用、基线、种子方差、鲁棒性和 OOD |
| E5 | 最终任务或轨迹结果、实时可靠性、隐私/安全、完整成本、容量和每次成功成本 |

实时记录不能自行声明级别。每个实时执行器拥有不可变注册表，将精确证据源 ID 绑定到允许轨道、类型化载荷、broker 回执基数、所需证明和最大级别。未知源、源/轨道不匹配、缺失类型化事实、缺失或范围错误的回执、不完整密封账本，以及超过源或执行器上限的主张，全部解析为 E0。内置实时注册表只接纳这些源契约：routing 诊断 E3、model-pool 结果 E4、路由 joint 结果 E5、提供者 Agent 任务账本 E5、故障恢复账本 E5、硬策略账本 E4、生产实验账本 E5，以及闭环容量 E5。规范化实时执行器单独通过这些同一服务器拥有的源契约接纳 Benchmark Pack 的 routing、model-pool、joint、multimodal 和 capacity 工作负载。其精确多模态结果和已注册 declared-shift 路由证据可以达到 E4。

证据强度不是分数。E5 可以失败，E0 可以结构完美。带 routing E3 和 model-pool E4 的运行报告两个轨道级别以及运行级 E3。

## G0-G9 门槛语义 {#g0-g9-gate-semantics}

门槛处置为 `required`、`advisory` 或 `not_applicable`。
门槛裁决为 `pass`、`fail`、`unavailable` 或 `not_applicable`。
`unavailable` 表示声明的方法缺少做决定所需的类型化证据；它从不会渲染为通过。

`live-agent-task.v1` 有意不是门槛方法。完整 `evaluation-agent-task-ledger.v1` 必须包含至少 20 个不同任务，每个任务至少两次 `evaluation-agent-task-attempt.v1` 尝试，以及至少一个实际执行的工具。它必须绑定每次尝试、轨迹、评分/隐私/工具回执、精确快照和定价证据、按 token 计价的模型成本，以及完整 Mixture 快照。报告发布尝试成功及其单侧 95% 下界、全部重复的任务可靠性及其界、平均分数和步数、invalid-tool rate、隐私暴露、总成本和每次成功成本。该方法以 `benchmark_parity_claim=none` 赢得 Agentic E5；G6 仍专属于下面的注入故障连续性方法。

| 门槛 | 所需决策证据 | 当前严格决策路径 |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| G0 Reproducibility | 不可变清单/快照、记录、谱系、摘要、失败 | 服务器校验并锚定精确密封包 |
| G1 Static correctness | 严格 schema、引用、计划覆盖、确定性身份 | 服务器校验当前契约和完整计划核算 |
| G2 Hard policy | Router 拥有的策略/配置/运行时快照证明，加上恰好覆盖所需 `(rule, enforcement point)` 对的完整类型化动态决策 | `live-hard-policy` 只读取显式配置的 `hard_policy_ledger`；服务器校验保留响应，并重新归约证明/观察覆盖、阻断准确率和违规。没有该生产窗口时，该方法是 data-required，而不是 implementation-missing |
| G3 Offline value | 服务器控制的 AB/BA 基线/候选实时成对、完整稠密池、路由结果、绝对候选保障和成对非劣 | 单次运行 G3 从不是提升主张。Campaign v2 要求至少 20 个完整案例簇，并联合强制候选规范化 regret 上界 `<= 0.25`、成对 regret-delta 上界 `<= 0.05`、相对最佳固定臂无信息前沿的路由提升 `>= 0.05`、joint reliability `>= 0.80`、all-arm failure `<= 0.20`、质量非劣、worst-arm reliability、每个共享臂的失败非劣，以及仅候选臂可靠性 `>= 0.80` |
| G4 Declared-shift robustness | 已注册解析器、精确固定源和扰动 CAS、完整唯一源/目标对、broker 回执、实时执行证明和服务器拥有的成对/切片归约 | `declared-shift.server-live.v1` 只能从精确冻结语料上的 `normalized-suite-live.v1` 发布 routing E4。导入和回放保持 E0；主张限于声明的关系和切片，并不主张上游 runner 对等、通用 OOD 或污染覆盖 |
| G5 Live fidelity | 未变候选主体、合格实时参考、稍后新鲜证明的实时运行、精确案例队列和完整失败核算 | `evaluation-campaign-fidelity.v2` 报告精确决策/结果保真的单侧 95% 精确二项下界。该界必须 `>= 0.95`；少于 59 个对齐案例为 unavailable，因为即使全匹配队列也无法证明该阈值 |
| G6 Live fault recovery | 真实精确步注入故障、成对基线/处理回执、终端和状态结果、重复副作用、重试/延迟限制、重复种子 | `live-fault-recovery` 只消费已配置的生产故障账本。服务器要求完整密封窗口、至少 20 个独立对和 5 个种子，以及至少 `0.8` 的单侧 95% 恢复下界；Continuity 标注故障转移导出保持 E0 诊断，不证明真实注入故障，并使 G6 保持 unavailable |
| G7 Cost / latency / capacity | 冻结类型化 SLO、至少两个测得负载级别、每级别至少三个独立测量簇、尾延迟、最差簇错误预算、跨簇错误稳定性、最小吞吐量、相邻级别缩放、饱和和余量 | `capacity.slo-envelope.v1` 独立计算每个簇错误率和单侧 95% Wilson 上界，取每个负载级别的最差簇，要求跨簇错误率范围最多 5%，并让服务器在接受 `capacity.slo_headroom >= 0` 前重新证明每个字段 |
| G8 Shadow / canary | 完整生产分配/暴露账本、最小队列、策略臂支持、SRM、风险预算、停止规则和回滚就绪 | `live-production-experiment` 校验完整密封窗口，至少 20 次分配，冻结风险预算不大于 `0.2`。G8 将该预算与单侧 95% 风险率上界比较；即使回滚成功，触发的停止也会使候选失败。受控成对运行保持诊断 |
| G9 Online preference | 完整分配/暴露/结果交叉绑定、显式目标/参考策略、倾向、支持、ESS、分段、SNIPS lift 和 95% 区间 | 同一生产账本仅在完整结果、ESS 至少 `10`、有效样本比至少 `0.5`、每个声明分段至少 5 个观察时才具备 G9 资格。服务器在公共随机化窗口上估计目标和参考 SNIPS；lift 下界必须满足非负冻结最小值。缺失结果为 unavailable；合格回归失败 |

对于 G7，发布包络止于第一个达到或超过所需并发的测得负载级别。更高级别仍作为饱和证据可见，但所需包络之上的预期失败不会推翻已经合格的服务目标。

G4 的已注册实时发布者有意只复用精确固定扰动语料；它不执行不受信任的上游代码。解析器验证的导入仍是有用的 E0 诊断，不能自行提升。G2、G6、G8 和 G9 仅实时，并消费来自显式目标端点契约的服务器经纪账本；默认运行时不宣传这些轨道。Capacity 同样宣传 `capacity.slo-envelope.v1`，但只有冻结的每运行负载协议、SLO、记录和服务器归约才能发布 G7。仅方法注册永远不会发布门槛布尔值。

## 运行生命周期 {#run-lifecycle}

```text
catalog + change profile + suite/target/mode/tracks + budgets
                              |
                              v
                 immutable run manifest
                              |
                              v
          resolve one execution plan and executor
                              |
                              v
         visible cases -> execution -> hidden grading
                              |
                              v
 typed records -> metrics/statistics -> complete G0-G9 set
                              |
                              v
      sealed artifacts + server anchor + public report
```

创建会冻结规范运行 UUID、套件和修订、单个套件-执行器队列、目标快照、变更配置文件、所选轨道、种子、样本/并发预算、策略/binding/pool/工作负载/环境摘要、模型臂、后端拓扑、代码修订、脱敏策略，以及按环境引用的凭据。服务器在待处理运行启动时重新计算清单摘要并重新校验目标执行器接纳。随后报告密封要求每个计划、记录和证明身份匹配该冻结队列。

固定 Python worker 首先将 `report.json` 写为严格的 `worker-report-draft` 线路契约。该不受信任草稿有意省略服务器证明修订、每轨道密封证据级别、受控成对成员、方法归约和路由配方归约。若 worker 试图主张这些字段，控制面板会拒绝它们，从持久记录和服务器拥有的执行证据派生它们，并用密封公开报告原子替换草稿。独立 CLI 报告和比较仍明确为未密封草稿（`WorkerReportDraft` 和 `StandaloneComparison`）；它们对本地诊断有用，但不是控制面板发布证明。

可见案例仅在目标能力资格之后按确定性种子选择。隐藏评分保持私有，并在执行后加入。失败、超时和不可用的计划单元格仍留在覆盖分母中。

### 生命周期治理 {#lifecycle-governance}

每次运行和 Campaign 包都与私有生命周期元数据原子发布，该元数据绑定服务器派生的所有者主体、活动策略修订、留存类、可选证据扣留以及创建审计决策。原始用户身份和电子邮件地址不存储在公开资源或生命周期响应中。Campaign 创建还要求每个绑定运行具有同一所有者，除非管理员执行该操作。启动、取消、删除、扣留、释放和留存更改需要资源所有者或管理员。受保护、被扣留、被基线引用和被 Campaign 引用的证据不能被删除或收集。

本地存储强制有界的每所有者运行/Campaign/字节配额、总物理字节配额和审计字节界。`GET /api/evaluation/v1/lifecycle/usage` 返回确定性的可计费、预留、物理和审计用量；非管理员只看到自己的匿名所有者条目。收集是仅管理员的两步协议：先用 `{"apply":false}` 调用 `POST /api/evaluation/v1/lifecycle/collection`，再用 `{"apply":true,"plan_digest":"sha256:..."}` 重复。Apply 重新计算并绑定精确合格的运行、Campaign、状态、策略、证据引用和字节快照；过期计划会失败且不删除任何内容。过期 Campaign 在同一有序计划中、在其新近解除固定的运行证据之前被移除。

生命周期决策进入不可变、有界、哈希链式活动段，覆盖创建、启动、取消、扣留、释放、留存、删除和垃圾收集，包括被拒绝的授权和保护决策。在配置界处，存储将段头和序列原子密封到密码学检查点，为每个活动运行和 Campaign 保留创建绑定，并移除已压缩事件体。启动校验检查点锚、任何活动后缀以及每个活动资源绑定；全局策略、检查点、链或绑定损坏会失败关闭。此本地契约有意不提供已压缩每事件体的可查询归档。因此内置本地存储不满足长期事件级保留要求；此类部署需要单独设计的外部不可变归档接收器，然后才能启用此生命周期存储。运行本地生命周期损坏会隔离该运行，并阻止需要完整账本的用量、收集或科学决策。活动 `evaluation-lifecycle-policy.v2` / `evaluation-lifecycle-policy.2026-09-01` 契约仅为全新存储：必须移除未发布的 v1 中间状态，且有意没有遗留迁移路径。

## 提升 Campaign {#promotion-campaign}

运行比较保持诊断。`evaluation-campaign.v2` 则从所选变更配置文件的目录槽位组合一个不可变提升决策。没有固定角色包，也没有浏览器拥有的适用性矩阵。

| 槽位 | 绑定 | 精确证据契约 |
| ---- | ------- | ----------------------- |
| G2 | 一次运行 | 实时安全 E3 或更强，结论性服务器拥有硬策略回执 |
| G3 | 受控成对 | `live-runtime.v1`；完整 routing E3、model-pool E4、joint E5；成对回执为 E5 |
| G4 | 一次运行 | `normalized-suite-live.v1`，routing E4，`declared-shift.server-live.v1` |
| G5 | 保真成对 | 合格实时参考加上稍后新鲜实时运行；`normalized-suite-live.v1` 或 `live-runtime.v1` |
| G6 | 一次运行 | `live-fault-recovery` Agentic E5，结论性故障恢复回执 |
| G7 | 一次运行 | 实时容量 E5，结论性冻结 SLO 包络 |
| G8 | 一次运行 | 来自生产分配/暴露控制窗口的实时偏好 E5 |
| G9 | 一次运行 | 来自倾向合格生产结果窗口的实时偏好 E5 |

每个配置文件将每个 G2-G9 槽位声明为 `required`、`advisory` 或 `not_applicable`。Required 绑定不能省略；`not_applicable` 绑定不能提供。每个运行 ID 在 Campaign 中单次使用。每个证据锚绑定槽位、门槛、角色、运行、清单、公开报告、私有回执、可选执行证明和候选主体摘要。所有候选锚必须标识同一精确代码/配置、配方、selector/adaptation/binding/pool 以及模型/支持臂主体，即使每个门槛可能使用不同套件或工作负载。

仅当其两次运行通过服务器拥有的受控成对端点创建时，G3 才是因果的。服务器解析同时可寻址、具有同一逻辑 Mixture ID 和配方名的部署限定目标，要求不同的 Router 和 Envoy 源，解析私有凭据，然后在确定性 AB/BA 块中交错每个共享案例/轨道/尝试/操作坐标。第二次请求仅在第一次结束后开始。成对回执绑定会话、协议、坐标/块摘要、变体清单、顺序、时机和负载坐标。`evaluation-campaign-paired-live.v3` 分别记录基线和候选目标 ID，且每个臂的报告、来源和执行证明必须绑定其自己的精确目标。独立事后运行、共享部署目标或共享 Router/Envoy 源会被拒绝。

G3 归约器按独立案例聚类，并要求至少 20 个完整簇，带双侧 95% 区间。它将无信息前沿派生为完整稠密 case-by-arm 矩阵上的最佳固定候选臂，而不是零，也不是调用者提供的分数。通过要求满足所有冻结边界：候选规范化 regret `<= 0.25`、成对 regret delta `<= 0.05`、相对该固定臂前沿的路由提升 `>= 0.05`、joint reliability `>= 0.80`、all-arm failure `<= 0.20`，以及带 `0.05` 裕度的每轨道质量非劣。全池 worst-arm 统计在每个案例 bootstrap 内重新计算。共享臂必须满足 `0.02` 失败风险裕度和绝对 `0.80` 可靠性下限；仅候选臂必须满足同一绝对下限。仅基线臂会被披露，但不能开脱候选风险。缺失单元格、零质量或全部失败的 oracle、重复坐标和不确定区间不能通过。

G5 单独将未变候选绑定到合格实时参考，以及稍后对精确案例队列的新鲜、已证明实时执行。其公开观察是精确决策/结果保真的单侧 95% Clopper-Pearson 下界，阈值为 `0.95`。少于 59 的队列为 unavailable；59/59 是能证明该阈值的最小全成功队列。回放不是 G5 Campaign 源。对于通用配置文件，G5 绑定 joint E5 证据。`agent_multimodal` 配置文件是显式例外：G3 为 `not_applicable`，G5 绑定来自 `normalized-suite-live.v1` 的 multimodal E4 证据。

门槛所有权保持狭窄：受控成对失败/延迟观察是诊断，永远不能替代生产 G8；G8/G9 需要真正的分配、暴露和结果窗口。Campaign 发布与运行发布/删除共享一个协调器。被引用或依赖基线的运行不能被删除，重启校验会拒绝悬空锚、契约漂移、摘要漂移或损坏的私有证据。

## 统计与报告阅读顺序 {#statistics-and-report-reading-order}

统计契约由主张驱动：

- 案例对齐比较是成对的；聚合点增量是描述性的；
- 已注册成对统计使用独立案例聚类分析单元、显式非劣裕度、至少 20 个单元和双侧 95% 区间；两次重复或相同观察不能使发布合格；
- 比例在支持处携带样本数和区间；
- 聚类会话/轨迹不视为独立轮次；
- 缺失单元格保持可见，从不转换为零质量成功；
- 成本、延迟、质量和安全保留独立轴；
- 运行时成本、评测开销和容量/TCO 保持独立账本；
- 确定性排序和规范浮点归约使报告摘要可复现。

按此顺序阅读报告：

1. 运行和每轨道证据级别；
2. 所需门槛裁决以及精确缺失证据理由；
3. 计划覆盖、失败和不可用单元格；
4. 带样本数和区间的质量、安全、偏好和按切片指标；
5. best single、pool oracle、实现价值、regret、per-arm/worst-arm 可靠性、池可用性和失败重叠；
6. 延迟、可靠性以及全部三个成本账本；
7. 基准、策略、binding、pool、环境、目标、代码和评分器谱系；
8. 失败案例和架构反馈。

E0 报告有意省略提升头条指标。最弱轨道为 E0 的混合运行仍可从其更强密封轨道显示架构发现，但不能发布运行级提升摘要。

## 从评测到配方与池设计 {#from-evaluation-to-recipe-and-pool-design}

评测是架构反馈循环，不是记分板：

| 观察 | 可能原因 | 下一步受控处理 |
| ------------------------------------------------------------ | -------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| 低路由准确率或覆盖 | 信号/投影/决策拓扑、候选资格、选择器校准 | 固定池和环境；检查决策跟踪和切片；更改一层配方 |
| 高路由准确率但最终质量弱 | 期望决策标签是差的效用代理、生成策略漂移或评分器不匹配 | 保持配方固定；重跑任务结果和评分器校准 |
| best single 接近 pool oracle | 很少有用的池互补性 | 移除质量被支配的臂或添加能力缺口臂；重跑稠密矩阵 |
| 高 oracle、低 oracle capture、大规范化 regret | 有价值的池但选择器无法利用 | 在冻结池和工作负载时改进特征/校准/探索 |
| 一臂或 worst-arm 可靠性回归，而 all-arm failure 保持低 | 退化成员被更健康的臂掩盖；仅池可用性不足 | 检查每臂成对区间，修复或移除回归臂，并重跑同一稠密队列 |
| 高成对失败 Jaccard 或高 all-arm failure rate | 相关失败或缺失能力 | 多样化失败域/能力；不要只添加相似臂 |
| Pareto 被支配的臂 | 没有质量收益的运维成本 | 校验硬能力需求，然后剪枝并重跑容量 |
| Agentic 单步质量保持但终端成功下降 | 切换、状态连续性、工具有效性、恢复或延迟信用 | 分别评测决策点和完整轨迹 |
| 模态接纳通过但质量失败 | 后端能力、媒体传输、grounding 或模态评分器 | 隔离接纳、路由、执行和评分阶段 |
| 低倾向覆盖或 ESS | 在线日志/分配支持不匹配 | 在解释偏好前修复暴露策略和支持重叠 |
| 安全假阴性或任何硬违规 | 策略执行或阻断器覆盖 | 立即阻断提升；扩大对抗和切片覆盖 |
| 吞吐量上升但尾延迟/错误越过 SLO | 饱和、排队、重试/回退放大 | 调优服务放置/运行时；保持逻辑配方归因分开 |

### 配方实验设计 {#recipe-experiment-design}

冻结 `workload × policy instance × binding × pool × environment × budget × seed`。先评测结构和可达性，然后是信号/投影质量、决策行为、固定池算法价值，最后是实时端到端结果。每次比较只更改一个处理因子。优先级碰撞、默认分支、语言/领域/模态切片、缺失信号、分布外数据和 expected-invariant/expected-change 对都属于工作负载契约。

### 模型池实验设计 {#model-pool-experiment-design}

在评判 Router 之前构建稠密 case-by-arm 核心。报告 best single、最便宜成功臂、每案例 oracle、边际贡献、unique wins、质量支配、质量-成本 Pareto 支配、成对失败重叠、能力覆盖和池大小扫描。大型研究池在上下文、模态、工具、信任域、可用性、失败域和容量约束被校验之前，不是可部署池。

### 联合实验设计 {#joint-experiment-design}

仅当策略/binding 含义在池之间保持可识别时，才使用析因设计。始终发布与池无关的路由行为、池规范化 regret/oracle capture，以及端到端质量/成本/可靠性。当池、预算、评分器、价格快照或环境同时变化时，永远不要将改进归于选择器。

## 持久证据与信任边界 {#durable-evidence-and-trust-boundary}

控制面板存储是私有的，并根于其配置的 Evaluation 数据目录：

```text
evaluation-root/
  objects/sha256/<digest>
  attestations/<run-id>.json
  campaigns/<campaign-id>/
    campaign.json
    lifecycle.json
  suites/
    objects/{visible,grading,metadata}/sha256/<digest>
    manifests/sha256/<digest>
    index/<suite-id>.json
  runs/<run-id>/
    run-manifest.json
    status.json
    control-events.jsonl
    records.jsonl
    routing-traces.jsonl       # private request-level evidence
    metrics.json
    gates.json
    report.json
    lineage.json
    provenance.json
    failure-summary.json
    capacity-profile.json
    checksums.sha256
    private-checksums.sha256
    report-anchor.json
```

精确每运行集合仅在当前契约将产物标为可选时变化。最终证据不可变且按内容寻址。可变状态使用原子替换。运行创建将状态、清单和初始事件作为一条目录边界发布。Worker 证据导入、规范对象发布、执行证明、报告锚定、删除和重启恢复共享一个发布协调器，因此没有读者会观察到部分决策状态。

信任边界包括：

- 跨评测契约的严格 JSON 解码；worker 和公开报告、方法声明以及类型化证据输入额外拒绝重复对象键；ID、记录数、文件大小和事件流有界；
- 带符号链接拒绝的私有常规文件和目录检查；
- 无法安装 Linux Landlock 或 seccomp 时失败关闭的无网络 worker 沙箱；
- 用于允许方法/源/路径、凭据、正文限制、重定向、模型 ID 和内联媒体的 Go 拥有 broker；
- 绑定到清单、目标、策略、pool、binding、拓扑和时机的精确服务器转录和实时执行证明；
- 对拥有指标/成本/覆盖和当前类型化门槛的独立服务器归约；
- 报告和服务器锚上的 `evaluation-server-attestation.v2`；
- 公开产物允许列表、密钥模式拒绝，以及每个已认证响应上的 `Cache-Control: private, no-store`；
- 带 `Last-Event-ID` 回放和重复抑制的持久数字 SSE ID；
- 有界账本分页、隔离警告，以及持久账本不完整时的决策阻断；
- Campaign 的被引用运行删除保护和重启校验。

Worker 草稿、记录、评分标签、路由跟踪、私有谱系、提示词、输出、目标源、凭据和基础设施标识符永远不是公开产物。

## 控制面板体验契约 {#dashboard-experience-contract}

Evaluation 页面契约定义五个一致工作区：

- **Overview**：能力、目录方法就绪、最新密封证据和不完整账本警告；
- **New experiment**：变更配置文件、套件/目标/模式/轨道/执行器资格、基线锁定、种子/样本/并发预算，以及显式创建/启动意图；从另一个已接受执行器队列选择套件会显式替换先前队列；
- **Runs**：分页持久账本、过滤器、状态、进度、时间线、取消/删除动作，以及区分执行与观察状态的检查器；
- **Reports**：决策边界、轨道覆盖、指标、门槛、成本、失败、来源、架构反馈和安全产物；
- **Compare**：同队列运行对诊断，加上 Promotion Campaign 构建器和密封 Campaign 决策。

每个控件必须在适用处暴露加载、空、权限拒绝、错误、重试、禁用、键盘、焦点和响应式状态。破坏性动作使用共享控制面板对话框处理和类型化确认。URL 状态支持到视图、运行、报告、比较和 Campaign 的直接链接。契约禁止将目录能力、成功 HTTP 请求或缺失指标变成正面就绪状态。

## API 与 CLI {#api-and-cli}

控制面板只暴露当前资源：

- `GET /api/evaluation/v1/catalog`
- `GET|POST /api/evaluation/v1/runs`
- `GET|DELETE /api/evaluation/v1/runs/{id}`
- `POST /api/evaluation/v1/runs/{id}/start|cancel`
- `GET /api/evaluation/v1/runs/{id}/events`
- `GET /api/evaluation/v1/runs/{id}/report`
- `GET /api/evaluation/v1/runs/{id}/artifacts/{artifact-id}`
- `GET|POST /api/evaluation/v1/runs/{id}/lifecycle`
- `GET /api/evaluation/v1/compare?baseline_run_id=...&candidate_run_id=...`
- `POST /api/evaluation/v1/controlled-pairs`
- `GET|DELETE /api/evaluation/v1/controlled-pairs/{id}`
- `POST /api/evaluation/v1/controlled-pairs/{id}/cancel`
- `GET /api/evaluation/v1/lifecycle/usage`
- `POST /api/evaluation/v1/lifecycle/collection`
- `POST /api/evaluation/v1/campaign-readiness`
- `POST /api/evaluation/v1/campaigns`
- `GET|DELETE /api/evaluation/v1/campaigns/{id}`
- `GET /api/evaluation/v1/campaigns/{id}/decision`
- `GET|POST /api/evaluation/v1/campaigns/{id}/lifecycle`

`vllm-sr benchmark` 下的 CLI 表面提供目录、基准/规范化器清单、源验证、内置套件规范化/安装、第三方 `benchmark-install`、套件列表/展示、清单校验、执行、本地 worker 草稿检查、比较和门槛检查。`vllm-sr benchmark --help` 是已安装构建的精确命令参考。

## 规模与扩展接纳 {#scale-and-extension-admission}

当前编排器通过狭窄版本化契约接纳扩展，从而保持精简：

- 内置基准规范化定义声明封闭的原生导出 schema、所需产物、精确受信任解析器、指标映射、限制、源固定和对等测试；
- 第三方 Benchmark Pack 只在一个固定布局中声明元数据和规范化数据；它们复用平台执行器，不能提供可执行钩子；
- 执行器声明支持的模式/轨道，并消费一个类型化 `EvaluationInputs` 边界；
- 目标提供者声明服务器拥有的源、按引用的凭据、入口、每模式已接受执行器、直接臂执行/关联、模型/运行时修订和证据上限；
- 负载提供者声明到达过程、预热、级别、持续时间、重复、资源观察、SLO、饱和和余量；
- 在线提供者声明分配、暴露、倾向、支持重叠、风险预算、停止和回滚账本；
- 归约器声明类型化输入记录、精确指标/门槛所有权以及跨语言黄金测试。

新执行行为仅在其不可变身份、能力契约、受信任派生、有界输入/输出、证据上限和测试内置到平台之后才进入规划。新数据集通过 Benchmark Pack 进入，无需代码注册步骤。不支持的能力在规划期间被拒绝；已接纳运行或 Campaign 仅对无观察的已执行单元格，或有效报告缺少所需类型化证明的门槛使用 `unavailable`。核心不推断观察或门槛裁决。

### 规范目录资源 {#canonical-catalog-resources}

Python 包拥有 `src/vllm-sr/cli/evaluation/golden/` 下的规范指标分析目录和研究基准清单。Go 服务和浏览器保持生成的字节相同镜像，以便在不加载 Python 的情况下校验同一契约。不要直接编辑这些镜像。更改任一规范 JSON 文件后，同步两个镜像家族：

```bash
python3 tools/ci/sync_evaluation_catalogs.py
```

`make dashboard-check` 运行对应的 `--check` 模式，并拒绝缺失或过期镜像。

## 提升规则 {#promotion-rule}

Evaluation 报告建议，它不部署。提升要求预期的每轨道证据级别、所有所需门槛通过、完整覆盖、精确基线、未变比较因子、已评审校验和与谱系，以及显式发布/回滚所有者。容量和生产主张需要固定的实时环境。在线主张额外要求生产分配/暴露证据。不能赢得服务器拥有证据级别的 E0 fixture、回放和实时诊断源对回归和诊断有用，但永远不是提升证据。

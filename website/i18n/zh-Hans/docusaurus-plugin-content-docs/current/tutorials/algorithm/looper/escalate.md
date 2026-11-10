---
is_mtpe: false
translation:
  source_commit: "72b3d17c5fdcaa506e603b1697fdb2d4b6318a6e"
  source_file: "docs/tutorials/algorithm/looper/escalate.md"
  outdated: false
---

# Escalate

## 概览 {#overview}

`escalate` 按顺序逐个调用决策的候选模型，返回第一个被接受的回答。它可以先用小模型或低成本模型作答，只有当第一个回答不够好时才为更强的模型付费。

它在两种 API 上都是执行器（executor）：

- **Chat**：用一项接受检查评判每个回答：按其 token 置信度、按决策模型的 `answer_check` 问题，或按 `refusal` 问题。
- **System One**：每个阶段回答原始的类型化请求；在尝试下一阶段之前，原生接受规则会检查每个答案，完整响应原样返回。

## 主要优势 {#key-advantages}

- 从小到大或从便宜到昂贵逐级升级，而不是固定一个胜出者。
- 让停止条件在每个决策上显式、可审查。
- 用执行器的 `budget` 约束最坏情况：所有尝试共享一个截止时间和调用上限。
- 保持 System One 答案完整：不会把原生答案转成 Chat 文本，也不会为缺失的答案编造置信度。

## 它解决什么问题？ {#what-problem-does-it-solve}

总是调用最强的模型，会在简单请求上花掉全部成本。Escalate 先尝试较便宜的候选，只有当其回答未通过显式检查时才继续，因此只在真正需要的请求上为更强的模型付费。该策略写在路由配置中，而不是应用代码里。

## 何时使用 {#when-to-use}

- 路由有大小或成本不同的候选，且较便宜的回答通常已经足够。
- 你能说清楚什么样的回答可以接受：token 置信度、回答质量检查、拒答检查，或在 System One 上的原生答案概率。
- 路由可以接受第一个回答被拒后第二次调用带来的额外延迟。

## 配置 {#configuration}

Escalate 的设置位于 `escalate` 块中。与每个执行器一样，它还读取与 `type` 并列的 `budget` 和 `on_error`。

### 在 Chat 上 {#on-chat}

```yaml
recipes:
  - name: default
    decisions:
      - name: verified-answers
        priority: 100
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: qwen3-8b
          - model: qwen3-32b
        algorithm:
          type: escalate
          budget: {deadline: 60s, max_calls: 2}
          on_error: last_attempt
          escalate:
            order: size
            accept:
              confidence:
                method: hybrid
                gte: 0.72
                hybrid_weights: {logprob_weight: 0.65, margin_weight: 0.35}
```

`order` 设置升级顺序：`size` 先尝试 `providers.models[].param_size` 最小的模型，`cost` 先尝试定价最低的模型；省略时保持 `modelRefs` 的顺序。

`accept` 只能指定一项检查：

| 检查 | 在以下情况接受回答 |
| --- | --- |
| `confidence: {method, gte}` | 回答的置信度不低于 `gte` |
| `systemone: {gte}` | 决策模型的 `answer_check` 问题认为回答可接受的概率不低于 `gte` |
| `refusal: false` | 决策模型的 `refusal` 问题没有把回答标记为拒答 |

置信度方法读取候选响应中的 token 对数概率，因此后端必须返回它们：

| 方法 | 置信度 |
| --- | --- |
| `avg_logprob`（默认） | 平均 token 对数概率，从 `[-3, 0]` 映射到 `[0, 1]` |
| `margin` | 前两个 token 对数概率之间的平均差值，映射到 `[0, 1]` |
| `hybrid` | `logprob_weight × avg_logprob + margin_weight × margin`；两个权重之和必须为 `1` |
| `self_verify` | 同一模型在一次额外调用中为自己的回答打分 |

`gte` 为必填项，取值在 `[0, 1]` 内。`token_filter: tool_call_args` 只对生成的工具调用参数打分，而不是每个 token；`all`（默认）对每个 token 打分。

`systemone: {gte, deployment, timeout_ms}` 就请求和候选回答，向决策模型（或指定的 System One 部署）提出 `answer_check` 预设问题。`refusal: false` 提出 `refusal` 预设问题；`refusal: true` 会被拒绝，因为它只会接受拒答。两种检查对每个被评判的回答各进行一次决策模型调用；该调用计入执行器的 `deadline`，但不计入 `max_calls`。

`on_error` 决定没有回答被接受时的处理方式：

- `last_attempt`（默认）返回最后一个完成检查的回答。
- `fail` 使请求失败。

模型调用失败，或回答的检查无法运行（例如置信度方法需要的对数概率没有返回）时，总是继续尝试下一个候选。如果没有任何候选返回经过评判的回答，请求失败。

`budget` 约束整次运行：`max_calls` 限制候选调用（以及 `self_verify` 的额外调用）次数，且低于 Router 的单请求上限；`deadline` 限制它们的总耗时。耗尽预算的运行会失败。

#### 尝试可观测性 {#attempt-observability}

Escalate 执行会创建一个 `looper.execute` trace span，并为每次分派的候选或校验调用创建一个 `looper.attempt` 子 span。Router Replay 在 `route_diagnostics.looper` 下保存相应的、有界且不含内容的尝试记录，包括状态、阈值结果、token 用量、延迟和最终尝试序号。该结构从不包含提示词、响应、推理内容、端点 URL、凭据或原始错误。

每次升级都会把请求发送给另一个候选模型。请确保每个候选都符合该路由的数据策略，并为最坏情况下的调用链设定延迟和成本上限。

### 在 System One 上 {#on-system-one}

以 `api: systemone` 发布的配方使用 `escalate.stages` 运行 escalate。本指南以 **Decision 2.0 Kai 0.6B → Vega 27B** 为例：Kai 回答原始请求；escalate 先检查该回答，再决定是否调用 Vega。它不会额外强制进行一次 Kai 分类调用，也不会在第一个回答之后重新运行配方。它支持 `choice`、`score` 和 `noul` 请求，并保持原始问题集合完整。

#### 连接模型 {#connect-the-models}

模型别名是 `providers.models` 下的共享资源。本地别名引用模型运行时部署；远程别名引用具体的 Engine 或兼容的 System One 端点。每个别名只选择一种传输方式。

```yaml
providers:
  models:
    - name: kai
      api_format: systemone
      deployment: local-kai
    - name: vega
      api_format: systemone
      provider_model_id: vllm-sr/Decision-2.0-Vega-27B
      backend_refs:
        - provider: systemone-compatible
          endpoint: http://engine.example.com:8900/v1

global:
  model_catalog:
    deployments:
      local-kai:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Kai-0.6B
        device: auto
```

`provider_model_id` 是发送给远程端点的模型名。如果后端需要认证，在其 backend ref 上设置 `auth: {api_key_env: <VARIABLE>}`。凭据只留在该后端上。同一远程模型的副本放在其 `backend_refs` 中；不同的模型是不同的别名和阶段。

#### 发布 auto 入口 {#publish-an-auto-entrypoint}

按 API 划分的入口将配方以 `vllm-sr/auto` 的名称提供。监听器必须显式允许这个原生名称。Chat 与 System One 的别名相互独立：发布此入口不会改变 Chat 的默认配方。

```yaml
listeners:
  - name: inference
    address: 127.0.0.1
    port: 8801
    systemone:
      models: [vllm-sr/auto]

entrypoints:
  - api: systemone
    model_names: [vllm-sr/auto]
    recipe: native-escalate
```

在把监听器暴露到受信任的开发环境之外前，请先为其添加客户端认证。若还要允许某个具体模型，把它的原生提供商别名加入 `systemone.models`。

#### 定义阶段 {#define-the-stages}

下面的片段接受 `choice`、`score` 和 `noul` 请求。只有当每个答案都通过其类型对应的门限时，Kai 才会提前结束。下面的阈值是 Decision 2.0 Kai → Vega 的一个冻结试点工作点，不是通用的推荐默认值，也不是准确率保证。[复现指南](https://github.com/vllm-project/semantic-router/tree/main/tools/calibration/systemone_auto)固定了该实验的模型和公开基准。请在独立的校准数据上为自己的工作负载选择阈值，并在评估前冻结。

```yaml
recipes:
  - name: default
  - name: native-escalate
    decisions:
      - name: answer
        rules: {}
        modelRefs: [{model: kai}, {model: vega}]
        algorithm:
          type: escalate
          budget: {deadline: 10s, max_calls: 2}
          escalate:
            quality:
              type: uncalibrated
              accept:
                native:
                  - {question_type: choice, field: top_probability, gte: 0}
                  - {question_type: score, field: top_probability, gte: 0}
                  - {question_type: noul, field: top_probability, gte: 0}
            stages:
              - name: fast
                model: kai
                accept:
                  native:
                    - {question_type: choice, field: top_probability, gte: 0.6059704079536342}
                    - {question_type: score, field: top_probability, gte: 0.6059704079536342}
                    - {question_type: noul, field: top_probability, gte: 0.6059704079536342}
              - {name: strong, model: vega}
entrypoints:
  - model_names: [vllm-sr/auto]
    recipe: default
    default: true
```

通用的 `gte: 0` 规则要求每个答案都有有效的分布；它们并不声称有准确率下限。当 Kai 未通过更严格的提前退出门限时，Vega 仍可以返回完整有效的答案。此示例没有依赖模型的路由信号，因此其两次调用的预算覆盖 Kai 以及（需要时的）Vega。如果一次传输重试消耗了一次调用，后续阶段可用的调用就会减少。

阶段默认运行原生模型，除非它写明 `kind: judge`；未设置 `name` 时以其模型命名。`modelRefs` 是完整的候选名单。每个阶段指定其中一个成员，并可以设置自己的超时。阶段名不能使用 `abstain`，它保留给评审（judge）的"不选择"结果。阶段不能添加决策未声明的模型。

对于原生请求，Router 在公共请求处理程序开始时（读取请求体之前）启动一个截止时间和物理调用账本。请求上限取自配方中各原生决策最大的 `algorithm.budget.deadline` 和 `algorithm.budget.max_calls`。选中某个决策后，Router 把这两个限制收紧到该决策配置的预算。这样既保留了现有的按决策配置，又无需新增配方级的预算字段。

决策信号和内置模型信号在请求截止时间和调用账本之内运行。它们各自的超时仍是上限。每一次物理交换都会计入，包括远程决策模型调用和提供商传输重试。信号遥测记录信号交换；每个算法阶段的调用遥测包括其重试和评审调用。如果信号耗尽了请求上限，其余信号保持未知，执行停止。如果所选决策更紧的预算已经耗尽，其算法不会启动。这两种情况都会返回带 `reason: max_calls` 或 `reason: deadline` 的 `systemone_budget_exhausted`；`model_calls` 报告到目前为止计入的所有物理交换。调用账本看不到不透明外部提供商内部的工作。

Router 在接受响应之前会校验每个必需的答案。缺失的答案、无效的值、不支持的问题类型以及未经证明的全输入覆盖都不会成为成功结果。如果在预算耗尽之前没有任何声明的阶段满足规则，请求为未解决（unresolved）：System One 上的 escalate 总是失败而不会返回被拒绝的答案，因此其 `on_error` 只能是 `fail`。没有 `quality` 时，阶段接受任何通过其自身规则的完整答案。

原生预算并不取代完整请求的 HTTP 服务器、代理和客户端超时。特别是，Dashboard 的运维诊断使用管理 API 两分钟的响应写入上限；对于更长的请求，请相应地配置完整的服务路径。

#### 试用路由 {#try-the-route}

把模型、监听器、入口和配方片段合并到一个 `version: v0.5` 的配置文件中，连接 Vega Engine 端点，然后启动 Router：

```bash
vllm-sr serve --config config.yaml
```

然后向其原生端点发送请求。问题名称和判据会原样到达每个被尝试的模型。如果启用了认证，请使用监听器的 API 密钥。

```bash
curl -sS http://localhost:8801/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "vllm-sr/auto",
    "state": "I cannot sign into my account.",
    "questions": {
      "task": {
        "type": "choice",
        "require_full_input": true,
        "instructions": "Which team should handle this request?",
        "criteria": {
          "account": "Account access or authentication",
          "billing": "Payments, invoices or refunds"
        }
      }
    }
  }'
```

响应的 `routing` 字段报告所选模型、阶段以及算法的物理模型调用次数（包括重试）。带有 `systemone_unresolved` 的 `503` 表示没有完整答案通过所配置的规则；它并不表示被省略的答案为假。响应字段和错误约定见 [Router API（英文）](https://vllm-sr.ai/docs/next/api/router#route-a-system-one-request)。

在 Dashboard 中，打开 **System One → Decision Playground**，然后在 **Automatic routes** 下选择 `vllm-sr/auto`。加载配方支持的示例或输入自己的问题，然后选择 **Run**。Dashboard 使用你已登录的管理权限，因此无需粘贴公开 API 密钥。**Decision Monitoring** 显示阶段结果、请求速率和延迟。这些描述的是实时执行；要衡量准确率，请另行评估带标签的请求。

#### 添加路由信号 {#add-routing-signals}

原生信号检查完整的任务文档，包括其状态和问题。请求不提供 Chat 对话或受信任的身份信封。因此，引用 `authz`、`metadata`、`conversation`、`reask`、`user_feedback` 和 `input_modality`（包括通过投影引用）会导致配置校验失败。原生配方中也不能使用 KB 信号。

偏好信号必须使用原生决策任务部署。在它们共享请求生命周期之前，显式对比式或外部偏好适配器以及 MCP 领域分类器在原生配方中不可用。受支持的依赖模型的信号在共享的请求截止时间和调用账本之内保留各自的传输超时。

配方的 `strategy` 仍在匹配的决策之间进行选择。原生执行使用其显式阶段和预算，而不是 Chat 回退、候选要求、决策可靠性、输出约定或自适应控制。请为该原生配方关闭 Chat 回退。`modelRefs` 只声明别名；选择由阶段顺序决定，因此 Chat 推理控制和模型权重会被拒绝。提供商级别的重试和超时设置仍适用于远程交换。

一个配方可以包含多个决策，每个都有自己的阶段和预算。例如，一个关键词决策可以把提到申诉的请求用单个阶段直接发给 Vega，而默认决策尝试 Kai → Vega。只有被选中的决策会执行，其预算在选中后从共享的请求上限收紧。关键词会检查任务文档（包括问题文本），调用方可以有意触发它们。请把它们用作路由提示，而不是授权手段或更高准确率的承诺。

#### 选择接受证据 {#choose-acceptance-evidence}

未校准的规则可以检查以下观测值：

| 字段 | 含义 |
| --- | --- |
| `top_probability` | 原生答案分布中的最大概率；对于 `noul`，为 `max(p, 1-p)` |
| `confidence` | 原生模型报告的置信度（如果有）：System One 为 Choice 和 Score 发布的统计量（见 [Decisions](https://vllm-sr.ai/docs/next/model-runtime/reference#decisions)）；自运行时约定 3.2.0 起其取值与早期版本不同，请重新检查阈值 |
| `probability_margin` | 对于 `noul`，为 `2 × abs(p - 0.5)`；无论是有把握的真还是有把握的假，差值都很大 |

规则用 `gt`、`gte`、`lt` 或 `lte` 约束其字段，取值在 `[0, 1]` 内。规则可以指定 `question_type`、具名的 `question` 或两者，并可选地指定具名的 `state`。所有必需的答案都需要规则覆盖。阶段本地的 `accept` 规则会增加一个条件；它不能放宽算法的通用质量规则。

模型的高概率并不是经过测量的准确率保证。`quality.type: calibrated` 则引用 `evaluation.calibrations` 中的一个不可变资源，声明 `loss: bundle_error` 和显式的 `max_risk`。运行时必须同时校验该工件的哈希以及它对所部署模型、任务和到达人群的适用性。缺失或不兼容的工件绝不能悄悄变成未校准的通过。

#### 使用独立认证的接受 {#use-independently-certified-acceptance}

校准执行是一条实验性的、更窄的路径。目前它支持没有信号或投影的配方中的原生阶段。评审阶段和依赖模型的信号路径需要额外的推理溯源信息，才能使用认证的接受。未校准的阶段仍可使用完整的路由路径和可选的评审。

准备 `systemone-calibration/v1` 工件的步骤：

1. 冻结任务的问题指令、判据和推理选项、策略，以及每个概率分箱的接受决定。输入文本可以变化；更改问题会产生不同的任务模板。`options.return_meta` 控制证据可见性，不改变任务身份。
2. 独立于训练和阈值选择收集新的认证样本组。按源问题集合计数，而不是按其中的问题或增强副本计数。统计假设是从声明的条件人群中独立同分布抽样；分层基准平均值并不会自动满足这一点。
3. 记录每个响应的完整原生到达历史和实际模型身份。运行时会检查模型、修订版本、内容哈希、引擎、数据类型、数值精度和加速器。仅有模型名是不够的。
4. 记录每个冻结分箱中的问题集合错误计数，并保留源清单、标注规则和门限定义摘要以供审计。当任何必需答案未通过样本组预先声明的正确性规则时，该问题集合即为错误。工件的 `certification` 标识此被评估人群并声明独立抽样方法。

运行时根据计数计算单侧 Clopper–Pearson 上界，并对所有声明的分箱进行 Bonferroni 校正。每个预先声明为接受的分箱都必须满足 `max_risk`，否则加载失败。计数不能把先前拒绝的分箱变成接受的分箱：这样做会改变后续阶段的到达人群。未被访问的分箱只有在保持拒绝时才可以没有样本。

该工件绑定的是配方的执行语义和冻结门限，而不是全局配置哈希。移动凭据、更改监听器或更新说明文字都不会使其失效。缺失运行时身份、未知任务、错误历史或未覆盖的分箱会使请求处于未解决状态。工件的声明仍需要数据集审计；该上界描述的是其被评估的人群，不对分布漂移或单个请求作任何保证。

Go 工件辅助函数是 `pkg/systemone` 中的 `CalibrationTaskSHA256`、`CalibrationPolicySHA256` 和 `CalibrationGateSHA256`。试点调参结果不是认证工件：即使在 96 个独立样本中零错误，在对多个分箱进行校正之前，95% 置信度下的风险上界也约为 3.1%。

#### 添加可选的 LLM 评审 {#add-an-optional-llm-judge}

`kind: judge` 阶段调用一个声明的 OpenAI 兼容提供商别名，并从先前的原生尝试中选出一个完整响应。它保留该响应的原生答案概率；LLM 自报的置信度不会替代它们。声明 `generation.max_output_tokens` 以限制其输出。

对于未校准的实验，请在通用质量规则中保留一个最低接受下限，并把更严格的提前退出阈值放在原生阶段的 `accept` 规则中。这样，评审可以选择一个满足通用下限的有效早期答案，即使它没有达到提前退出阈值。如果通用规则本身拒绝该答案，评审仅凭选中它并不能让它通过。评审是需要评估的额外证据，而不是准确率保证。

```yaml
stages:
  - model: kai
    accept:
      native:
        - {question_type: choice, field: top_probability, gte: 0.9}
  - model: vega
    accept:
      native:
        - {question_type: choice, field: top_probability, gte: 0.9}
  - name: review
    kind: judge
    model: reviewer
    generation: {max_output_tokens: 256}
```

把 `reviewer` 加入决策的 `modelRefs`，用 `api_format: openai` 连接它，并增加该算法的预算以允许第三次调用。评审与原生阶段消耗同一个共享的请求截止时间和调用账本。其输出是从先前完整原生响应中做出的选择，而不是由 LLM 合成的替代答案。

#### 衡量取舍 {#measure-the-tradeoff}

在相同的请求上比较直接使用 Kai、直接使用 Vega 和 escalate。在独立的校准数据上选择接受阈值，然后在评估前冻结完整配置。同时报告完整响应、未解决请求、升级率和测得的前端延迟。当请求分布变化时，阈值并不能保证固定的快速路径比例。

衡量请求总成本时，请计入依赖模型的信号、重试和评审调用；算法调用次数只覆盖决策被选中后的执行。

离线重放有助于选择实验；它不能确定实时延迟或 GPU 成本。[校准工具](https://github.com/vllm-project/semantic-router/tree/main/tools/calibration/systemone_auto)包含一个固定的公开 JevBench 命令和独立的研究重放工具。

### 从 confidence 和 cascade 迁移 {#migrating-from-confidence-and-cascade}

`escalate` 取代了 Looper `confidence` 算法和原生 `cascade`。`vllm-sr config migrate` 会改写这两者：

- `confidence.confidence_method` 和 `threshold` 变为 `escalate.accept.confidence.method` 和 `gte`；`escalation_order: size` 或 `small_to_large` 变为 `order: size`，`cost` 保持 `cost`，`declared` 保持 `modelRefs` 的顺序。
- `confidence_method: automix_entailment` 变为 `accept.systemone`：决策模型的 `answer_check` 问题取代外部校验服务器，其阈值变为 `gte`。请重新拟合该阈值。
- `confidence.on_error: fail` 变为 `algorithm.on_error: fail`，现在它在没有回答被接受时使请求失败；失败的调用会继续尝试下一个候选，而不是终止运行。
- cascade 的 `quality` 和 `stages` 移到 `escalate` 下，每个 `acceptance` 变为 `accept.native`，谓词边界直接写在每条规则上，原生阶段去掉 `kind: native`。

可复用片段：[`config/fragments/algorithm/looper/escalate.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/looper/escalate.yaml)。

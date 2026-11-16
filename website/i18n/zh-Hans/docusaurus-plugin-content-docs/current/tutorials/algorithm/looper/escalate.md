---
is_mtpe: false
translation:
  source_commit: "bc5c6bfb747ffe9c5f35d98b203427dada8c4c56"
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

`order` 设置升级顺序：`size` 先尝试 `providers.models[].param_size` 最小的模型，`cost` 先尝试定价最低的模型；省略时保持 `modelRefs` 的顺序。`size` 读取规模开头的数值，因此 `397B (17B active)` 为 397B，`1.6T` 为 1600B。无法读取规模的模型，或在 `cost` 下没有定价的模型，最后尝试。

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

`systemone: {gte, deployment, timeout_ms}` 就请求和候选回答，向决策模型（或指定的 System One 部署）提出 `answer_check` 预设问题。请求指最后一个含有用户自身文字的用户轮次中的这些文字：其字符串内容或拼接后的文本片段，并略去 `<system-reminder>` 块。只含提醒块的轮次会把请求留给更早的轮次；如果没有任何轮次含有用户自身的文字，则以最后一个轮次的提醒块作为请求。`refusal: false` 提出 `refusal` 预设问题；`refusal: true` 会被拒绝，因为它只会接受拒答。两种检查对每个被评判的回答各进行一次决策模型调用；该调用计入执行器的 `deadline`，但不计入 `max_calls`。调用工具的回答在评判时，于文本之后附上每个调用的工具名和参数。

`on_error` 决定没有回答被接受时的处理方式：

- `last_attempt`（默认）返回最后一个完成检查的回答。
- `fail` 使请求失败。

模型调用失败，或回答的检查无法运行（例如置信度方法需要的对数概率没有返回）时，总是继续尝试下一个候选。如果没有任何候选返回经过评判的回答，请求失败。

`budget` 约束整次运行：`max_calls` 限制候选调用（以及 `self_verify` 的额外调用）次数，且低于 Router 的单请求上限；`deadline` 限制它们的总耗时。耗尽预算的运行会失败。

#### 尝试可观测性 {#attempt-observability}

Escalate 执行会创建一个 `looper.execute` trace span，并为每次分派的候选或校验调用创建一个 `looper.attempt` 子 span。Router Replay 在 `route_diagnostics.looper` 下保存相应的、有界且不含内容的尝试记录，包括状态、阈值结果、token 用量、延迟和最终尝试序号。该结构从不包含提示词、响应、推理内容、端点 URL、凭据或原始错误。

每次升级都会把请求发送给另一个候选模型。请确保每个候选都符合该路由的数据策略，并为最坏情况下的调用链设定延迟和成本上限。

### 在 System One 上 {#on-system-one}

以 `api: systemone` 发布的配方用 `escalate.stages` 运行 escalate。本指南从 **Decision 2.0 Kai 0.6B → Vega 27B** 开始：

- Kai 回答原始请求，escalate 先检查这个回答，再决定是否调用 Vega。
- 不需要单独的、强制的 Kai 分类调用，配方也不会在第一个回答之后重新运行。
- 它支持 `choice`、`score`、`noul` 和 `box` 请求，并保持原始问题包完整。只有当阶段的模型能回答请求所需的一切时，该阶段才会运行；参见[按模型能回答的内容路由](#route-by-what-each-model-answers)。

#### 连接模型 {#connect-the-models}

模型别名是 `providers.models` 下的共享资源。本地别名指向模型运行时部署；远程别名指向具体的 Engine 或兼容的 System One 端点。每个别名只选择一种传输方式。

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

- `provider_model_id` 是发往远程端点的模型名。
- 如果后端需要身份验证，请在它的 backend ref 上设置 `auth: {api_key_env: <VARIABLE>}`。凭据只留在该后端上。
- 同一远程模型的副本放在它的 `backend_refs` 中；不同的模型是不同的别名和阶段。

#### 发布 auto 入口 {#publish-an-auto-entrypoint}

按 API 区分的入口让该配方以 `vllm-sr/auto` 的名字对外提供。listener 必须显式允许这个原生名称。Chat 和 System One 的别名相互独立：发布这个入口不会改变 Chat 的默认配方。

```yaml
listeners:
  - name: inference
    address: 127.0.0.1
    port: 8801
    apis:
      systemone:
        models: [vllm-sr/auto]

entrypoints:
  - api: systemone
    model_names: [vllm-sr/auto]
    recipe: native-escalate
```

在把 listener 暴露到受信任的开发环境之外之前，请为它添加客户端身份验证。如果还想允许某个具体模型，请把它的原生 provider 别名加入 `apis.systemone.models`。

#### 定义阶段 {#define-the-stages}

下面的片段接受 `choice`、`score` 和 `noul` 请求。只有当每个回答都通过其类型对应的门槛时，Kai 才会提前结束。

:::warning 该阈值是试点工作点
它是为 Decision 2.0 Kai → Vega 冻结的，不是推荐的默认值，也不是准确率保证。[复现指南](https://github.com/vllm-project/semantic-router/tree/main/tools/calibration/systemone_auto)固定了该实验的模型和公开基准。请在独立的校准数据上为你的工作负载选择阈值，并在评估前冻结。
:::

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

- **公共规则检查有效性，而不是准确率。** `gte: 0` 规则要求每个回答都是有效的分布；它们并不声明任何准确率下限。
- **Kai 未通过门槛后由 Vega 回答。** Kai 未通过更严格的提前结束门槛后，Vega 可以返回一个完整有效的回答。
- **预算。** 这个示例没有依赖模型的路由信号，因此它的两次调用预算覆盖 Kai，以及需要时的 Vega。传输重试消耗一次调用后，后续阶段可用的调用就会减少。

阶段规则：

- 除非写了 `kind: judge`，阶段运行一个原生模型；除非设置了 `name`，阶段以它的模型命名。
- `modelRefs` 是完整的候选名单。每个阶段指定其中一个成员，可以设置自己的超时，并且不能添加决策没有声明的模型。
- 阶段名不能使用 `abstain`，它保留给评审的“不选择”结果。

预算与截止时间：

- **每个请求一份预算。** Router 在公共请求处理器开始时（读取请求体之前）启动一个截止时间和物理调用账本。
- **上限**取配方中各原生决策最大的 `algorithm.budget.deadline` 和 `algorithm.budget.max_calls`。选中决策后，两者收紧到该决策自己的预算。
- **信号也计入。** 决策信号和内置模型信号在同一截止时间和账本内运行；它们自己的超时仍是上限。
- **每一次物理交换都计入**，包括远程决策模型调用和提供方的传输重试。信号遥测记录信号的交换，每个阶段的调用遥测包括它的重试和评审调用。

预算耗尽时：

| 情况 | 结果 |
| --- | --- |
| 信号耗尽了请求上限 | 其余信号保持未知，执行停止。 |
| 选中决策更紧的预算已经用完 | 它的算法不会启动。 |

两种情况都返回带 `reason: max_calls` 或 `reason: deadline` 的 `systemone_budget_exhausted`，`model_calls` 报告目前为止计入的所有物理交换。账本看不到隐藏在不透明外部提供方内部的工作。

接受：

- **接受响应前会校验每个必需的回答。** 缺失的回答、无效的值、不支持的问题类型以及未经证实的完整输入覆盖，都不会成为成功的结果。
- **没有阶段接受即为未解决。** System One escalate 总是失败，而不是返回被拒绝的回答，因此它的 `on_error` 只能是 `fail`。
- **没有 `quality` 时，**阶段接受任何通过它自身规则的完整回答。

:::note
原生预算不能替代整个请求的 HTTP 服务器、代理和客户端超时。例如，Dashboard 的运维诊断使用管理 API 两分钟的响应写入上限。
:::

#### 试用路由 {#try-the-route}

把模型、listener、入口和配方片段合并到一个 `version: v0.5` 的配置文件中，连接 Vega Engine 端点，然后启动 Router：

```bash
vllm-sr serve --config config.yaml
```

然后向它的原生端点发送请求。问题名称和标准会原样到达每个被尝试的模型。如果启用了身份验证，请使用 listener 的 API key。

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

响应的 `routing` 字段报告所选模型、阶段以及算法的物理模型调用次数（包括重试）。带有 `systemone_unresolved` 的 `503` 表示没有完整答案通过所配置的规则；它并不表示被省略的答案为假。响应字段和错误约定见 [Router API（英文）](https://vllm-sr.ai/docs/api/router#route-a-system-one-request)。

在 Dashboard 中试用：

1. 打开 **System One → Decision Playground**，在 **Automatic routes** 下选择 `vllm-sr/auto`。
2. 加载你的配方支持的示例，或输入你自己的问题，然后选择 **Run**。Dashboard 使用你登录后的管理权限，因此不需要公共 API key。
3. 打开 **Decision Monitoring**，查看阶段结果、请求速率和延迟。

这些视图描述的是线上执行；要衡量准确率，请单独评估带标签的请求。

#### 按模型能回答的内容路由 {#route-by-what-each-model-answers}

在第一个阶段之前，Router 会把请求所需的内容与每个阶段的模型能回答的内容进行比较：

- **请求所需的内容：** 它的问题类型；带图像时需要 `image`，带视频时需要 `video`（位于 `images`、`videos` 或 `assets` 中）；有问题设置了 `require_full_input` 时需要 `full_input`。
- **模型能回答的内容**取自下列来源中第一个给出答案的：
  - 其别名所指部署的模型卡，由模型运行时报告，或由 `provider: systemone` API 列出；
  - 其后端为它的 `provider_model_id` 列出的条目（`vllm-sr/5.0` 下为端点旁的 `GET /v1/models`，`vllm-sr/4.0` 下为 `GET <endpoint>/models`），每份配置只读取一次；
  - 部署的 `capabilities`（`question_types`、`modalities`），或运行时所服务模型的内置目录。

  没有任何来源描述的模型照旧被询问。
- **模型缺少所需内容的阶段会被跳过**，不发起调用，并以 `outcome="incapable"` 计入 `sr_systemone_stage_total`。
- **没有任何阶段的模型能回答时**，请求以 `400` 和 `unsupported_capability` 失败。其 `detail` 写明缺少什么，以及每个模型缺少什么、这一信息来自哪里：

  ```json
  {
    "detail": "no candidate model of decision \"answer\" answers this request, which needs box, image: kai lacks box, image (card); vega lacks box (listing)",
    "code": "unsupported_capability"
  }
  ```

- **只需要针对文本的 `choice`、`noul` 和 `score` 的请求不做检查**，因为每个 System One 模型都能回答它。

因此同一个池里可以同时放文本模型和视觉模型：`choice` 请求照旧从第一个阶段开始，`box` 问题只交给视觉模型。对于什么都不列出的 API，在其别名所指的部署上声明它的模型能回答什么：

```yaml
providers:
  models:
    - name: vision
      api_format: systemone
      deployment: gateway-vision

global:
  model_catalog:
    deployments:
      gateway-vision:
        provider: systemone
        endpoint: https://gateway.example.com/v1/systemone
        model: vision-27b
        capabilities:
          question_types: [choice, noul, score, box]
          modalities: [text, image]
```

`box` 答案的框位于图像之内时才算完整。接受规则可以用 `{question_type: box, field: confidence, gte: ...}` 对它把关。经过校准的策略永远不会接受它，因为校准只覆盖 `choice`、`score` 和 `noul`。auto 目前还不路由 `set`、`span` 或 `track` 问题：它们的答案因模型家族而异，因此带这类问题的请求会以 `400` 被拒绝。

#### 添加路由信号 {#add-routing-signals}

原生信号检查完整的任务文档，包括其状态和问题。请求不携带 Chat 对话或受信任的身份，因此引用 `authz`、`metadata`、`conversation`、`turn_relation`、`user_feedback` 和 `input_modality`（包括通过投影引用）会导致配置校验失败。

受支持的依赖模型的信号在共享的请求截止时间和调用账本之内保留各自的传输超时。

原生配方保留和去掉的内容：

- **保留：** 配方的 `strategy` 仍在匹配的决策之间选择，提供方层面的重试和超时设置仍适用于远程交换。
- **去掉：** 原生执行使用显式的阶段和预算，而不是 Chat 回退、候选要求、决策可靠性、输出契约或自适应控制。请为这个配方关闭 Chat 回退。
- **只有别名：** `modelRefs` 只声明别名，由阶段顺序决定选择，因此 Chat 推理控制和模型权重会被拒绝。

一个配方可以包含多个决策，每个决策有自己的阶段和预算。例如，关键词决策可以把提到申诉的请求用一个阶段直接发给 Vega，而默认决策尝试 Kai → Vega。只有被选中的决策会运行，它的预算从共享的请求上限收紧而来。

:::warning 关键词是提示，不是授权
关键词会检查任务文档（包括问题文本），因此调用方可以有意触发它们。不要把它们用于授权，也不要把它们当作更高准确率的保证。
:::

#### 选择接受证据 {#choose-acceptance-evidence}

未校准的规则可以检查以下观测值：

| 字段 | 含义 |
| --- | --- |
| `top_probability` | 原生回答分布中的最大概率；对 `noul` 为 `max(p, 1-p)` |
| `confidence` | 原生模型报告的置信度（如果有）：System One 为 Choice 和 Score 发布的统计量（见 [Decisions](https://vllm-sr.ai/docs/model-runtime/reference/http-api#decisions)）；自运行时约定 3.2.0 起其取值与早期版本不同，请重新检查阈值 |
| `probability_margin` | 对 `noul` 为 `2 × abs(p - 0.5)`；自信的真和自信的假回答都有较大的间隔 |

- **边界。** 规则用 `gt`、`gte`、`lt` 或 `lte` 限定它的字段，取值在 `[0, 1]` 内。
- **目标。** 规则可以针对 `question_type`、具名的 `question` 或两者，还可以针对具名的 `state`。每个必需的回答都需要规则覆盖。
- **阶段规则只能增加条件。** 阶段本地的 `accept` 规则增加一个条件；它不能放宽算法的公共质量规则。

模型给出的高概率并不是经过测量的准确率保证。`quality.type: calibrated` 改为引用 `evaluation.calibrations` 中的一个不可变资源，并声明 `loss: bundle_error` 和明确的 `max_risk`。运行时必须校验工件的哈希，以及它对所部署模型、任务和到达总体的适用性；缺失或不兼容的工件绝不会悄悄变成未校准的通过。

#### 使用独立认证的接受 {#use-independently-certified-acceptance}

:::warning 实验性
校准执行是一条更窄的路径。目前它支持在没有信号或投影的配方中使用原生阶段；评审和依赖模型的信号需要更多推理来源信息，才能使用它。未校准的阶段仍可以使用完整的路由路径和可选的评审。
:::

准备 `systemone-calibration/v1` 工件：

1. **冻结任务：** 它的问题说明、标准和推理选项、策略，以及每个概率区间的接受决定。
   - 输入文本可以变化；修改问题会产生不同的任务模板。
   - `options.return_meta` 控制证据的可见性，不改变任务身份。
2. **收集新的认证样本集**，与训练和阈值选择相互独立。
   - 按来源问题包计数，而不是按其中的问题或增强副本计数。
   - 统计假设样本是从所声明的条件总体中独立同分布抽取的；分层基准的平均值并不会自动满足这一点。
3. **记录完整的原生到达历史**，以及每个响应中实际的模型身份。运行时会检查模型、revision、内容哈希、引擎、dtype、数值和加速器；只有模型名是不够的。
4. **记录每个冻结区间中的问题包错误计数**，并保留来源清单、标注规则和门槛定义摘要以供审计。
   - 只要某个必需的回答不符合样本集预先声明的正确性规则，该问题包就算错误。
   - 工件的 `certification` 标识所评估的总体，并声明独立的抽样方法。

运行时如何使用这些计数：

- 它计算单侧 Clopper–Pearson 上界，并在所有声明的区间之间做 Bonferroni 校正。
- 每个预先声明为接受的区间都必须满足 `max_risk`，否则加载失败。
- 计数不能把拒绝区间变成接受区间，因为那会改变后续阶段的到达总体。只有保持拒绝的区间，才可以在未被访问时没有样本。

工件绑定的内容：

- **配方的执行语义和冻结的门槛**，而不是全局配置哈希。移动凭据、修改 listener 或编辑说明文字都不会使它失效。
- **未解决的请求：** 缺失运行时身份、未知任务、错误历史或未覆盖的区间，都会让请求保持未解决。
- **它的局限：** 声明仍需要数据集审计。该上界描述的是所评估的总体，对分布漂移或单个请求不作任何保证。

Go 中的工件辅助函数是 `pkg/systemone` 中的 `CalibrationTaskSHA256`、`CalibrationPolicySHA256` 和 `CalibrationGateSHA256`。

:::note
试点调参结果不是认证工件。即使 96 个独立样本中零错误，在校正多个区间之前，95% 置信度下的风险上界也约为 3.1%。
:::

#### 添加可选的 LLM 评审 {#add-an-optional-llm-judge}

`kind: judge` 阶段调用一个已声明的 OpenAI 兼容 provider 别名，并从之前的原生尝试中选出一个完整响应。它保留该响应的原生回答概率；LLM 自报的置信度不能替代它们。请声明 `generation.max_output_tokens` 来限制它的输出。

对于未校准的实验：

- 在公共质量规则中保留最低接受下限，把更严格的提前结束阈值放在原生阶段的 `accept` 规则中。
- 这样，即使某个之前的有效回答没有达到提前结束阈值，只要满足公共下限，评审仍可以选择它。
- 如果公共规则拒绝了某个回答，评审不能仅仅通过选择它而让它通过。

评审是需要评估的额外证据，而不是准确率保证。

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

把 `reviewer` 加入决策的 `modelRefs`，用 `api_format: openai` 连接它，并增加该算法的预算以允许第三次调用。评审与原生阶段消耗同一个共享的请求截止时间和调用账本。它的输出是从之前完整的原生响应中做出的选择，而不是由 LLM 合成的替代回答。

#### 衡量取舍 {#measure-the-tradeoff}

- **在相同的请求上比较：** 直接使用 Kai、直接使用 Vega，以及 escalate。
- **评估前冻结。** 在单独的校准数据上选择接受阈值，然后冻结完整配置。
- **一并报告：** 完整响应、未解决的请求、升级率和测得的前端延迟。
- **预期会漂移。** 当请求分布变化时，阈值并不能保证固定的快速路径比例。

衡量请求总成本时，请把依赖模型的信号、重试和评审调用都计算在内；算法调用次数只覆盖选中决策之后的执行。

离线回放有助于选择实验，但不能确定线上延迟或 GPU 成本。[校准工具](https://github.com/vllm-project/semantic-router/tree/main/tools/calibration/systemone_auto)包括固定版本的公开 JevBench 命令和单独的研究回放工具。

### 从 confidence 和 cascade 迁移 {#migrating-from-confidence-and-cascade}

`escalate` 取代了 Looper `confidence` 算法和原生 `cascade`。`vllm-sr config migrate` 会改写这两者：

- `confidence.confidence_method` 和 `threshold` 变为 `escalate.accept.confidence.method` 和 `gte`；`escalation_order: size` 或 `small_to_large` 变为 `order: size`，`cost` 保持 `cost`，`declared` 保持 `modelRefs` 的顺序。
- `confidence_method: automix_entailment` 变为 `accept.systemone`：决策模型的 `answer_check` 问题取代外部校验服务器，其阈值变为 `gte`。请重新拟合该阈值。
- `confidence.on_error: fail` 变为 `algorithm.on_error: fail`，现在它在没有回答被接受时使请求失败；失败的调用会继续尝试下一个候选，而不是终止运行。
- cascade 的 `quality` 和 `stages` 移到 `escalate` 下，每个 `acceptance` 变为 `accept.native`，谓词边界直接写在每条规则上，原生阶段去掉 `kind: native`。

可复用片段：[`config/fragments/algorithm/looper/escalate.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/looper/escalate.yaml)。

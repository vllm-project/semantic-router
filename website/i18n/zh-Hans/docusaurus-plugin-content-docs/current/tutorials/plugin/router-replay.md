---
is_mtpe: false
translation:
  source_commit: "bc5c6bfb747ffe9c5f35d98b203427dada8c4c56"
  source_file: "docs/tutorials/plugin/router-replay.md"
  outdated: false
---

# 路由回放

## 概览 {#overview}

使用路由回放在 Dashboard 的 Insights 中检查请求：选中的路由、模型、token 用量、响应和工具轨迹。存储在 `global.services.router_replay` 中配置，开关和采集默认值在 `global.plugins` 的 `router_replay` 插件中配置；配方或决策的 `router_replay` 插件只需覆盖不同的采集字段。

## 解决什么问题？ {#what-problem-does-it-solve}

回放把请求级的证据集中在一起，便于理解一次路由选择，并跨请求追踪一段对话。

## 何时使用 {#when-to-use}

需要请求历史来检查或排查问题时，启用回放。请根据允许保留的数据，选择保留期限和采集上限。

## 配置 {#configuration}

### 重启后保留记录 {#keep-records-across-restarts}

五个内置 MoM 配方（包括 Vault）都默认启用 PostgreSQL 回放。本地 `vllm-sr serve` 会管理 PostgreSQL 及其持久卷。使用其他部署方式时，请在[共享的回放服务](../learning/memory-and-replay#configuration)中配置 PostgreSQL 连接。

通用的 `memory` 存储适合临时检查，但 Router 重启或重新加载配置时会丢失记录。需要持久的历史时，请使用 PostgreSQL 或 Redis，并为打算保留的数据设置保留期限。

### 设置共享的采集默认值 {#set-shared-capture-defaults}

```yaml
global:
  services:
    router_replay:
      store_backend: postgres
  plugins:
    - type: router_replay
      configuration:
        enabled: true
        capture_request_body: true
        capture_response_body: true
        capture_personal_data: false
        capture_tool_results: false
        max_records: 10000
        max_body_bytes: 4096
        max_tool_trace_bytes: 0
        max_tool_trace_steps: 100
```

没有 `router_replay` 插件时，回放保持关闭。`global.plugins` 中的插件保存共享的采集默认值；配方的插件覆盖它们，决策的插件再覆盖其配方的插件，逐字段生效。存储、保留期限和异步写入属于全局服务设置。

| 采集设置 | 默认值 |
| --- | --- |
| 请求体和响应体 | 采集 |
| 个人数据 | 采集 |
| 工具结果 | 脱敏 |
| `max_records` | 10,000 |
| `max_body_bytes` | 每个 body 4,096 字节 |
| `max_tool_trace_bytes` | 不限制单个字段 |
| `max_tool_trace_steps` | 100 |

### 限制某个决策的采集 {#limit-capture-on-a-decision}

把这个片段加入决策的 `plugins`，记录有上限的摘录：

```yaml
plugins:
  - type: router_replay
    configuration:
      enabled: true
      capture_request_body: true
      capture_response_body: false
      capture_personal_data: false
      max_body_bytes: 4096
      max_tool_trace_steps: 100
```

上限的计算方式：

- **按 UTF-8 字节计算**，适用于 body 和结构化文本。摘录在完整字符处结束，因此可能比上限略短。
- **被截断的原始 body** 可能不再是完整的 JSON；请检查它的截断标志。
- **工具轨迹**有自己的上限，与请求体和响应体分开。`max_tool_trace_bytes` 限制每个结构化文本字段：提示词、工具定义，以及每个轨迹步骤的文本、参数和输出。被截断的步骤带有 `truncated: true`。

body 开关覆盖的范围：

- **`capture_request_body: false`** 让记录不含任何请求文本：请求体、提示词、工具定义，以及工具轨迹中的每个请求步骤，包括用户轮次、工具参数和工具结果。
- **`capture_response_body: false`** 对响应同样处理：响应体、轨迹中的回答和工具调用、幻觉片段摘录，以及失败的回退尝试中提供方返回的内容；这些尝试只保留状态码和固定的原因。
- **始终保留：**步骤类型、工具名、调用 ID、结果状态和路由元数据。以这种方式省略的文本直接缺失，不标记为 `content_redacted`，因此 Dashboard 显示为未采集。

各层如何组合：

- **让某个决策退出采集**：设置 `configuration: {enabled: false}`。
- **修改整个部署的默认值**：在 `global.plugins` 的 `router_replay` 插件上设置 `enabled: false`，或者不写这个插件。配方或决策仍可单独开启。
- **省略的字段沿用**更通用一层的值。显式的 `false` 是一次覆盖；`max_tool_trace_bytes: 0` 或 `max_tool_trace_steps: 0` 会取消对应的上限。
- **在选出决策之前**，被拒绝的请求使用其配方的采集默认值；选出决策后，决策的覆盖才生效。

这些设置不会改变提供方的保留策略，也不会删除已有记录。

:::note 零值
显式的 `max_body_bytes: 0` 回退为 4,096 字节；对 memory 存储，`max_records: 0` 回退为 200 条记录。这些零值会覆盖继承来的值，所以要继承某个字段，请省略它。
:::

### 不把个人数据写入回放 {#keep-personal-data-out-of-replay}

在全局或某个决策的 `router_replay` 插件上设置 `capture_personal_data: false`：

- **保留的：**路由证据、模型、信号和检测到的 PII 类型。
- **含个人数据的请求会省略：**请求体和响应体、提示词、工具定义和工具轨迹。
- **请求从不等待这项检查。**除非某个路由决策已经用 PII 信号读过提示词，记录会先在不含提示词的情况下写入。随后，配方配置的 PII 规则只读取最新一条用户消息；这次读取没有发现个人数据时，提示词才会加入记录。这类读取同时最多两个，每个最长 30 秒。无法在这段时间内读完的读取不会开始，提示词照旧不加入记录，与读取超时的结果相同：例如按决策模型实测的速度 30 秒读不完的消息；在这一速度测得之前，还包括路由时防护扫描已经超时的消息。

如果没有 PII 证据、`pii` 规则的模型没有给出回答，或者配方没有 `pii` 规则，采集会保守地省略内容。PII 信号是可选的；配置 PII 信号后，它判定为不含个人数据的请求就可以采集内容。例如：

```yaml
recipes:
  - name: default
    signals:
      pii:
        - name: personal_data
```

### 为开发和审计采集工具结果 {#capture-tool-results-for-development-and-audit}

默认情况下，记录会保留每个客户端工具结果的调用 ID 及其成功或失败，但不保留输出；采集的请求体也会去掉工具结果。无论 PII 信号是否读取工具结果，都是如此。要调试智能体的工具循环，请在全局或某个决策的 `router_replay` 插件上设置 `capture_tool_results: true`：

```yaml
plugins:
  - type: router_replay
    configuration:
      enabled: true
      capture_request_body: true
      capture_tool_results: true
      max_tool_trace_bytes: 16384
```

开启后：

- **每个 `client_tool_result` 步骤**把结果文本存入 `output`，并标记 `content_redacted: false`。
- **采集的请求体**保留工具结果，最多 `max_body_bytes`。
- **`max_tool_trace_bytes`** 限制每条存储的结果；截断时会设置该步骤的 `truncated` 标志。
- **`capture_request_body: false`** 时仍不存储工具结果，因为它们属于请求文本。

:::tip 设置上限
默认值 `0` 会完整存储结果。较大的工具输出会让记录变大，存储也可能拒绝超出其大小上限的记录。
:::

工具结果是原始的客户端数据，因此：

- **`capture_tool_results: true` 要求 `capture_personal_data: true`。**如果全局默认值或任一决策生效的 `router_replay` 设置把 `capture_tool_results: true` 与 `capture_personal_data: false` 组合在一起，Router 会拒绝加载该配置。
- **掩码路由**从不记录其请求轨迹或请求体。
- **没有 `replay.detail` 的主体**收到的工具结果仍是脱敏的。

除非这些记录本就是有意保留的审计存储，否则生产环境请保持该选项关闭。

:::caution
请求体、响应和工具轨迹可能包含密钥或个人数据。只采集需要的内容，并用 `replay.read` 和 `replay.detail` 限制访问。参见[回放 API 和隐私控制](../../api/router#router-replay)。
:::

## 影子请求 {#shadow}

`shadow` 块把决策已批准请求的一份有上限、按比例抽样的副本发给候选模型，并把结果保存在回放记录中。主响应永远不会被改变或延迟，影子请求也只在该决策记录回放时运行。

推广一个模型需要来自生产形态请求的证据：离线评估覆盖不到真实的提示词，灰度发布又会让用户接触未经验证的模型。影子请求填补了这一空白。主模型仍回答每个请求，候选模型的结果则作为不可变的结果存储，供之后比较。

影子副本携带的内容：

- **已批准的请求。**它与构建主请求派发所依据的请求相同，在所有请求插件运行之后生成。
- **影子模型自己的编码。**它经过相同的编码和提供方适配步骤，因此推理控制遵循影子模型的模型族。它的寻址方式与主请求访问该后端时相同，包括 profile 的 `chat_path` 和 Azure OpenAI 的 `api_version` 查询参数。
- **不使用流式。**它总是以非流式发送。
- **只传 trace 头。**只传播 W3C 的 `traceparent` 和 `tracestate` 请求头。客户端的请求头（包括 `baggage`）都不传播。
- **不带主请求的凭据。**决策的请求头修改只留在主请求路径上，除非 `forward_headers` 列出了它们。已知的凭据载体和每个 profile 的 `auth.header` 即使被列出也会被丢弃，只使用 Router 为影子后端配置的静态凭据。
- **不跟随重定向。**调用只发往配置的后端地址；重定向响应永远不会被跟随。

适合使用影子请求的场景：

- 候选模型应在灰度发布前用实时流量评估
- 无论影子请求成功、失败还是超时，主响应都必须保持不变
- 这项观测的资源使用必须明确，并按路由设置上限
- 回放或审计工具需要按请求身份关联主请求和影子请求的观测

不要用它影响实时请求、评判输出质量或构建训练集。影子请求只记录结果。

:::note
`shadow` 块取代了 `shadow_dispatch` 插件；`vllm-sr config migrate` 会把后者改写为前者。
:::

### 配置影子请求 {#configure-the-shadow}

```yaml
plugins:
  - type: router_replay
    configuration:
      enabled: true
      shadow:
        model: candidate-model
        sample_rate: 0.05
        max_concurrency: 2
        max_queue_depth: 8
        timeout_seconds: 30
        max_response_bytes: 1048576
        max_retries: 0
        capture_response_body: false
        max_capture_bytes: 4096
        tls_skip_verify: false
        forward_headers: []
```

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `model` | 必填 | 接收影子副本的已配置逻辑模型。必须在 `providers.models` 中有后端。 |
| `sample_rate` | `1.0` | 发送影子副本的合格请求比例，取值 `[0, 1]`。`0` 保留影子声明，但从不派发。 |
| `max_concurrency` | `2` | 该决策同时进行的影子调用数。 |
| `max_queue_depth` | `8` | 等待空位的调用数。超出部分以原因 `queue_full` 丢弃。 |
| `timeout_seconds` | `30` | 排队等待加执行的截止时间，所有重试共用。 |
| `max_response_bytes` | `1048576` | 读取的最大影子响应体。更大的响应体以 `response_too_large` 失败。 |
| `max_retries` | `0` | 遇到传输错误或可重试状态码时的额外尝试次数，上限为 `3`。 |
| `capture_response_body` | `false` | 在结果中保存影子文本的有上限摘录。默认关闭，只保留大小、token 数和 SHA-256。 |
| `max_capture_bytes` | `4096` | 开启采集时摘录的上限。 |
| `tls_skip_verify` | `false` | 对证书由内部 CA 签发的 https 影子后端跳过证书校验。该设置只作用于影子 HTTP 客户端，不配置主后端的传输。 |
| `forward_headers` | `[]` | 影子副本可以携带的决策 `header_mutation` 名称，不区分大小写匹配。决策为主后端设置的其他内容都不会转发，因此像 `X-Internal-Token` 这样的自定义凭据只留在主请求路径上。已知的凭据载体（`Authorization`、`Proxy-Authorization`、`Cookie`、`x-api-key`、`api-key`、`x-goog-api-key`、`x-user-*-key` 请求头）即使被列出，也会在加载配置时被拒绝，并在运行时被丢弃。 |

当请求未被抽中，或主请求派发已经选中了影子模型时，影子请求会被跳过：只记录指标，不产生结果。算法为执行器（`escalate`、`fusion`、`remom`、`ratings` 或 `workflows`）的决策会在加载配置时拒绝影子请求，因为影子请求只在单模型的提供方派发上运行。

### 失败时放行 {#fail-open-behavior}

请求路径只做一次非阻塞的空位检查就返回；其余工作都在主请求派发的响应构建完成后，由有上限的工作线程运行。缓慢、不可用、格式错误或过载的影子端点都无法改变主响应或其延迟。每个影子请求恰好以一个结果结束：

| 结果 | 原因 |
| --- | --- |
| `completed` | `completed` |
| `failed` | `backend_unresolved`, `credential_unresolved`, `encode_failed`, `timeout`, `transport_error`, `upstream_status`, `redirect_rejected`, `response_too_large`, `malformed_response` |
| `dropped` | `queue_full`, `queue_timeout`, `router_closing`, `same_as_primary`, `internal_request`, `request_unavailable` |
| `sampled_out` | `sampled_out` |

- **指标。**结果和原因导出为 `sr_shadow_dispatch_total{decision,result,reason}`，另有 `sr_shadow_dispatch_latency_seconds`、`sr_shadow_dispatch_inflight` 和 `sr_shadow_dispatch_queued`。
- **因资源上限造成的丢弃**通过指标和结构化的 `shadow_dispatch_dropped` 事件报告，而不写入回放，因此过载的影子通道不会给回放存储增加负载。

### 解读失败 {#interpreting-failures}

`failed` 结果并不总是说明候选模型有问题。请结合结果元数据中的 `status_code`、`attempts` 和截断后的 `error` 一起解读原因：

| 含义 | 原因 | 如何解读 |
| --- | --- | --- |
| 候选模型拒绝了请求 | `upstream_status`（`status_code` 为 4xx） | 影子模型无法接受已批准的请求，例如不支持某个参数，或上下文窗口太小。计入候选模型。 |
| 候选模型的健康或容量 | `upstream_status`（`status_code` 为 5xx）、`timeout`、`transport_error` | 后端不可达、过载，或在 `max_retries` 次重试后仍未在 `timeout_seconds` 内完成。这衡量的是部署，不是回答质量。 |
| 候选模型尝试重定向 | `redirect_rejected`（`status_code` 为 3xx） | 后端返回了重定向。Router 从不跟随，因此提示词和影子凭据只到达了配置的后端。请把影子模型指向后端的最终地址。 |
| 候选模型的输出问题 | `malformed_response`、`response_too_large` | 后端作了回答，但响应体不是其线格式的有效响应，或超过了 `max_response_bytes`。 |
| Router 侧的问题，与候选模型无关 | `backend_unresolved`、`credential_unresolved`、`encode_failed` | Router 无法构建影子调用或为其寻址。请修正配置，并把这些结果排除在候选模型的比较之外。 |

`dropped` 和 `sampled_out` 永远不会进入回放记录。它们表示 Router 选择不发送影子请求，因此不包含关于候选模型的任何信息，只在指标和日志中可见。

### 回放与审计采集 {#replay-and-audit-capture}

完成和失败的影子请求会在主请求的回放记录中追加一条结果。结果只能追加：

| 部分 | 内容 |
| --- | --- |
| 结果 | `source: shadow_dispatch`、`target: model`、`target_ref: <shadow model>`、判定和原因 |
| 元数据 | 主请求和影子请求的身份；主模型和影子模型及其后端；决策和配方；抽样率；入队、开始和结束时间戳；排队等待和延迟；尝试次数；状态码；响应大小；停止原因；token 数；影子文本的 SHA-256 |

回放脱敏对影子结果的作用与对记录其他部分相同：没有内容权限的查看者能看到路由和耗时字段，但看不到 `target_ref`、`reason` 或 `metadata`。除非回放存储及其读者已获准接触提示词级的内容，否则请保持 `capture_response_body` 关闭。参见片段：
[`config/fragments/plugin/router-replay/shadow.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/router-replay/shadow.yaml)。

### 导出比较数据集 {#export-a-comparison-dataset}

采集到的结果可以通过 `GET /api/v1/observability/replays/dataset` 变成比较数据集。该端点读取常规回放过滤条件选中的记录，并返回一个带版本的清单，列出每个输入的主请求和影子请求两组结果。

清单由它自己的摘要标识，因此导出时要在请求中给出切分方案：

- **`seed`** 固定切分的分配。
- **每个重复的 `split`** 写作 `name:weight`，例如 `?seed=2026-q3&split=train:8&split=eval:2`。
- **可复现。**相同的记录在相同的种子和切分下会重建相同的清单，包括每个样本落入的切分，因此后来的观测永远不会移动已放置的样本。

```bash
curl -H "Authorization: Bearer $ROUTER_MANAGEMENT_TOKEN" \
  "$ROUTER_MANAGEMENT_URL/api/v1/observability/replays/dataset?recipe=vault&seed=2026-q3&split=train:8&split=eval:2"
```

清单包含的内容：

- **不含文本。**它包含身份、输出摘要和来源，从不包含提示词或响应文本，因此可以和它所支撑的数字一起发布。
- **完整的观测。**观测要么完整进入，要么完全不进入：失败、未完成、输入因截断而丢失或从未记录摘要的请求会被排除，并计入某个排除原因。`counts` 报告保留了什么、丢弃了什么。
- **完整的选择范围。**超过 5000 条记录的选择会被拒绝，而不是分页导出。请收窄过滤条件后重新导出。

实时流量通常由一个配方或决策主导，基于它构建的数据集读起来像是对整个 Router 的结论，实际上只是对该决策的结论。要平衡数据集：

- **`balance_by` 和 `balance_max`** 限制单个分组能贡献多少，可按 `recipe`、`decision` 或 `primary_model` 分组。两者必须同时给出，只给一个会被拒绝。
- **上限保留哪些行由种子决定**，而不是时间，因此平衡后的数据集是对流量的抽样，而不是对到达时间的抽样。
- **因平衡而丢弃的行**和其他排除一样，计入 `balance_cap`。

导出需要 `replay.read`，读取的记录与列表 API 相同。被比较的决策必须开启 body 采集，因为没有采集请求的观测会以 `request_body_missing` 被排除。

### 把比较交给评判者 {#hand-the-comparison-to-a-judge}

`GET /api/v1/observability/replays/dataset/judge-tasks` 接受与导出相同的选择范围和切分方案，外加一个 `blinding_key`。它返回成对的任务：一个主请求回答对一个影子请求回答，并附上各自收到的请求。

- **两种顺序。**每对都会以交换两侧的方式出现两次，因此偏好先出现的回答的评判者，会表现为两种顺序之间的不一致。
- **盲标签。**每一侧都带有一个不透明的组标签，既不指明模型，也不指明哪一侧是主请求，且每对的标签都不同。
- **密钥。**只有 `blinding_key` 能把标签映射回模型，所以不要让评判者拿到它。它不能等于会被发布的清单种子。

```bash
curl -H "Authorization: Bearer $ROUTER_MANAGEMENT_TOKEN" \
  "$ROUTER_MANAGEMENT_URL/api/v1/observability/replays/dataset/judge-tasks?recipe=vault&seed=2026-q3&split=train:8&split=eval:2&blinding_key=$JUDGE_KEY"
```

只有当请求和两个回答都能哈希回清单中的摘要时，才会构建一对：

- **输入摘要**覆盖去掉顶层模型字段（记录的请求中为 `Model`，客户端请求体中为 `model`）后的请求体，因为存储的请求写明了 Router 把它发给了哪个模型。这个请求体也是评判者读取的 `input`，每个任务都带有它的 `input_digest`。
- **版本。**清单为 `shadow-dataset.v2`，任务为 `shadow-judge-tasks.v2`。`shadow-dataset.v1` 清单会被拒绝。
- **影子文本**来自采集的摘录，因此影子决策需要开启 `capture_response_body`，并把 `max_capture_bytes` 设得足以容纳完整的回答。

被排除的对按原因计数：

| 原因 | 起因 |
| --- | --- |
| `arm_text_digest_mismatch` | 摘录被截断，或者响应阶段的插件在存储前改写了主请求的回答。 |
| `arm_text_missing` | 这一对完全没有文本。 |
| `input_text_digest_mismatch` | 请求与其摘要不匹配。 |
| `input_text_missing` | 缺少请求。 |

写出了自己模型名的回答会被标记为 `names_own_model`，因为任何标签都无法对评判者隐藏这一点。

这些任务包含提示词和回答文本，因此该路由需要 `replay.detail`，并直接拒绝没有该权限的调用方，而不是对任务脱敏。评判者在 Router 之外运行。

### 发布数据集及其评判 {#publish-the-dataset-and-its-judgments}

`tools/calibration/shadow-dataset` 把清单，以及评判者针对它返回的评判（可选），发布到一个存储目录；任何挂载为文件系统的对象存储都可以。该工具在写入任何内容之前会校验每一项输入：

- **清单**必须与它自己的摘要相符。
- **任务**必须正是清单和盲化密钥根据它们所带的文本生成的任务。密钥只为这项检查而读取，从不发布。
- **评判**必须回答这些任务，且不能包含评判记录未定义的字段。

```bash
make run-shadow-dataset GO_TOOL_ARGS="--dest /mnt/shadow-datasets \
  --manifest $PWD/manifest.json --tasks $PWD/tasks.json --judgments $PWD/judgments.json \
  --blinding-key $JUDGE_KEY"
```

| 内容 | 落盘位置 |
| --- | --- |
| 清单 | `manifests/<digest>/<digest of the file>.json` |
| 评判集 | `judgments/<manifest digest>/<digest of the set>.json` |
| 评判报告 | 评判集旁边的 `.report.json` |

- **清单路径**包含文件自身的摘要，因为清单摘要不包括计数，而同一批样本的两次导出可能排除了不同的内容。
- **报告**统计结果、位置选择，以及有多少对跟随了位置而不是回答。
- **按内容寻址。**每个名称都来自内容，因此重复发布相同的文件不会有任何变化，已发布的名称也永远不会被替换为不同的字节。
- **任务不公开。**评判任务只用于核对评判，从不发布，因为它们包含提示词和两个回答。

不要把数据集放进本仓库。

## 查看结果 {#read-the-results}

在 Dashboard 中打开 **Insights**，或使用[回放 API](../../api/router#router-replay) 过滤记录并查看会话。搜索按请求 ID、配方或会话 ID 查找记录，因此 Agent Harness 发送的会话 ID（例如 Claude Code 的 `x-claude-code-session-id`）能找到该会话的每次调用。记录的会话路由历史按顺序列出会话的调用，并用 Agent 身份标头的短哈希标出每次调用所属的子代理，主对话显示为 `main`。只有 `identity.headers.agent` 指定了该标头时才会记录哈希（见[会话标识](../../api/session-identification)）；未设置时每次调用都显示为 `main`。没有 `replay.detail` 权限的查看者拿不到哈希（与被脱敏的会话策略一致），因此同样每次调用都显示为 `main`。成本是[根据记录的 token 和配置的费率得出的估算](../../api/router#router-replay)，不是账单。缺失的价格和用量保持未知。

每条记录的 `route_diagnostics` 解释该路由：

| 字段 | 内容 |
| --- | --- |
| `decision_ranking` | 每条记录都有：策略、胜出者的层级、候选池是否按置信度排序以及由哪个决策终止、`runner_up` 及区分两者的键，以及列出比较值的 `reason`。它与 eval API 的报告一致。 |
| `prepared_dispatch` | 仅在详情响应中：主请求派发时发往提供方的确切请求体的带版本回执，包含其线格式、SHA-256 摘要和字节长度，但不含请求体副本。执行器调用和回退尝试不属于它。 |
| `looper` | 保留执行轨迹的执行器请求，例如 `escalate`：有上限的尝试、token 和成本统计、耗时、处置原因和最终尝试。 |

- `prepared_dispatch` 来源于内容，因此没有 `replay.detail` 的响应会省略它。
- `looper` 不包含提示词、工具参数、凭据和原始错误。它的尝试详情需要 `replay.detail`，任何截断都会被标注。

完整的决策片段见
[`config/fragments/plugin/router-replay/debug.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/router-replay/debug.yaml)。

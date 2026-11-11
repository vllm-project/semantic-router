---
title: System One API schema
description: Router 提供、调用或转发的每个决策模型 API 都有精确、带版本的 schema；listener 和上游共用一个 API 注册表，二者之间只有一层 codec。
is_mtpe: false
translation:
  source_commit: "6b782822edead98397acc6401112e6ec1e45af96"
  source_file: "docs/proposals/system-one-api-schemas.md"
  outdated: false
---

# System One API schema {#system-one-api-schemas}

**状态：** 已实现。

## 目标 {#goals}

1. **处处使用精确、带版本的 schema。** Router 对外提供、内部调用或在后端池中调用的每个决策模型 API，都遵循一个精确定义、带版本、保存在仓库中并经过一致性测试的 schema。
2. **`state` 始终是标准内容。** vllm-sr 只在自然的位置增加可选字段（输入和输出都是），因此标准客户端或服务器不受影响。批量中的每一项都是完整的标准请求。
3. **一层 codec** 在客户端和服务端两侧完成 schema 之间的转换，与 Router 的 LLM 协议 codec（`llmprotocol` 和 `protocolcodec`）对应：稳固、可插拔、易维护。
4. **Schema 文档** 由规范生成，并在其上提供指南。

## Schema {#schemas}

| ID | 定义 |
| --- | --- |
| `systemone/0.2` | 标准：公开的 System One API 0.2 版。已发布的 OpenAPI 文档没有许可证，因此不复制进仓库；`config/schemas/systemone/systemone-0.2.schema.yaml` 是我们用自己的话对该接口的描述（字段名、类型和结构），附带来源 URL 和核对时的 sha256。`make systemone-schemas-drift` 把它与已发布的文档进行比较。 |
| `vllm-sr/5.0` | `systemone/0.2` 加上 vllm-sr 扩展集 5.0，即 `config/schemas/systemone/vllm-sr-5.0.extensions.yaml`。它取代运行时契约 4.0.0。 |
| `vllm-sr/4.0` | 旧版，作为兼容超集保留：运行时契约 4.0.0 接受的一切（由 state 键推断的隐式角色、`choices`/`levels`/`labels`、`images`/`videos`、只有 preset 的问题、`/v1/systemone` 上的 `states` 批量、`{error}` 错误体、`data` 模型列表）。在 listener 上它还接受所有 5.0 形式，新客户端可以在同一端点使用 5.0，而现有客户端无需改动。其冻结的契约为 `config/schemas/systemone/vllm-sr-4.0.openapi.yaml`。 |

标准的新版本是新 ID（例如 `systemone/0.3`）；只要还有部署引用，旧 ID 就保留。标准和扩展集各自独立演进版本：5.x 小版本只增加字段，破坏性变更是新的大版本。

### listener 和上游共用一个 API 注册表 {#one-api-registry-for-listeners-and-upstreams}

一个注册表（`apiregistry`）描述 Router 使用的每个 API。每项声明其名称、路径、schema ID 及默认值、设置项、所服务的模型种类，以及每个 schema 的 codec：

| API | 路径 | 默认 schema | 其他 schema |
| --- | --- | --- | --- |
| `chat_completions` | `/v1/chat/completions` | `openai/v1` | |
| `responses` | `/v1/responses` | `openai/v1` | |
| `messages` | `/v1/messages` | `anthropic/2023-06-01` | |
| `images` | `/v1/images/*` | `openai/v1` | |
| `systemone` | `/v1/systemone`、`/v1/systemone/batch`、`/v1/decisions` | listener 上为 `vllm-sr/5.0`，上游为 `systemone/0.2` | `vllm-sr/4.0` |

API 按 API 本身命名，而不是按厂商命名：一个厂商会提供多个 API，许多厂商实现同一个 API。旧配置使用的名称 `openai` 和 `anthropic` 是 `chat_completions` 和 `messages` 的别名，凡是出现 API 名称的地方都会解析；旧配置无需修改即可运行。在同一个 `apis` 映射中两次给出同一个 API（别名和正式名称）是校验错误。

新 API 是一条注册表项，带它的路径、schema 和 codec；某个 API 的新版本是一个 schema ID 及其 codec。两者都不改变配置结构，也不影响任何调用方。

### listener API {#listener-apis}

listener 在一个 `apis` 映射中列出它提供的 API，键为注册表中的名称。每项结构相同：`schema`（只用于有版本的 API，目前是 `systemone`）、`models`（省略时为该 API 种类的全部模型）以及该 API 自己的设置（`systemone` 的 `max_deadline_ms`）。

- 没有 `apis` 时，listener 为所有模型提供 `chat_completions`、`responses`、`messages` 和 `images`；`systemone` 除非列出否则关闭，因为它直接发布决策模型。
- 有 `apis` 时，listener 只提供列出的 API；其他 API 路径返回 404，`detail` 中写明 listener。生成的 Envoy listener 只路由所提供 API 的路径。
- `GET /v1/models` 只列出所提供 API 的模型：LLM API 用 OpenAI 的 `data`，`systemone` 用 System One 的 `models`。
- listener 级别的 `models` 和 `systemone` 字段已移除；`vllm-sr config migrate` 会把它们移入 `apis`。

只有 System One 有版本：Chat Completions、Responses 和 Messages 以增量方式演进，而 System One 有不兼容的版本，为每个版本单独设路径会迫使旧客户端改 URL。

### `schema` 在哪里配置 {#where-schema-is-configured}

- **服务端：** `listeners[].apis.systemone.schema`，即 listener 对其客户端使用的契约；默认 `vllm-sr/5.0`。
- **客户端：** `providers.models[]` 上与 `api_format` 并列的 `schema`，以及部署上与 `provider: systemone` 并列的 `schema`；默认 `systemone/0.2`。`deviations` 和 `max_concurrency` 是部署的客户端设置。

### `vllm-sr/4.0` listener {#a-vllm-sr40-listener}

在一个生产部署上测得，公开端点有相当大比例的请求（大多来自外部客户端）使用只有 4.0 接受的形式，主要是内联 `images` 和 `states` 批量。因此 4.0 listener 在同一路径上、无需请求头地同时接受两种形式：

- 4.0 独有的形式正好是 5.0 服务器会拒绝的那些，所以读取它们不会改变合法 5.0 请求的含义。有两处重叠，即合法的 5.0 请求在 4.0 listener 上保留 4.0 的读法：没有 `roles` 的对象 state 保留 4.0 的隐式角色；由放在一个条目前面的 asset 占位符组成的 state 仍是一个为其 asset 加标签的数组，而 5.0 会把它读作该条目，媒体放在它前面。listener 对这两种情况都计数。
- 回答保持 4.0 形式；错误体同时带有 `detail`（标准）和 `error`（4.0），两类客户端都能读取。
- 每个 listener 都有一个按形式统计客户端所用 4.0 独有形式的计数器，告诉运维何时可以安全切换到 `vllm-sr/5.0`。

## 设计规则 {#design-rules}

1. **标准优先。** 标准能表达的，都用标准的方式表达；没有第二种写法。
2. **每条 vllm-sr 消息都是完整的标准消息。** 每个问题都带标准的 `instructions` 和 `criteria`；扩展只是附加提示。例如 `preset` 让 vllm-sr 服务器使用训练过的 head。
3. **扩展字段平铺在自然位置，** 没有命名空间对象，也没有请求头，与 vLLM 的额外参数一样：SDK 用户通过 `extra_body` 传递。
4. **未知字段以 400 拒绝并写明名称，** 在每个 vllm-sr 服务器上都如此，因此不会有字段被悄悄忽略，也不需要能力协商。
5. **state 是标准内容。** 它的键没有隐藏含义。区分角色的模型族读取显式的 `roles` 映射；没有 `roles` 时，整个 state 在所有模型族上都是用户内容。每个模型族保留自己把对象 state 渲染为模型输入的方式，就像 LLM 的对话模板：渲染是 API 背后的模型细节，模型的启动自检保持其输入不变。
6. **三层名称；部分名称保持不变。** API 字段由两份规范定义。state 内的键由客户端决定；Router 自己构建多部分 state 时，沿用模型训练和校准时使用的 `request`、`history`、`previous_turns`、`previous_answer`、`tool_results`、`candidate`、`answer` 和 `context`。Router 配置中 `input:` 的名称是同一组。没有任何模型输入发生变化，因此无需重新校准。
7. **遵循标准的限制和行为：** 状态码（400、带 `detail` 列表的 422、带 `Retry-After` 的 429 和 529）、长上下文请求头，以及最大 32 MiB 的请求体。
8. **批量是单独的扩展端点** `POST /v1/systemone/batch`，其 `requests` 中是完整的标准请求；`/v1/systemone` 只接受一个请求。`/v1/decisions` 保留为原生端点，在 200 中给出每个问题的错误；没有任何有效问题的请求在 Router 和运行时上都返回 400 `invalid_request`，并列出每个问题的错误。
9. **`vllm-sr/auto` 按能力路由。** 请求只发给能回答其全部问题类型、能读取其媒体（box 与图像、视频、完整输入等）的池成员，依据成员的模型卡或列表，否则依据其部署声明的 `capabilities`。没有成员能回答时，请求以 400 `unsupported_capability` 被拒绝，其 `detail` 写明缺少什么以及每个成员。

## 扩展集（vllm-sr/5.0） {#the-extension-set-vllm-sr50}

| 层级 | 字段 |
| --- | --- |
| 请求 | `roles`（state 键到 `user`、`context` 或 `answer`）；`options`（`deadline_ms`、`return_meta`、`max_tokens`） |
| 问题 | `preset`、`head`、`decision_adapter`、`over`、`overflow`、`require_full_input`、`threshold`、`at`、`rate` |
| 问题类型 | `span` 和 `set`，选项在 `criteria` 中；视频上的 `track` |
| 资源类型 | `video` |
| 答案 | `abstain_probability`、`input_coverage`；在 `/v1/decisions` 上还有 `error`、`reason` 和 `message` |
| 响应 | `spans`、`sets`、`thresholds`、`span_heads`、`meta`、`routing` |
| 错误 | 标准的 `detail`，加平铺的 `code`、`question`、`state`、`reason`、`limit` |
| 模型列表项 | `version`、`features`、`card` |
| 端点 | `POST /v1/systemone/batch`、`POST /v1/decisions`、`GET /health` |

因为标准已能表达而移除的：`choices`、`levels` 和 `labels`（改用 `criteria`）；`images` 和 `videos`（改用 `assets`）；顶层 `states`（改用批量端点）；`{error: {...}}`（改用 `detail` 和平铺字段）；`options.long_context`（改用标准请求头）；隐式角色（改用 `roles`）。

[扩展参考](../system-one/reference/extension-reference.md)和[兼容性矩阵](../system-one/reference/compatibility-matrix.md)由扩展规范生成。

## Router 作为客户端 {#the-router-as-a-client}

```yaml
deployments:
  gateway-d3:
    provider: systemone
    endpoint: https://gateway.example.com/v1/systemone
    model: d3
    schema: systemone/0.2      # 默认；vllm-sr serve 目标使用 vllm-sr/5.0
    deviations: [state_text]   # 只用于已声明并测试过的不一致
    max_concurrency: 8         # 默认：拆分后同时进行的调用数
```

- **`systemone/0.2`** 只发送标准的问题类型和字段。preset 根据 Router 的规范 preset 文本展开为显式的 `instructions` 和 `criteria`；扩展字段按从两份规范推导出的清单去除。标准无法携带的内容（span 和 set 问题、`over`、`head`、`decision_adapter`、视频、完整覆盖问题）在加载时以一行错误拒绝，写明规则和部署。批量和阶段 bundle 在 `max_concurrency` 和调用方截止时间内拆分，答案和每个 state 的错误重新组装；错过截止时间的调用只让它自己的问题变为未知。不发送截止时间字段。会读取 400 `detail`、422 `detail[]` 和 `Retry-After`；由于标准没有模型卡，使用部署声明的 `capabilities`。
- **`vllm-sr/5.0`** 使用完整的扩展集、批量端点和模型卡。
- **偏差（deviations）** 是在 codec 之后应用、可组合的改写器，每个都有自己的测试和文档行。第一个是 `state_text`：对象或数组 state 作为其精确的 `json.dumps(state, ensure_ascii=False)` 文本发送，Decision 3 渲染结果完全相同，Vela 会解析回对象。拒绝对象 state 的企业 LLM 网关声明为带 `[state_text]` 的 `systemone/0.2`。
- Router 的两条路径都使用 codec：`Decide`（Router 自己的决策信号、judge、escalate 检查和选择器）和 `Forward`（`vllm-sr/auto` 池），其答案和错误按客户端的形式重新编码，包括批量重新组装。

## Codec 架构 {#codec-architecture}

### IR 是唯一的中间表示 {#the-ir-is-the-only-pivot}

`systemoneprotocol` 定义与 schema 无关的表示：请求、批量、问题、答案、错误和模型列表，JSON 内容保持顺序。角色有三种模式：隐式（4.0 的，由 state 键推断）、无（没有 `roles` 的 5.0）和显式。所有转换都经过 IR：listener 的 codec 把客户端请求读入 IR，目标的 codec 把 IR 写成目标的 schema。

### 内部传输形式 {#the-internal-wire-form}

部署内部的跳转（Router 到模型运行时的 `/v1/bundle` 和 `/v1/decisions`，Engine 到其 worker）以内部形式携带 IR：4.0 请求形式加上 `roles`，其中没有 `roles` 表示 4.0 的隐式角色，`roles: null` 表示没有角色。内部 codec 不丢失信息：成员顺序、4.0 写法和数字字面量都保留，它读入的请求写回时保持不变。往返测试在黄金语料（由 schema 生成器从各模型族源码中提取的自检请求，以及 4.0 请求形式）和模糊测试语料上检查这一点。因此 Router 构建的每个模型输入都逐字节保持不变。

### 计划（Plan） {#plans}

listener 决定每个请求如何继续转发：

- **透传：** 内部形式能读取的请求（4.0 listener 上的每个 4.0 请求，以及每个 listener 上的 `/v1/decisions`）先解码并检查，再以客户端自己的字节转发。
- **转换：** 其他请求由 IR 写成内部形式：5.0 listener 的请求、批量端点、带长上下文请求头的请求。

客户端 codec 规划一个批量的传输调用：一次批量调用，或在滑动窗口内每个请求一次调用。拆分、preset 展开和重新组装都由 codec 的计划决定，从不由调用方决定。两类计划的诊断都进入指标和日志：`vsr_systemone_listener_diagnostics_total` 统计每个 listener 的旧形式以及透传和转换次数。

```go
type SchemaCodec interface {
    ID() SchemaID                    // "systemone/0.2", "vllm-sr/5.0"
    Spec() SpecRef                   // 规范文件及其 sha256
    Capabilities() CapabilitySet     // batch, roles, span, set, video, presets, ...
    Stateless() bool
    Plan(Batch, Target, Policy) (Plan, Diagnostics, error)
    DecodeReply(Plan, WireCall, WireReply) ([]ItemResult, error)
    ListingPath() string
    DecodeListing(status int, body []byte) (Listing, error)
}

type ListenerCodec interface {
    Read(path string, body []byte, header http.Header) (ClientRequest, error) // IR 及其计划
    AnswerSuccess(*ClientRequest, []byte) ([]byte, error)
    AnswerError(*ClientRequest, path string, status int, body []byte) (int, []byte)
    AnswerListing([]byte) ([]byte, error)
}

type Deviation interface {
    ID() string                      // "state_text"
    AppliesTo(SchemaID) bool
    RewriteRequest(*WireCall) (Diagnostics, error)
    RewriteReply(*WireReply) error
}
```

- **注册表** 不可变，启动时校验：ID 格式、无状态、无重复、每份规范的 sha256，以及每个偏差都适用于某个已注册的 schema。配置通过它解析 `schema` 和 `deviations`；未知 ID 是加载错误，并列出支持的 ID。
- **单一事实来源。** 扩展规范对每个字段只定义一次：层级、类型、出现的版本和支持它的模型族。生成器据此写出 Go 字段表、模型运行时的字段表、标准目标使用的去除清单、内部往返语料以及生成的参考页面。
- **CI 检查：** 生成产物是最新的；两份规范以不同含义定义同一字段时构建失败；一致性测试通过。
- **测试：** 每个 codec 和每个偏差的黄金测试、基于规范的请求和响应检查、IR 往返、跨模型族等价（对象 state 及其 `state_text`），以及针对 Python 的规范 JSON 编码器模糊测试。

## vllm-sr serve 作为标准服务器 {#vllm-sr-serve-as-a-standard-server}

模型运行时以及 Router 和 Engine 前端都提供标准：完全按标准定义接受标准请求，加上扩展字段，其他字段以 400 拒绝。`vllm-srun serve --systemone-schema` 设置运行时的 schema：默认 `vllm-sr/5.0`，或为直接服务 4.0 客户端的运行时设置 `vllm-sr/4.0`。运行时契约为 5.0.0。

## 上线 {#rollout}

这些工作在一个 pull request 中交付：规范、IR、注册表和 codec、listener `apis`、上游 `schema`、带 `state_text` 的 `systemone/0.2`、`vllm-sr/5.0`、`vllm-sr/4.0`、运行时的 5.0 接口、迁移和文档。

基于 Router 或 Engine 构建的托管 Decision API 分两步升级。第一步，升级所有组件（包括后端），Router 或 Engine 与运行时之间使用 `vllm-sr/5.0`，公开 listener 保持 `apis.systemone.schema: vllm-sr/4.0`。之后，把内部调用方改为 5.0 形式，并在 listener 的旧形式计数显示外部 4.0 流量已经消失后，把它切换到 `vllm-sr/5.0`。

## 风险与后续 {#risks-and-follow-ups}

- **标准还很年轻（0.2）。** 通过固定版本、版本 ID 以及漂移和冲突检查来缓解。
- **标准目标上的零样本回答。** 展开的 preset 得到零样本回答而不是训练过的 head，因此在训练 head 上调好的阈值在标准目标上需要重新校准。
- **意外的输入漂移。** 对 Router 的部分名称，显式角色精确再现 4.0 的隐式角色；测试覆盖了各模型族的读取逻辑，内部形式保持 Router 构建的每个输入不变。
- **5.0 会破坏依赖隐式角色、第二种写法或 `/v1/systemone` 上 `states` 的外部客户端；** 4.0 listener 让它们继续工作。
- **批量拆分会增加上游调用；** `max_concurrency` 和截止时间对其加以限制。
- **后续：** 运行时仍要求 choice 和 score 至少两个选项，noul 需要 instructions 或带描述的 criteria，而标准允许一个选项和裸 noul。
- **后续：** `vllm-sr/auto` 按能力路由 `box` 问题；`set`、`span` 和 `track` 的答案因模型家族而异（`sets`、`spans`），auto 会在后续改动中校验并路由它们。

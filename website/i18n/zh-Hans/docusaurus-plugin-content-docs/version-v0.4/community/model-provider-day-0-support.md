---
title: 模型与提供商 Day-0 支持
description: 一次加入内置模型或提供商，再从共享目录生成所有运行时和产品视图。
translation:
  source_commit: "2b7519a84aec96963b02a3534e82908beba33f76"
  source_file: "docs/community/model-provider-day-0-support.md"
  outdated: false
---

# 模型与提供商 Day-0 支持

内置支持是一份经过校验的资源图，而不是把名字加进几份互不相干的清单。`config/catalog/` 下的仓库目录会生成运行时注册表、内置发行、控制面板模型库和添加模型卡片，以及公开的[模型页](/models)。

## 选择最小改动

| 变更 | 目录资源 | 代码适配器 |
| --- | --- | --- |
| 在已有兼容提供商上新增模型 | Model Card；该提供商 `models[]` 中的一条；已知时再加推理/评测记录 | 否 |
| 新增兼容 OpenAI 或 Anthropic 的提供商 | 一份提供商文件及其内置 `models[]` 映射 | 否 |
| 新的线协议或真正不同的语义 | 协议与提供商定义、一致性夹具 | 是，在协议/认证/传输接缝 |
| 某位运维自托管的模型 | 不需要；写普通自定义模型绑定和可选 Model Card | 否 |

提供商卡片并不宣称每个模型都已内置。内置模型要有经过校验的 Model Card；托管模型只有在提供商 `models[]` 把它映射到提供商原生 ID 时才可选。因此每个活跃的物理 Model Card 都必须至少有一条提供商映射。虚拟配方不同：它们推荐的池可以点名运维定义的自定义模型，而不是内置目录的外键。

## 精选目录准入

内置物理目录按主流模型创建方精选，而不是按全球模型数量排名。对已有创建方，Day-0 贡献通常把目录补到或轮换到大约最近三代或有代表性的产品线。加入新创建方会改变 `manifest.yaml.inventory.physical` 中的已审基线；应说明该公司为何属于主流集合，以及哪些当前产品线构成有用的运维面。生成器强制创建方成员资格和最低深度，人工评审决定时效和相关性。

提供商支持是独立的。新的 Provider ID 可以没有任何内置模型映射，但对添加模型里的手写自定义模型仍然有价值。不要为了让提供商卡片看起来“有货”而加一张冷门 Model Card。

## 添加模型

1. 在 `config/catalog/resources/models/single/` 下新增或更新一份聚焦的家族文件。只记录内在事实：规范 ID、发布方与展示、分发来源/许可证、修订、限额、模态、能力、协议、生命周期，以及推理家族引用。由配方支撑的逻辑模型放在 `models/virtual/`。
2. 当 vLLM-SR 需要知道提供商原生模型 ID、协议限制、价格或其他提供商专有事实时，在 `config/catalog/resources/providers/<provider>.yaml` 的 `models[]` 下加映射。不要再造第二份提供商/模型清单。
3. 复用推理家族。只有请求投影本身是新的才加新家族；不要在用户配置里复制内置家族。
4. 只有结果可归因到主模型来源且可再分发时，才在 `evaluations/single/` 下加基准记录。保持每个基准版本、精确主体和原始指标明确。虚拟模型配方运行使用 `evaluations/virtual/` 下的同一 schema。知道运行日期时记录 `measured_at`；否则记录 `observed_at` 作为已发布数值的审阅日期，不要把它写成运行日期。
5. 为卡片/提供商映射声称的能力或协议行为添加一致性夹具。

模型 ID 带命名空间，例如 `organization/model`。基准和指数 ID 使用完整语义版本。数据集、评分器、提示词协议或聚合规则变更时，需要新的基准版本。

默认智能指数对 MMLU-Pro、GPQA Diamond、Humanity's Last Exam、SWE-bench Verified 和 Terminal-Bench 2.1 等权。覆盖率达到 60% 才给出头条分数。新模型可以带着更少证据发布；模型库会显示已有分量和 `Not yet measured`，而不是编造数值。同一模型和带版本指标出现两个可用值会被拒绝。厂商发布结果保留精确模型变体、推理模式、工具模式和 harness 元数据，并标为 claimed；只有带产物的冻结 vLLM-SR 运行才能标为 reproduced。

`reasoning_effort` 是运行时控制，不是证据来源。若分数未披露运行时 effort，使用 `unspecified`；把结果是厂商发布还是独立测量放在 `evidence.provenance` 和 `evidence.verification`。

生成会为每个模型和每个可选推理 effort 恰好发出这五个基准槽。没有可信证据的槽是显式 `missing`，绝不是零。来源报告多个 effort 时，为每个 effort 单独写评测记录；不要把 `high`、`xhigh` 或 `max` 的结果抄到其他行。额外基准仍可作为证据可见，但不会静默进入该版本的默认指数。

## 添加提供商

1. 在 `config/catalog/resources/providers/` 下的聚焦文件中，给提供商一个稳定的小写 Provider ID。
2. 只声明该提供商实际传输的协议，以及它实现的完全限定 `supported_operations` 子集。每个声明的协议都必须包含 `#create`；只有该提供商面真的暴露兼容发现时才加 `#list_models`。一等语义集成用 `native`，兼容 API 用 `compatible`，vLLM 或 SGLang 这类私有服务运行时用 `runtime`。
3. 仅在提供商改变推理字段位置时，才添加默认 base URL、认证策略、请求头/前缀、不含密钥的默认请求头、路径覆盖、API 版本行为，以及已有的 `reasoning_transport`。生成会拒绝密钥和带凭据的默认请求头。
   当提供商把所选级别作为 `output_config.effort` 传递时，复用 `output_config_effort`。只有激活是与模型 effort 阶梯分开的布尔控制时，才用 `activation_parameter`。
   对混合思考的 Chat API，开关属于本地 `chat_template_kwargs` 时选 `top_level_effort_template_switch`；两个字段都是顶层提供商扩展时选 `top_level_effort_boolean_switch`。Responses 保持标准的 `reasoning.effort` 形状。
4. 添加显示名、类别、标志来源或字母回退，以及一致性状态。缺图绝不能挡住配置。
5. 只有通用协议、URL 和认证处理无法表达该提供商时才加代码。把适配器放在一致性测试旁边；不要给通用配置 helper 加提供商分支。

提供商 API 操作定义在 `config/catalog/resources/protocols.yaml`。提供商用 `supported_operations` 选子集，并可覆盖所选操作路径；不能发明或覆盖未声明的操作。运行时分发、控制面板模型发现和连接校验，都从同一套定义解析路径、认证和不含密钥的请求头。提供商专有的推理放置复用目录中已校验的传输模式；端点主机名从不作为提供商身份。

### 推理线协议

运维使用一套与协议无关的决策面：`use_reasoning`、可选 `reasoning_mode`，以及可选 `reasoning_effort`。
内置模型家族限制可选值，提供商-模型映射可以再收窄。所选协议和目录传输随后只产生一种提供商原生形状：

| 提供商/运行时表面 | 最终请求控制 |
| --- | --- |
| 本地 vLLM/SGLang 模板控制 | `chat_template_kwargs` 内的开关和/或 effort |
| 本地混合 effort 加模板开关 | 顶层 effort 加 `chat_template_kwargs.<activation_parameter>` |
| 本地模板 effort 标志 | 激活布尔值加一个互斥 effort 标志；默认/完整可以是省略 |
| 兼容 OpenAI 的 Chat effort | 顶层 `reasoning_effort` |
| OpenAI Responses effort | `reasoning.effort` |
| 提供商布尔开关 | 顶层 `<parameter>: true\|false` |
| 网关归一化推理 | `reasoning.effort` 或 `reasoning.enabled` |
| thinking 对象 API | `thinking.type: enabled\|disabled\|adaptive` |
| thinking 对象加 effort | `thinking.type` 加顶层 `reasoning_effort` |
| DeepSeek Chat | 启用时为 `thinking.type` 加顶层 `reasoning_effort` |
| Anthropic Messages | 启用时为 `thinking.type` 加 `output_config.effort` |

DeepSeek Responses 仍是标准 Responses 请求，因此使用 `reasoning.effort`，而不是 Chat 特有的双字段形状。提供商适配在写入所选形状前，会去掉互相竞争的 `reasoning`、`thinking`、顶层 effort、`output_config.effort` 和本地模板控制。改变推理语义的 Day-0 贡献，必须为每个声称的协议添加最终编码请求夹具，包括该模型实际暴露的启用、禁用、自适应和 effort 变体。

对模板使用布尔 effort 标志的自定义模型，把公开决策留在 `reasoning_effort`，只在内联 `reasoning.effort_flags` 中映射线名称。一个活跃级别可以故意不映射，表示“省略全部 effort 标志”；其他每个活跃级别必须有唯一标志。内置模型已带此映射，不需要用户 YAML。

控制面板发现比普通运行时分发使用更严的网络边界。云模型 API 必须使用精确的内置 origin。自托管运行时可以有意使用私有或回环地址，但链路本地和元数据端点会被拒绝。重定向和环境代理被禁用，拨号器在发送凭据前会再次检查解析后的地址。

## 完整示例：GPT-6 Astra

GPT-6 Astra 是在已有提供商上加模型，所以实现留在共享目录和现有 OpenAI 协议适配器内。它是上述贡献路径的完整示例，并不声称改动落在模型发布当天。其源数据包是官方
[模型参考](https://developers.openai.com/api/docs/models/gpt-6-astra)、
[最新模型指南](https://developers.openai.com/api/docs/guides/latest-model)
和[发布评测报告](https://openai.com/index/gpt-6-astra/)：

| 契约 | 真相源 |
| --- | --- |
| 身份、限额、模态、能力和展示 | `config/catalog/resources/models/single/openai.yaml` |
| OpenAI 模型 ID、Chat/Responses 可用性、定价和 API 约束 | `config/catalog/resources/providers/openai.yaml` |
| 始终可用的 `low`、`medium`、`high`、`xhigh` 和 `max` 推理 | `config/catalog/resources/reasoning-families.yaml` |
| 精确的厂商发布基准结果 | `config/catalog/resources/evaluations/single/openai.yaml` |
| 最终 Chat 和 Responses 请求形状 | `src/semantic-router/pkg/extproc/provider_request_catalog_contract_test.go` |
| 可运行别名和 mock 提供商凭据 | `e2e/profiles/response-api/values.yaml` |
| 默认 CI 配置成员 | `e2e/profiles/response-api/profile.go` |
| 黑盒模型 ID 和推理投影 | `e2e/testcases/model_catalog_astra.go` |

官方模型页没有命名默认推理 effort，所以目录也不编造一个。它也不暴露禁用模式：省略 effort 让 API 选默认值，而配置校验会拒绝 `use_reasoning: false`。已发布的发布分数标为 `unspecified`，因为来源报告的是支持 effort 上的最大结果，而不是把每个分数归因到一个 effort。

OpenAI 绑定记录：工具需要 Responses；超过 272,000 输入 token 的请求使用已发布的长上下文乘数；`temperature`、`top_p` 和 `top_logprobs` 不受支持（Chat Completions 也不支持 `logprobs`，Responses 不能包含 `message.output_text.logprobs`）。它还把 Chat Completions 收窄到 `low`、`medium`、`high` 和 `xhigh`，因为 `max` 仅限 Responses。这些是可发现的目录约束，而不是静默改写请求：Router 精确投影协议和推理字段，不支持的请求字段仍由 OpenAI 拒绝。

Astra 的工具调用请用 Responses。最小路由配置不需要手写 Model Card 或推理家族：

```yaml
version: v0.3
providers:
  models:
    - name: astra
      catalog: openai/gpt-6-astra
      api_format: responses
      backend_refs:
        - name: primary
          provider: openai
          api_key_env: OPENAI_API_KEY

routing:
  decisions:
    - name: astra_default
      priority: 1
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: astra
          use_reasoning: true
          reasoning_effort: high
```

对没有工具的普通 Chat Completions 工作负载，省略 `api_format: responses`；同一目录绑定会发出顶层 `reasoning_effort`，并在启动时拒绝仅限 Responses 的 effort。Responses 发出 `reasoning.effort`。黑盒 E2E 还证明 Responses 工具定义能在 Router 到提供商的一跳中存活。生成的 Router、CLI、控制面板和网站视图都来自这些编写的资源。

按名称运行该示例：

```bash
make e2e-test-specific \
  E2E_PROFILE=response-api \
  E2E_TESTS=model-catalog-astra
```

该配置使用夹具凭据和本地 mock 提供商；从不调用外部模型 API。测试发送带 `xhigh` 的 Chat、带仅限 Responses 的 `max` 的 Responses，以及一条 Responses `high` 工具请求。然后读取 mock 提供商捕获的请求，检查提供商原生模型 ID、协议特定推理字段、没有竞争协议形状，以及工具是否保留。把这个示例复制到协议或推理契约不同的其他模型时，请保留这次最终一跳断言。

## 内置与自定义用户配置

内置卡片用一个可选的 `catalog` 引用选择。模型 `name` 仍是面向请求的别名，而 `backend_refs[].provider` 是目录 Provider ID：

```yaml
version: v0.3
providers:
  defaults:
    model: production
    reasoning_effort: medium
  models:
    - name: production
      catalog: organization/model
      backend_refs:
        - name: primary
          provider: provider-id
          api_key_env: PROVIDER_API_KEY
```

内置卡片不需要 `routing.modelCards`。有意覆盖时，用规范目录身份作为卡片名：

```yaml
routing:
  modelCards:
    - name: organization/model
      tags: [production, approved]
```

`api_format` 只选择上游线契约；绝不要用它从当前注册表内容推断 Provider。Router 自有 listener 后面的物理模型必须有显式 `backend_refs[].provider`，这样将来新增 Provider 不会改变已有 YAML 的含义。

当提供商绑定声明运维定义的 `deployment_name` 时，其目录 ID 只记录可用性；用户必须显式设置 `providers.models[].provider_model_id`（或该提供商的 `external_model_ids` 条目）。

把同一别名上的多个 `backend_refs` 当作一个 Envoy 池里的同质子副本。HTTP 端点地址、端口和权重可以不同；HTTPS 子副本可以按端口和权重变化，但保持一个 DNS 主机名。Provider ID、协议/模型映射、凭据来源、认证和默认请求头、有效请求路径，以及 DNS/TLS 行为在其他方面必须一致。把异构提供商或凭据拆成不同别名，这样 Router 请求整形不会和 Envoy 选中的端点分叉。配置加载器和 CLI 生成器会拒绝不安全的混合池，并点名不同的语义字段，错误中从不包含凭据值。

内置推理来自该卡片。只有省略 `catalog` 的自定义模型才接受 `providers.models[].reasoning` 块。

自定义卡片也可以声明可选的 `publisher`、`presentation` 和 `distribution` 元数据。这样私有或新发布的模型可以渲染成完整的有效卡片，而不必加仓库资源；这些字段都不存凭据。

对私有模型，省略 `catalog`。其别名就是本地卡片身份；推理和模型关联的评测记录仍可选：

```yaml
providers:
  defaults:
    model: private-reasoner
  models:
    - name: private-reasoner
      provider_model_id: private-awq
      reasoning:
        family: qwen3
      backend_refs:
        - name: lab
          provider: vllm
          endpoint: model-gateway.example:8000/v1

evaluation:
  records:
    - model: private-reasoner
      benchmark: organization/private-eval@1.0.0
      benchmark_profile: published-standard
      reasoning_effort: high
      metrics:
        pass_rate: 0.82

routing:
  modelCards:
    - name: private-reasoner
      context_window_size: 131072
      capabilities: [chat, tools, reasoning]
```

用户编写的记录刻意保持小表面：`model`、`benchmark` 和 `metrics`，外加可选的 `benchmark_profile`、`reasoning_effort`、`source`、`measured_at` 和标量 `metadata`。省略 `benchmark_profile` 以使用已知基准的默认配置；测量来自特定模型 effort 时设置 `reasoning_effort`。目录记录在内部保留更丰富的证据和出处。自定义基准身份仍带命名空间和版本，指标值必须有限。

## 生成和校验

```bash
make model-catalog-generate
make model-catalog-check
make impact ENV=cpu BASE_REF=upstream/main
make check BASE_REF=upstream/main
make verify PROFILE=response-api
```

一起提交编写的资源、内置发行快照、Router embed 和共享公开快照。CLI 包资产是构建时暂存输出，被 Git 忽略，不得提交。当模型或提供商属于其他配置时，使用 `impact` 报告的 E2E 配置，而不是 `response-api`；显式集成域用 `make verify DOMAIN=<domain>`。需要完整本地 PR 基线时跑 `make ci-full BASE_REF=upstream/main`。完整的 Day-0 pull request 应证明：

- 身份稳定，引用有效；
- 每个支持声明都有协议和能力一致性；
- 生成的控制面板提供商/模型卡片和标志回退；
- 生成的网站支持和基准对比行；
- 没有密钥或受限基准数据；
- 缺分数数据，而不是编造的零或占位行。

不要手改生成的 JSON、Go 或内置目录快照。若生成视图不对，修源资源或生成器后再生成。

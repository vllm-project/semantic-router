---
translation:
  source_commit: "a149800"
  source_file: "docs/api/crd-reference.md"
  outdated: true
sidebar_position: 3
title: CRD API 参考
description: vLLM Semantic Router 的 Kubernetes 自定义资源定义 (CRD) API 参考
---

# API 参考

## 软件包

- [vllm.ai/v1alpha1](#vllmaiv1alpha1)

## vllm.ai/v1alpha1 {#vllmaiv1alpha1}

软件包 v1alpha1 包含了 v1alpha1 API 组的 API Schema 定义。

### 资源类型

- [IntelligentPool](#intelligentpool)
- [IntelligentPoolList](#intelligentpoollist)
- [IntelligentRoute](#intelligentroute)
- [IntelligentRouteList](#intelligentroutelist)

#### Decision (决策) {#decision}

Decision 定义了基于规则组合的路由决策。

_出现位置:_

- [IntelligentRouteSpec](#intelligentroutespec)

| 字段 | 描述 | 默认值 | 验证 |
| --- | --- | --- | --- |
| `name` _string_ | Name 是此决策的唯一标识符 |  | MaxLength: 100 <br />MinLength: 1 <br />Required: \{\} <br /> |
| `priority` _integer_ | Priority 定义了此决策的优先级（值越高 = 优先级越高）<br />当策略为 "priority" 时使用 | 0 | Maximum: 1000 <br />Minimum: 0 <br /> |
| `description` _string_ | Description 提供了对此决策的人类可读描述 |  | MaxLength: 500 <br /> |
| `signals` _[SignalCombination](#signalcombination)_ | Signals 定义了信号组合逻辑 |  | Required: \{\} <br /> |
| `modelRefs` _[ModelRef](#modelref) array_ | ModelRefs 定义了此决策的模型引用（目前仅支持一个模型） |  | MaxItems: 1 <br />MinItems: 1 <br />Required: \{\} <br /> |
| `plugins` _[DecisionPlugin](#decisionplugin) array_ | Plugins 定义了应用于此决策的插件 |  | MaxItems: 10 <br /> |

#### DecisionPlugin (决策插件) {#decisionplugin}

DecisionPlugin 定义了决策的插件配置。

_出现位置:_

- [Decision](#decision)

| 字段 | 描述 | 默认值 | 验证 |
| --- | --- | --- | --- |
| `type` _string_ | Type 是插件类型 (semantic-cache, jailbreak, pii, system_prompt, header_mutation) |  | Enum: [semantic-cache jailbreak pii system_prompt header_mutation] <br />Required: \{\} <br /> |
| `configuration` _[RawExtension](https://kubernetes.io/docs/reference/generated/kubernetes-api/v/#rawextension-runtime-pkg)_ | Configuration 是作为原始 JSON 对象的插件特定配置 |  | Schemaless: \{\} <br /> |

#### DomainSignal (领域信号) {#domainsignal}

DomainSignal 定义了用于分类的领域类别。

_出现位置:_

- [Signals](#signals)

| 字段 | 描述 | 默认值 | 验证 |
| --- | --- | --- | --- |
| `name` _string_ | Name 是此领域的唯一标识符 |  | MaxLength: 100 <br />MinLength: 1 <br />Required: \{\} <br /> |
| `description` _string_ | Description 提供了对此领域的人类可读描述 |  | MaxLength: 500 <br /> |

#### EmbeddingSignal (嵌入信号) {#embeddingsignal}

EmbeddingSignal 定义了基于嵌入的信号提取规则。

_出现位置:_

- [Signals](#signals)

| 字段 | 描述 | 默认值 | 验证 |
| --- | --- | --- | --- |
| `name` _string_ | Name 是此信号的唯一标识符 |  | MaxLength: 100 <br />MinLength: 1 <br />Required: \{\} <br /> |
| `threshold` _float_ | Threshold 是匹配的相似度阈值 (0.0-1.0) |  | Maximum: 1 <br />Minimum: 0 <br />Required: \{\} <br /> |
| `candidates` _string array_ | Candidates 是用于语义匹配的候选短语列表 |  | MaxItems: 100 <br />MinItems: 1 <br />Required: \{\} <br /> |
| `aggregationMethod` _string_ | AggregationMethod 定义了如何聚合多个候选相似度 | max | Enum: [mean max any] <br /> |

#### IntelligentPool (智能池) {#intelligentpool}

IntelligentPool 定义了带有配置的模型池。

_出现位置:_

- [IntelligentPoolList](#intelligentpoollist)

| 字段 | 描述 | 默认值 | 验证 |
| --- | --- | --- | --- |
| `apiVersion` _string_ | `vllm.ai/v1alpha1` | | |
| `kind` _string_ | `IntelligentPool` | | |
| `metadata` _[ObjectMeta](https://kubernetes.io/docs/reference/generated/kubernetes-api/v/#objectmeta-v1-meta)_ | 有关 `metadata` 的字段，请参阅 Kubernetes API 文档。 |  |  |
| `spec` _[IntelligentPoolSpec](#intelligentpoolspec)_ |  |  |  |
| `status` _[IntelligentPoolStatus](#intelligentpoolstatus)_ |  |  |  |

#### IntelligentPoolList (智能池列表) {#intelligentpoollist}

IntelligentPoolList 包含 IntelligentPool 列表。

| 字段 | 描述 | 默认值 | 验证 |
| --- | --- | --- | --- |
| `apiVersion` _string_ | `vllm.ai/v1alpha1` | | |
| `kind` _string_ | `IntelligentPoolList` | | |
| `metadata` _[ListMeta](https://kubernetes.io/docs/reference/generated/kubernetes-api/v/#listmeta-v1-meta)_ | 有关 `metadata` 的字段，请参阅 Kubernetes API 文档。 |  |  |
| `items` _[IntelligentPool](#intelligentpool) array_ |  |  |  |

#### IntelligentPoolSpec (智能池规范) {#intelligentpoolspec}

IntelligentPoolSpec 定义了 IntelligentPool 的期望状态。

_出现位置:_

- [IntelligentPool](#intelligentpool)

| 字段 | 描述 | 默认值 | 验证 |
| --- | --- | --- | --- |
| `defaultModel` _string_ | DefaultModel 指定未选择特定模型时使用的默认模型 |  | MaxLength: 100 <br />MinLength: 1 <br />Required: \{\} <br /> |
| `models` _[ModelConfig](#modelconfig) array_ | Models 定义了此池中可用模型的列表 |  | MaxItems: 100 <br />MinItems: 1 <br />Required: \{\} <br /> |

#### IntelligentPoolStatus (智能池状态) {#intelligentpoolstatus}

IntelligentPoolStatus 定义了观察到的 IntelligentPool 状态。

_出现位置:_

- [IntelligentPool](#intelligentpool)

| 字段 | 描述 | 默认值 | 验证 |
| --- | --- | --- | --- |
| `conditions` _[Condition](https://kubernetes.io/docs/reference/generated/kubernetes-api/v/#condition-v1-meta) array_ | Conditions 代表 IntelligentPool 状态的最新可用观察结果 |  |  |
| `observedGeneration` _integer_ | ObservedGeneration 反映了最近观察到的 IntelligentPool 的代 (generation) |  |  |
| `modelCount` _integer_ | ModelCount 表示池中模型的数量 |  |  |

#### IntelligentRoute (智能路由) {#intelligentroute}

IntelligentRoute 定义了智能路由规则和决策。

_出现位置:_

- [IntelligentRouteList](#intelligentroutelist)

| 字段 | 描述 | 默认值 | 验证 |
| --- | --- | --- | --- |
| `apiVersion` _string_ | `vllm.ai/v1alpha1` | | |
| `kind` _string_ | `IntelligentRoute` | | |
| `metadata` _[ObjectMeta](https://kubernetes.io/docs/reference/generated/kubernetes-api/v/#objectmeta-v1-meta)_ | 有关 `metadata` 的字段，请参阅 Kubernetes API 文档。 |  |  |
| `spec` _[IntelligentRouteSpec](#intelligentroutespec)_ |  |  |  |
| `status` _[IntelligentRouteStatus](#intelligentroutestatus)_ |  |  |  |

#### IntelligentRouteList (智能路由列表) {#intelligentroutelist}

IntelligentRouteList 包含 IntelligentRoute 列表。

| 字段 | 描述 | 默认值 | 验证 |
| --- | --- | --- | --- |
| `apiVersion` _string_ | `vllm.ai/v1alpha1` | | |
| `kind` _string_ | `IntelligentRouteList` | | |
| `metadata` _[ListMeta](https://kubernetes.io/docs/reference/generated/kubernetes-api/v/#listmeta-v1-meta)_ | 有关 `metadata` 的字段，请参阅 Kubernetes API 文档。 |  |  |
| `items` _[IntelligentRoute](#intelligentroute) array_ |  |  |  |

#### IntelligentRouteSpec (智能路由规范) {#intelligentroutespec}

IntelligentRouteSpec 定义了 IntelligentRoute 的期望状态。

_出现位置:_

- [IntelligentRoute](#intelligentroute)

| 字段 | 描述 | 默认值 | 验证 |
| --- | --- | --- | --- |
| `signals` _[Signals](#signals)_ | Signals 定义了用于路由决策的信号提取规则 |  |  |
| `decisions` _[Decision](#decision) array_ | Decisions 定义了基于信号组合的路由决策 |  | MaxItems: 100 <br />MinItems: 1 <br />Required: \{\} <br /> |

#### IntelligentRouteStatus (智能路由状态) {#intelligentroutestatus}

IntelligentRouteStatus 定义了观察到的 IntelligentRoute 状态。

_出现位置:_

- [IntelligentRoute](#intelligentroute)

| 字段 | 描述 | 默认值 | 验证 |
| --- | --- | --- | --- |
| `conditions` _[Condition](https://kubernetes.io/docs/reference/generated/kubernetes-api/v/#condition-v1-meta) array_ | Conditions 代表 IntelligentRoute 状态的最新可用观察结果 |  |  |
| `observedGeneration` _integer_ | ObservedGeneration 反映了最近观察到的 IntelligentRoute 的代 (generation) |  |  |
| `statistics` _[RouteStatistics](#routestatistics)_ | Statistics 提供了有关已配置决策和信号的统计信息 |  |  |

#### KeywordSignal (关键词信号) {#keywordsignal}

KeywordSignal 定义了基于关键词的信号提取规则。

_出现位置:_

- [Signals](#signals)

| 字段 | 描述 | 默认值 | 验证 |
| --- | --- | --- | --- |
| `name` _string_ | Name 是此规则的唯一标识符（也用作类别名称） |  | MaxLength: 100 <br />MinLength: 1 <br />Required: \{\} <br /> |
| `operator` _string_ | Operator 定义了关键词的逻辑运算符 (AND/OR) |  | Enum: [AND OR] <br />Required: \{\} <br /> |
| `keywords` _string array_ | Keywords 是要匹配的关键词列表 |  | MaxItems: 100 <br />MinItems: 1 <br />Required: \{\} <br /> |
| `caseSensitive` _boolean_ | CaseSensitive 指定关键词匹配是否区分大小写 | false |  |

#### LoRAConfig (LoRA 配置) {#loraconfig}

LoRAConfig 定义了 LoRA 适配器配置。

_出现位置:_

- [ModelConfig](#modelconfig)

| 字段 | 描述 | 默认值 | 验证 |
| --- | --- | --- | --- |
| `name` _string_ | Name 是此 LoRA 适配器的唯一标识符 |  | MaxLength: 100 <br />MinLength: 1 <br />Required: \{\} <br /> |
| `description` _string_ | Description 提供了对此 LoRA 适配器的人类可读描述 |  | MaxLength: 500 <br /> |

#### ModelConfig (模型配置) {#modelconfig}

ModelConfig 定义了单个模型的配置。

_出现位置:_

- [IntelligentPoolSpec](#intelligentpoolspec)

| 字段 | 描述 | 默认值 | 验证 |
| --- | --- | --- | --- |
| `name` _string_ | Name 是此模型的唯一标识符 |  | MaxLength: 100 <br />MinLength: 1 <br />Required: \{\} <br /> |
| `reasoningFamily` _string_ | ReasoningFamily 指定了推理语法家族（例如 "qwen3", "deepseek"）<br />必须在全局静态配置的 ReasoningFamilies 中定义 |  | MaxLength: 50 <br /> |
| `pricing` _[ModelPricing](#modelpricing)_ | Pricing 定义了此模型的成本结构 |  |  |
| `loras` _[LoRAConfig](#loraconfig) array_ | LoRAs 定义了此模型可用的 LoRA 适配器列表 |  | MaxItems: 50 <br /> |

#### ModelPricing (模型定价) {#modelpricing}

ModelPricing 定义了模型的定价结构。

_出现位置:_

- [ModelConfig](#modelconfig)

| 字段 | 描述 | 默认值 | 验证 |
| --- | --- | --- | --- |
| `inputTokenPrice` _float_ | InputTokenPrice 是每个输入 token 的成本 |  | Minimum: 0 <br /> |
| `outputTokenPrice` _float_ | OutputTokenPrice 是每个输出 token 的成本 |  | Minimum: 0 <br /> |

#### ModelRef (模型引用) {#modelref}

ModelRef 定义了不带评分的模型引用。

_出现位置:_

- [Decision](#decision)

| 字段 | 描述 | 默认值 | 验证 |
| --- | --- | --- | --- |
| `model` _string_ | Model 是模型名称（必须存在于 IntelligentPool 中） |  | MaxLength: 100 <br />MinLength: 1 <br />Required: \{\} <br /> |
| `loraName` _string_ | LoRAName 是要使用的 LoRA 适配器名称（必须存在于模型的 LoRAs 中） |  | MaxLength: 100 <br /> |
| `useReasoning` _boolean_ | UseReasoning 指定是否为此模型启用推理模式 | false |  |
| `reasoningDescription` _string_ | ReasoningDescription 提供了何时使用推理的上下文 |  | MaxLength: 500 <br /> |
| `reasoningEffort` _string_ | ReasoningEffort 定义了推理努力等级 (low/medium/high) |  | Enum: [low medium high] <br /> |

#### RouteStatistics (路由统计) {#routestatistics}

RouteStatistics 提供了有关 IntelligentRoute 配置的统计信息。

_出现位置:_

- [IntelligentRouteStatus](#intelligentroutestatus)

| 字段 | 描述 | 默认值 | 验证 |
| --- | --- | --- | --- |
| `decisions` _integer_ | Decisions 表示决策数量 |  |  |
| `keywords` _integer_ | Keywords 表示关键词信号的数量 |  |  |
| `embeddings` _integer_ | Embeddings 表示嵌入信号的数量 |  |  |
| `domains` _integer_ | Domains 表示领域信号的数量 |  |  |

#### SignalCombination (信号组合) {#signalcombination}

SignalCombination 定义了如何组合多个信号。

_出现位置:_

- [Decision](#decision)

| 字段 | 描述 | 默认值 | 验证 |
| --- | --- | --- | --- |
| `operator` _string_ | Operator 定义了组合条件的逻辑运算符 (AND/OR) |  | Enum: [AND OR] <br />Required: \{\} <br /> |
| `conditions` _[SignalCondition](#signalcondition) array_ | Conditions 定义了信号条件列表 |  | MaxItems: 50 <br />MinItems: 1 <br />Required: \{\} <br /> |

#### SignalCondition (信号条件) {#signalcondition}

SignalCondition 定义了单个信号条件。

_出现位置:_

- [SignalCombination](#signalcombination)

| 字段 | 描述 | 默认值 | 验证 |
| --- | --- | --- | --- |
| `type` _string_ | Type 定义了信号类型 (keyword/embedding/domain) |  | Enum: [keyword embedding domain] <br />Required: \{\} <br /> |
| `name` _string_ | Name 是要引用的信号名称 |  | MaxLength: 100 <br />MinLength: 1 <br />Required: \{\} <br /> |

#### Signals (信号) {#signals}

Signals 定义了信号提取规则。

_出现位置:_

- [IntelligentRouteSpec](#intelligentroutespec)

| 字段 | 描述 | 默认值 | 验证 |
| --- | --- | --- | --- |
| `keywords` _[KeywordSignal](#keywordsignal) array_ | Keywords 定义了基于关键词的信号提取规则 |  | MaxItems: 100 <br /> |
| `embeddings` _[EmbeddingSignal](#embeddingsignal) array_ | Embeddings 定义了基于嵌入的信号提取规则 |  | MaxItems: 100 <br /> |
| `domains` _[DomainSignal](#domainsignal) array_ | Domains 定义了用于分类的 MMLU 领域类别 |  | MaxItems: 14 <br /> |

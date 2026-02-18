---
sidebar_position: 3
title: CRD API Reference
description: Kubernetes Custom Resource Definitions (CRDs) API reference for vLLM Semantic Router
---

# API Reference

## Packages

- [vllm.ai/v1alpha1](#vllmaiv1alpha1)

## vllm.ai/v1alpha1

Package v1alpha1 contains API Schema definitions for the v1alpha1 API group

### Resource Types

- [IntelligentPool](#intelligentpool)
- [IntelligentPoolList](#intelligentpoollist)
- [IntelligentRoute](#intelligentroute)
- [IntelligentRouteList](#intelligentroutelist)

#### ContextRule

ContextRule defines a rule for context-based (token count) classification

_Appears in:_

- [Signals](#signals)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `name` _string_ | Name is the signal name (e.g., "high_token_count") |  | MaxLength: 100 <br />MinLength: 1 <br />Required: \{\} <br /> |
| `minTokens` _string_ | MinTokens is the minimum token count (supports K/M suffixes) |  | Pattern: `^[0-9]+(\.[0-9]+)?[KMkm]?$` <br />Required: \{\} <br /> |
| `maxTokens` _string_ | MaxTokens is the maximum token count (supports K/M suffixes) |  | Pattern: `^[0-9]+(\.[0-9]+)?[KMkm]?$` <br />Required: \{\} <br /> |
| `description` _string_ | Description provides human-readable explanation |  | MaxLength: 500 <br /> |

#### Decision

Decision defines a routing decision based on rule combinations

_Appears in:_

- [IntelligentRouteSpec](#intelligentroutespec)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `name` _string_ | Name is the unique identifier for this decision |  | MaxLength: 100 <br />MinLength: 1 <br />Required: \{\} <br /> |
| `priority` _integer_ | Priority defines the priority of this decision (higher values = higher priority)<br />Used when strategy is "priority" | 0 | Maximum: 1000 <br />Minimum: 0 <br /> |
| `description` _string_ | Description provides a human-readable description of this decision |  | MaxLength: 500 <br /> |
| `signals` _[SignalCombination](#signalcombination)_ | Signals defines the signal combination logic |  | Required: \{\} <br /> |
| `modelRefs` _[ModelRef](#modelref) array_ | ModelRefs defines the model references for this decision (currently only one model is supported) |  | MaxItems: 1 <br />MinItems: 1 <br />Required: \{\} <br /> |
| `plugins` _[DecisionPlugin](#decisionplugin) array_ | Plugins defines the plugins to apply for this decision |  | MaxItems: 10 <br /> |

#### DecisionPlugin

DecisionPlugin defines a plugin configuration for a decision

_Appears in:_

- [Decision](#decision)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `type` _string_ | Type is the plugin type (semantic-cache, jailbreak, pii, system_prompt, header_mutation) |  | Enum: [semantic-cache jailbreak pii system_prompt header_mutation] <br />Required: \{\} <br /> |
| `configuration` _[RawExtension](https://kubernetes.io/docs/reference/generated/kubernetes-api/v/#rawextension-runtime-pkg)_ | Configuration is the plugin-specific configuration as a raw JSON object |  | Schemaless: \{\} <br /> |

#### DomainSignal

DomainSignal defines a domain category for classification

_Appears in:_

- [Signals](#signals)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `name` _string_ | Name is the unique identifier for this domain |  | MaxLength: 100 <br />MinLength: 1 <br />Required: \{\} <br /> |
| `description` _string_ | Description provides a human-readable description of this domain |  | MaxLength: 500 <br /> |

#### EmbeddingSignal

EmbeddingSignal defines an embedding-based signal extraction rule

_Appears in:_

- [Signals](#signals)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `name` _string_ | Name is the unique identifier for this signal |  | MaxLength: 100 <br />MinLength: 1 <br />Required: \{\} <br /> |
| `threshold` _float_ | Threshold is the similarity threshold for matching (0.0-1.0) |  | Maximum: 1 <br />Minimum: 0 <br />Required: \{\} <br /> |
| `candidates` _string array_ | Candidates is the list of candidate phrases for semantic matching |  | MaxItems: 100 <br />MinItems: 1 <br />Required: \{\} <br /> |
| `aggregationMethod` _string_ | AggregationMethod defines how to aggregate multiple candidate similarities | max | Enum: [mean max any] <br /> |

#### IntelligentPool

IntelligentPool defines a pool of models with their configurations

_Appears in:_

- [IntelligentPoolList](#intelligentpoollist)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `apiVersion` _string_ | `vllm.ai/v1alpha1` | | |
| `kind` _string_ | `IntelligentPool` | | |
| `metadata` _[ObjectMeta](https://kubernetes.io/docs/reference/generated/kubernetes-api/v/#objectmeta-v1-meta)_ | Refer to Kubernetes API documentation for fields of `metadata`. |  |  |
| `spec` _[IntelligentPoolSpec](#intelligentpoolspec)_ |  |  |  |
| `status` _[IntelligentPoolStatus](#intelligentpoolstatus)_ |  |  |  |

#### IntelligentPoolList

IntelligentPoolList contains a list of IntelligentPool

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `apiVersion` _string_ | `vllm.ai/v1alpha1` | | |
| `kind` _string_ | `IntelligentPoolList` | | |
| `metadata` _[ListMeta](https://kubernetes.io/docs/reference/generated/kubernetes-api/v/#listmeta-v1-meta)_ | Refer to Kubernetes API documentation for fields of `metadata`. |  |  |
| `items` _[IntelligentPool](#intelligentpool) array_ |  |  |  |

#### IntelligentPoolSpec

IntelligentPoolSpec defines the desired state of IntelligentPool

_Appears in:_

- [IntelligentPool](#intelligentpool)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `defaultModel` _string_ | DefaultModel specifies the default model to use when no specific model is selected |  | MaxLength: 100 <br />MinLength: 1 <br />Required: \{\} <br /> |
| `models` _[ModelConfig](#modelconfig) array_ | Models defines the list of available models in this pool |  | MaxItems: 100 <br />MinItems: 1 <br />Required: \{\} <br /> |

#### IntelligentPoolStatus

IntelligentPoolStatus defines the observed state of IntelligentPool

_Appears in:_

- [IntelligentPool](#intelligentpool)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `conditions` _[Condition](https://kubernetes.io/docs/reference/generated/kubernetes-api/v/#condition-v1-meta) array_ | Conditions represent the latest available observations of the IntelligentPool's state |  |  |
| `observedGeneration` _integer_ | ObservedGeneration reflects the generation of the most recently observed IntelligentPool |  |  |
| `modelCount` _integer_ | ModelCount indicates the number of models in the pool |  |  |

#### IntelligentRoute

IntelligentRoute defines intelligent routing rules and decisions

_Appears in:_

- [IntelligentRouteList](#intelligentroutelist)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `apiVersion` _string_ | `vllm.ai/v1alpha1` | | |
| `kind` _string_ | `IntelligentRoute` | | |
| `metadata` _[ObjectMeta](https://kubernetes.io/docs/reference/generated/kubernetes-api/v/#objectmeta-v1-meta)_ | Refer to Kubernetes API documentation for fields of `metadata`. |  |  |
| `spec` _[IntelligentRouteSpec](#intelligentroutespec)_ |  |  |  |
| `status` _[IntelligentRouteStatus](#intelligentroutestatus)_ |  |  |  |

#### IntelligentRouteList

IntelligentRouteList contains a list of IntelligentRoute

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `apiVersion` _string_ | `vllm.ai/v1alpha1` | | |
| `kind` _string_ | `IntelligentRouteList` | | |
| `metadata` _[ListMeta](https://kubernetes.io/docs/reference/generated/kubernetes-api/v/#listmeta-v1-meta)_ | Refer to Kubernetes API documentation for fields of `metadata`. |  |  |
| `items` _[IntelligentRoute](#intelligentroute) array_ |  |  |  |

#### IntelligentRouteSpec

IntelligentRouteSpec defines the desired state of IntelligentRoute

_Appears in:_

- [IntelligentRoute](#intelligentroute)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `signals` _[Signals](#signals)_ | Signals defines signal extraction rules for routing decisions |  |  |
| `decisions` _[Decision](#decision) array_ | Decisions defines the routing decisions based on signal combinations |  | MaxItems: 100 <br />MinItems: 1 <br />Required: \{\} <br /> |

#### IntelligentRouteStatus

IntelligentRouteStatus defines the observed state of IntelligentRoute

_Appears in:_

- [IntelligentRoute](#intelligentroute)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `conditions` _[Condition](https://kubernetes.io/docs/reference/generated/kubernetes-api/v/#condition-v1-meta) array_ | Conditions represent the latest available observations of the IntelligentRoute's state |  |  |
| `observedGeneration` _integer_ | ObservedGeneration reflects the generation of the most recently observed IntelligentRoute |  |  |
| `statistics` _[RouteStatistics](#routestatistics)_ | Statistics provides statistics about configured decisions and signals |  |  |

#### KeywordSignal

KeywordSignal defines a keyword-based signal extraction rule

_Appears in:_

- [Signals](#signals)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `name` _string_ | Name is the unique identifier for this rule (also used as category name) |  | MaxLength: 100 <br />MinLength: 1 <br />Required: \{\} <br /> |
| `operator` _string_ | Operator defines the logical operator for keywords (AND/OR) |  | Enum: [AND OR] <br />Required: \{\} <br /> |
| `keywords` _string array_ | Keywords is the list of keywords to match |  | MaxItems: 100 <br />MinItems: 1 <br />Required: \{\} <br /> |
| `caseSensitive` _boolean_ | CaseSensitive specifies whether keyword matching is case-sensitive | false |  |

#### LoRAConfig

LoRAConfig defines a LoRA adapter configuration

_Appears in:_

- [ModelConfig](#modelconfig)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `name` _string_ | Name is the unique identifier for this LoRA adapter |  | MaxLength: 100 <br />MinLength: 1 <br />Required: \{\} <br /> |
| `description` _string_ | Description provides a human-readable description of this LoRA adapter |  | MaxLength: 500 <br /> |

#### ModelConfig

ModelConfig defines the configuration for a single model

_Appears in:_

- [IntelligentPoolSpec](#intelligentpoolspec)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `name` _string_ | Name is the unique identifier for this model |  | MaxLength: 100 <br />MinLength: 1 <br />Required: \{\} <br /> |
| `reasoningFamily` _string_ | ReasoningFamily specifies the reasoning syntax family (e.g., "qwen3", "deepseek")<br />Must be defined in the global static configuration's ReasoningFamilies |  | MaxLength: 50 <br /> |
| `pricing` _[ModelPricing](#modelpricing)_ | Pricing defines the cost structure for this model |  |  |
| `loras` _[LoRAConfig](#loraconfig) array_ | LoRAs defines the list of LoRA adapters available for this model |  | MaxItems: 50 <br /> |

#### ModelPricing

ModelPricing defines the pricing structure for a model

_Appears in:_

- [ModelConfig](#modelconfig)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `inputTokenPrice` _float_ | InputTokenPrice is the cost per input token |  | Minimum: 0 <br /> |
| `outputTokenPrice` _float_ | OutputTokenPrice is the cost per output token |  | Minimum: 0 <br /> |

#### ModelRef

ModelRef defines a model reference without score

_Appears in:_

- [Decision](#decision)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `model` _string_ | Model is the name of the model (must exist in IntelligentPool) |  | MaxLength: 100 <br />MinLength: 1 <br />Required: \{\} <br /> |
| `loraName` _string_ | LoRAName is the name of the LoRA adapter to use (must exist in the model's LoRAs) |  | MaxLength: 100 <br /> |
| `useReasoning` _boolean_ | UseReasoning specifies whether to enable reasoning mode for this model | false |  |
| `reasoningDescription` _string_ | ReasoningDescription provides context for when to use reasoning |  | MaxLength: 500 <br /> |
| `reasoningEffort` _string_ | ReasoningEffort defines the reasoning effort level (low/medium/high) |  | Enum: [low medium high] <br /> |

#### RouteStatistics

RouteStatistics provides statistics about the IntelligentRoute configuration

_Appears in:_

- [IntelligentRouteStatus](#intelligentroutestatus)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `decisions` _integer_ | Decisions indicates the number of decisions |  |  |
| `keywords` _integer_ | Keywords indicates the number of keyword signals |  |  |
| `embeddings` _integer_ | Embeddings indicates the number of embedding signals |  |  |
| `domains` _integer_ | Domains indicates the number of domain signals |  |  |

#### SignalCombination

SignalCombination defines how to combine multiple signals

_Appears in:_

- [Decision](#decision)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `operator` _string_ | Operator defines the logical operator for combining conditions (AND/OR) |  | Enum: [AND OR] <br />Required: \{\} <br /> |
| `conditions` _[SignalCondition](#signalcondition) array_ | Conditions defines the list of signal conditions |  | MaxItems: 50 <br />MinItems: 1 <br />Required: \{\} <br /> |

#### SignalCondition

SignalCondition defines a single signal condition

_Appears in:_

- [SignalCombination](#signalcombination)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `type` _string_ | Type defines the type of signal (keyword/embedding/domain/fact_check/context) |  | Enum: [keyword embedding domain fact_check context] <br />Required: \{\} <br /> |
| `name` _string_ | Name is the name of the signal to reference |  | MaxLength: 100 <br />MinLength: 1 <br />Required: \{\} <br /> |

#### Signals

Signals defines signal extraction rules

_Appears in:_

- [IntelligentRouteSpec](#intelligentroutespec)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `keywords` _[KeywordSignal](#keywordsignal) array_ | Keywords defines keyword-based signal extraction rules |  | MaxItems: 100 <br /> |
| `embeddings` _[EmbeddingSignal](#embeddingsignal) array_ | Embeddings defines embedding-based signal extraction rules |  | MaxItems: 100 <br /> |
| `domains` _[DomainSignal](#domainsignal) array_ | Domains defines MMLU domain categories for classification |  | MaxItems: 14 <br /> |
| `contextRules` _[ContextRule](#contextrule) array_ | ContextRules defines context (token count) rules for signal classification |  | MaxItems: 20 <br /> |

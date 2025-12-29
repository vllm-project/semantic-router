---
translation:
  source_commit: "bac2743"
  source_file: "docs/overview/mom-model-family.md"
  outdated: false
---

# 什么是 MoM 模型家族？

**MoM (Mixture of Models) Model Family** 是一个精心挑选的专用轻量级模型集合，专为智能路由、内容安全和语义理解而设计。这些模型为 Semantic Router 的核心能力提供动力，实现快速、准确和隐私保护的 AI 操作。

## 概览

MoM 家族由专门构建的模型组成，用于处理路由管道中的特定任务：

- **分类模型**：Domain 检测、PII 识别、Jailbreak 检测
- **Embedding 模型**：语义相似度、缓存、检索
- **安全模型**：Hallucination 检测、内容审核
- **Feedback 模型**：用户意图理解、对话分析

所有 MoM 模型都具有以下特点：

- **轻量级**：33M-600M 参数，实现快速推理
- **专用**：针对特定路由任务进行微调
- **高效**：许多模型使用 LoRA 适配器，占用内存极小
- **开源**：可在 HuggingFace 上获取，以实现透明度和自定义

## 模型类别

### 1. 分类模型

#### 领域/意图分类器 (Domain/Intent Classifier)

- **模型 ID**: `models/mom-domain-classifier`
- **HuggingFace**: `LLM-Semantic-Router/lora_intent_classifier_bert-base-uncased_model`
- **用途**：将用户查询分类为 14 个 MMLU 类别（数学、科学、历史等）
- **架构**：BERT-base (110M) + LoRA 适配器
- **用例**：将查询路由到特定领域的模型或专家

#### PII 检测器 (PII Detector)

- **模型 ID**: `models/mom-pii-classifier`
- **HuggingFace**: `LLM-Semantic-Router/lora_pii_detector_bert-base-uncased_model`
- **用途**：检测 35 种类型的个人身份信息
- **架构**：BERT-base (110M) + LoRA 适配器
- **用例**：隐私保护、合规性、数据脱敏

#### Jailbreak Detector

- **模型 ID**: `models/mom-jailbreak-classifier`
- **HuggingFace**: `LLM-Semantic-Router/lora_jailbreak_classifier_bert-base-uncased_model`
- **用途**：检测 prompt injection 和 jailbreak 尝试
- **架构**：BERT-base (110M) + LoRA 适配器
- **用例**：内容安全、prompt 安全

#### 反馈检测器 (Feedback Detector)

- **模型 ID**: `models/mom-feedback-detector`
- **HuggingFace**: `llm-semantic-router/feedback-detector`
- **用途**：将用户反馈分类为 4 种类型（满意、需要澄清、错误答案、想要不同的答案）
- **架构**：ModernBERT-base (149M)
- **用例**：自适应路由、对话改进

### 2. 嵌入模型

#### Embedding Pro (高质量)

- **模型 ID**: `models/mom-embedding-pro`
- **HuggingFace**: `Qwen/Qwen3-Embedding-0.6B`
- **用途**：支持 32K 上下文的高质量嵌入
- **架构**：Qwen3 (600M 参数)
- **嵌入维度**：1024
- **用例**：长上下文语义搜索、高精度缓存

#### Embedding Flash (平衡)

- **模型 ID**: `models/mom-embedding-flash`
- **HuggingFace**: `google/embeddinggemma-300m`
- **用途**：支持 Matryoshka (套娃) 的快速嵌入
- **架构**：Gemma (300M 参数)
- **嵌入维度**：768 (支持通过 Matryoshka 使用 512/256/128)
- **用例**：平衡的速度/质量、多语言支持

#### Embedding Light (快速)

- **模型 ID**: `models/mom-embedding-light`
- **HuggingFace**: `sentence-transformers/all-MiniLM-L12-v2`
- **用途**：轻量级语义相似度
- **架构**：MiniLM (33M 参数)
- **嵌入维度**：384
- **用例**：快速语义缓存、低延迟检索

### 3. Hallucination 检测模型

#### Halugate Sentinel

- **模型 ID**: `models/mom-halugate-sentinel`
- **HuggingFace**: `LLM-Semantic-Router/halugate-sentinel`
- **用途**：第一阶段 hallucination 筛查
- **架构**：BERT-base (110M)
- **用例**：快速 hallucination 检测、预过滤

#### Halugate Detector

- **模型 ID**: `models/mom-halugate-detector`
- **HuggingFace**: `KRLabsOrg/lettucedect-base-modernbert-en-v1`
- **用途**：精准幻觉验证
- **架构**：ModernBERT-base (149M)
- **上下文长度**：8192 Tokens
- **用例**：事实准确性验证、基础检查

#### Halugate Explainer

- **模型 ID**: `models/mom-halugate-explainer`
- **HuggingFace**: `tasksource/ModernBERT-base-nli`
- **用途**：通过 NLI 解释幻觉推理
- **架构**：ModernBERT-base (149M)
- **类别**：3 (蕴含/中立/矛盾)
- **用例**：可解释 AI、幻觉分析

## 模型选择指南

### 按用例

| 用例 | 推荐模型 | 原因 |
|----------|------------------|-----|
| 领域路由 | mom-domain-classifier | 14 个 MMLU 类别，LoRA 高效 |
| 隐私保护 | mom-pii-classifier | 35 种 PII 类型，Token 级检测 |
| 内容安全 | mom-jailbreak-classifier | 提示词注入检测 |
| 语义缓存 | mom-embedding-light | 快速，384 维，低延迟 |
| 长上下文搜索 | mom-embedding-pro | 32K 上下文，1024 维 |
| 幻觉检查 | mom-halugate-detector | ModernBERT，8K 上下文 |
| 用户反馈 | mom-feedback-detector | 4 种反馈类型，ModernBERT |

### 按性能要求

| 要求 | 模型层级 | 示例 |
|-------------|-----------|----------|
| 超快 (&lt;10ms) | Light | mom-embedding-light, mom-jailbreak-classifier |
| 平衡 (10-50ms) | Flash | mom-embedding-flash, mom-domain-classifier |
| 高质量 (50-200ms) | Pro | mom-embedding-pro, mom-halugate-detector |

## 配置

### 在路由中使用 MoM 模型

MoM 模型在 `router-defaults.yaml` 中预先配置：

```yaml
# Domain classification
classifier:
  category_model:
    model_id: "models/mom-domain-classifier"
    threshold: 0.6
    use_cpu: true

# PII detection
classifier:
  pii_model:
    model_id: "models/mom-pii-classifier"
    threshold: 0.9
    use_cpu: true

# Jailbreak protection
prompt_guard:
  model_id: "models/mom-jailbreak-classifier"
  threshold: 0.7
  use_cpu: true
```

### 自定义模型注册表

在您的 `config.yaml` 中覆盖默认注册表：

```yaml
mom_registry:
  "models/mom-domain-classifier": "your-org/custom-domain-classifier"
  "models/mom-pii-classifier": "your-org/custom-pii-detector"
  "models/mom-embedding-pro": "your-org/custom-embeddings"
```

## 模型架构

### 基于 LoRA 的模型

许多 MoM 模型使用 LoRA (低秩适应) 以提高效率：

- **基础模型**: BERT-base-uncased (110M 参数)
- **LoRA 适配器**: 每项任务 &lt;1M 参数
- **内存占用**: ~440MB 基础 + ~4MB 每个适配器
- **推理速度**: 与基础模型相同 (CPU 上 ~10-20ms)

### ModernBERT 模型

较新的模型使用 ModernBERT 以获得更好的性能：

- **架构**: ModernBERT-base (149M 参数)
- **上下文长度**: 8192 Tokens (对比 BERT 的 512)
- **性能**: 在长上下文任务上具有更好的准确性
- **用例**: 幻觉检测、反馈分类

## 下一步

- **[信号驱动决策](./signal-driven-decisions.md)** - 了解 MoM 模型如何驱动路由决策
- **[领域路由](../tutorials/intelligent-route/domain-routing.md)** - 使用 mom-domain-classifier 进行路由
- **[PII 检测](../tutorials/content-safety/pii-detection.md)** - 配置 mom-pii-classifier
- **[语义缓存](../tutorials/semantic-cache/in-memory-cache.md)** - 使用 MoM 嵌入模型

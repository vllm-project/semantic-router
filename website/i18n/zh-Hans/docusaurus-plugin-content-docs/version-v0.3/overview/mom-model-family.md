---
translation:
  source_commit: "043cee97"
  source_file: "docs/overview/mom-model-family.md"
  outdated: false
sidebar_position: 5
---

# 什么是 MoM 模型族？

**MoM（Mixture of Models）模型族**是一组精选的、轻量且任务专精的模型，用于智能路由、内容安全与语义理解。它们为语义路由器的核心能力提供算力，实现快速、准确且注重隐私的 AI 操作。

## 概览

MoM 家族包含为路由流水线中特定任务打造的模型：

- **分类模型**：领域检测、PII 识别、越狱检测
- **嵌入模型**：语义相似度、缓存、检索
- **安全模型**：幻觉检测、内容审核
- **反馈模型**：用户意图理解、对话分析

所有 MoM 模型共同特点：

- **轻量**：约 3300 万–6 亿参数，推理快
- **专精**：针对具体路由任务微调
- **高效**：多数采用 LoRA 适配器，显存占用小
- **开源**：在 HuggingFace 上公开，便于审计与定制

## 模型类别

### 1. 分类模型

#### 领域/意图分类器

- **Model ID**：`models/mmbert32k-intent-classifier-merged`
- **HuggingFace**：`llm-semantic-router/mmbert32k-intent-classifier-merged`
- **用途**：将用户查询归入 14 个 MMLU 类别（数学、科学、历史等）
- **架构**：mmBERT-32K 合并分类器（307M）
- **场景**：按领域路由到专用模型或专家

#### PII 检测器

- **Model ID**：`models/mmbert32k-pii-detector-merged`
- **HuggingFace**：`llm-semantic-router/mmbert32k-pii-detector-merged`
- **用途**：检测 17 种 PII 实体类型，对应 35 个 BIO 标签
- **架构**：mmBERT-32K 合并 token 分类器（307M）
- **场景**：隐私保护、合规、脱敏

#### 越狱检测器

- **Model ID**：`models/mmbert32k-jailbreak-detector-merged`
- **HuggingFace**：`llm-semantic-router/mmbert32k-jailbreak-detector-merged`
- **用途**：检测提示注入与越狱企图
- **架构**：mmBERT-32K 合并分类器（307M）
- **场景**：内容安全、提示安全

#### 反馈检测器

- **Model ID**：`models/mmbert32k-feedback-detector-merged`
- **HuggingFace**：`llm-semantic-router/mmbert32k-feedback-detector-merged`
- **用途**：将用户反馈分为 4 类（满意、需澄清、答案错误、希望换种说法）
- **架构**：mmBERT-32K 合并分类器（307M）
- **场景**：自适应路由、对话改进

### 2. 嵌入模型

#### Embedding Pro（高质量）

- **Model ID**：`models/mom-embedding-pro`
- **HuggingFace**：`Qwen/Qwen3-Embedding-0.6B`
- **用途**：高质量嵌入，支持 32K 上下文
- **架构**：Qwen3（6 亿参数）
- **嵌入维度**：1024
- **场景**：长上下文语义搜索、高精度缓存

#### Embedding Flash（均衡）

- **Model ID**：`models/mom-embedding-flash`
- **HuggingFace**：`google/embeddinggemma-300m`
- **用途**：快速嵌入，支持 Matryoshka
- **架构**：Gemma（3 亿参数）
- **嵌入维度**：768（支持 512/256/128 等 Matryoshka）
- **场景**：速度/质量平衡、多语言

#### Embedding Ultra（默认）

- **Model ID**：`models/mom-embedding-ultra`
- **HuggingFace**：`llm-semantic-router/mmbert-embed-32k-2d-matryoshka`
- **用途**：长上下文多语言语义相似度，支持 2D Matryoshka
- **架构**：mmBERT 2D Matryoshka（307M）
- **嵌入维度**：768（可通过 Matryoshka 降维）
- **场景**：默认语义缓存、检索与工具相似度

### 3. 幻觉检测模型

#### Halugate Sentinel

- **Model ID**：`models/mom-halugate-sentinel`
- **HuggingFace**：`LLM-Semantic-Router/halugate-sentinel`
- **用途**：幻觉筛查第一阶段
- **架构**：BERT-base（110M）
- **场景**：快速幻觉检测、预过滤

#### Halugate Detector

- **Model ID**：`models/mom-halugate-detector`
- **HuggingFace**：`KRLabsOrg/lettucedect-base-modernbert-en-v1`
- **用途**：更精确的幻觉核验
- **架构**：ModernBERT-base（149M）
- **上下文长度**：8192 tokens
- **场景**：事实准确性、 grounding 检查

#### Halugate Explainer

- **Model ID**：`models/mom-halugate-explainer`
- **HuggingFace**：`tasksource/ModernBERT-base-nli`
- **用途**：通过 NLI 解释幻觉推理
- **架构**：ModernBERT-base（149M）
- **类别**：3（蕴含/中性/矛盾）
- **场景**：可解释 AI、幻觉分析

## 选型指南

### 按场景

| 场景 | 推荐模型 | 原因 |
| ---- | -------- | ---- |
| 领域路由 | mmbert32k-intent-classifier-merged | 14 个 MMLU 类别，32K 上下文 |
| 隐私保护 | mmbert32k-pii-detector-merged | 17 种实体、35 BIO 标签，32K |
| 内容安全 | mmbert32k-jailbreak-detector-merged | 合并 mmBERT 的提示注入检测 |
| 语义缓存 | mom-embedding-ultra | 默认 32K 多语言嵌入 |
| 长上下文检索 | mom-embedding-pro | 32K context，1024 维 |
| 幻觉检查 | mom-halugate-detector | ModernBERT，8K 上下文 |
| 用户反馈 | mmbert32k-feedback-detector-merged | 4 类反馈，合并 mmBERT |

### 按性能

| 需求 | 层级 | 示例 |
| ---- | ---- | ---- |
| 极快（&lt;10ms） | Light | mom-embedding-flash、mmbert32k-jailbreak-detector-merged |
| 均衡（10–50ms） | Default | mom-embedding-ultra、mmbert32k-intent-classifier-merged |
| 高质量（50–200ms） | Pro | mom-embedding-pro、mom-halugate-detector |

## 配置

### 在路由器中使用 MoM 模型

通过规范的 `global.model_catalog` 块配置 MoM 模型，模块级设置位于 `global.model_catalog.modules`：

```yaml
global:
  model_catalog:
    system:
      domain_classifier: "models/mmbert32k-intent-classifier-merged"
      pii_classifier: "models/mmbert32k-pii-detector-merged"
      prompt_guard: "models/mmbert32k-jailbreak-detector-merged"
    modules:
      classifier:
        domain:
          model_ref: "domain_classifier"
          threshold: 0.6
          use_cpu: true
        pii:
          model_ref: "pii_classifier"
          threshold: 0.9
          use_cpu: true
      prompt_guard:
        model_ref: "prompt_guard"
        threshold: 0.7
        use_cpu: true
```

### 自定义系统绑定

在 `config.yaml` 中覆盖内置系统模型绑定：

```yaml
global:
  model_catalog:
    system:
      domain_classifier: "models/your-domain-classifier"
      pii_classifier: "models/your-pii-classifier"
      prompt_guard: "models/your-prompt-guard"
```

## 模型架构

### 基于 LoRA 的模型

许多 MoM 模型使用 LoRA（低秩适配）以提升效率：

- **基座**：BERT-base-uncased（约 1.1 亿参数）
- **LoRA 适配器**：每任务 &lt;100 万参数
- **显存**：约 440MB 基座 + 每适配器约 4MB
- **推理速度**：与基座相当（CPU 上约 10–20ms）

### ModernBERT 模型

较新模型采用 ModernBERT 以提升表现：

- **架构**：ModernBERT-base（约 1.49 亿参数）
- **上下文**：8192 tokens（对比 BERT 的 512）
- **效果**：长上下文任务上更准确
- **场景**：幻觉检测、反馈分类

## 下一步

- **[信号驱动决策](./signal-driven-decisions)** — MoM 模型如何驱动路由决策
- **[Domain](../tutorials/signal/learned/domain)** — 使用 mmbert32k-intent-classifier-merged 做路由
- **[PII](../tutorials/signal/learned/pii)** — 配置 mmbert32k-pii-detector-merged
- **[RAG](../tutorials/plugin/rag)** — 将 MoM 嵌入用于路由侧检索

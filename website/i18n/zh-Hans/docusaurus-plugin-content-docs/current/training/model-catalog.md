---
title: 当前模型目录
sidebar_label: 模型目录
translation:
  source_commit: "e8c4109fd4151ad0c7c0163c8ead375bef882ddf"
  source_file: "docs/training/model-catalog.md"
  outdated: false
---

# 当前模型目录 {#current-model-catalog}

本目录覆盖当前
[MoM 多语言嵌入集合](https://huggingface.co/collections/llm-semantic-router/mom-multilingual-embed)
中的全部 5 个产物，以及
[分类器集合](https://huggingface.co/collections/llm-semantic-router/mom-multilingual-class)
中的全部 14 个产物。
一行代表一个逻辑架构，并列出其所有已发布形态。

## 嵌入与重排序产物 {#embedding-and-reranking-artifacts}

| 逻辑模型 | 已发布产物 | 架构 | 训练方法 |
| --- | --- | --- | --- |
| mmBERT-32K 基础 | [`mmbert-32k-yarn`](https://huggingface.co/llm-semantic-router/mmbert-32k-yarn) | 带 32K YaRN 上下文的 ModernBERT 掩码语言编码器 | 持续多语言掩码语言建模 |
| mmBERT-32K embedder | [`mmbert-embed-32k-2d-matryoshka`](https://huggingface.co/llm-semantic-router/mmbert-embed-32k-2d-matryoshka) | 可选层和维度的 Bi-encoder 嵌入 | 带 2D Matryoshka 监督的 multiple-negatives ranking |
| mmBERT-32K reranker | [`mmbert-rerank-32k-2d-matryoshka`](https://huggingface.co/llm-semantic-router/mmbert-rerank-32k-2d-matryoshka) | 带 20 个层/维度打分头的 Cross-encoder | 在所有头上平均的二元相关性损失 |
| 小型多模态 embedder | [`multi-modal-embed-small`](https://huggingface.co/llm-semantic-router/multi-modal-embed-small) | MiniLM、SigLIP 和 Whisper-tiny 塔加两层融合；384 维 | 分阶段图文和音文对比对齐，带 Matryoshka 损失 |
| 大型多模态 embedder | [`multi-modal-embed-large`](https://huggingface.co/llm-semantic-router/multi-modal-embed-large) | mmBERT-32K、SigLIP2-SO400M 和 Whisper-medium 三编码器；768 维 | 带难负例的缓存混合负例排序 |

数据流、目标、配置和命令见 [mmBERT-32K 模型](./mmbert-32k-models) 和[多模态嵌入](./multimodal-embeddings)。

## 分类器产物 {#classifier-artifacts}

下面前六个分类器使用多语言 mmBERT-32K/ModernBERT 编码器。两个已发布的安全产物使用 `jhu-clsp/mmBERT-base`；当前安全工作流可以训练 32K 后继产物。序列分类器为请求预测一个标签；PII 模型为每个 token 预测一个 BIO 标签。

| 逻辑模型 | 标签或输出 | 已发布产物 | 训练方法 |
| --- | --- | --- | --- |
| 意图分类器 | 14 个学科领域 | [`merged`](https://huggingface.co/llm-semantic-router/mmbert32k-intent-classifier-merged)、[`LoRA`](https://huggingface.co/llm-semantic-router/mmbert32k-intent-classifier-lora) | 在 MMLU-Pro 加上回退意图样本上做 LoRA 序列分类 |
| 越狱检测器 | `benign`、`jailbreak` | [`merged`](https://huggingface.co/llm-semantic-router/mmbert32k-jailbreak-detector-merged)、[`LoRA`](https://huggingface.co/llm-semantic-router/mmbert32k-jailbreak-detector-lora) | 在良性/有毒聊天、攻击数据和模式增强上做 LoRA 序列分类 |
| 反馈检测器 | 四种反馈状态 | [`merged`](https://huggingface.co/llm-semantic-router/mmbert32k-feedback-detector-merged)、[`LoRA`](https://huggingface.co/llm-semantic-router/mmbert32k-feedback-detector-lora) | 类别加权的 LoRA 序列分类 |
| 模态路由器 | `AR`、`DIFFUSION`、`BOTH` | [`merged`](https://huggingface.co/llm-semantic-router/mmbert32k-modality-router-merged)、[`LoRA`](https://huggingface.co/llm-semantic-router/mmbert32k-modality-router-lora) | 带 focal loss、类别平衡和可选合成混合模态提示词的 LoRA |
| 事实核查分类器 | `FACT_CHECK_NEEDED`、`NO_FACT_CHECK_NEEDED` | [`merged`](https://huggingface.co/llm-semantic-router/mmbert32k-factcheck-classifier-merged)、[`LoRA`](https://huggingface.co/llm-semantic-router/mmbert32k-factcheck-classifier-lora) | 在信息寻求和非信息寻求提示词上做平衡 LoRA 序列分类 |
| PII 检测器 | 17 种实体类型，由 35 个 BIO 标签表示 | [`merged`](https://huggingface.co/llm-semantic-router/mmbert32k-pii-detector-merged)、[`LoRA`](https://huggingface.co/llm-semantic-router/mmbert32k-pii-detector-lora) | 带字符偏移到 token 对齐的 LoRA token 分类 |
| 安全 Level 1 | `safe`、`unsafe` | [`LoRA adapter`](https://huggingface.co/llm-semantic-router/mmbert-safety-binary-merged) | 确定性仅提示词的 LoRA 序列分类 |
| 安全 Level 2 | 九种危害输出 | [`LoRA`](https://huggingface.co/llm-semantic-router/mmbert-safety-binary-hazard) | 带固定分类对照的确定性仅提示词 LoRA 序列分类 |

前六个任务见[分类器模型](./classifier-models)，两级安全流水线见[安全分类器](./mmbert-safety-classifier)。

## 选择发布形态 {#choose-a-release-shape}

当运行时期望独立 Transformers 模型时，使用合并产物。当运行时可以加载 PEFT adapter 且你想要更小的任务专用产物时，使用 LoRA 产物。两种形态都必须保留训练时使用的同一分词器、标签顺序、基础模型兼容性和预处理契约。

Model Card 描述已发布权重。仓库中已核对的训练配置描述新运行。两者不同时，将发布视为已有产物，将树内配置视为再训练的事实来源；没有原始数据和运行回执时，不要假设新检查点会逐位相同。

Level 1 安全产物是 PEFT adapter，即使其历史名称以 `-merged` 结尾。检查产物内容和元数据，而不是从后缀推断加载方法。

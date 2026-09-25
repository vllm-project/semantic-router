---
title: 训练 Vela Router 模型
sidebar_label: 概览
translation:
  source_commit: "915ddf56e0335e2046c38aa17c4aec6233908039"
  source_file: "docs/training/training-overview.md"
  outdated: false
---

# 训练 Vela Router 模型 {#train-vela-router-models}

根据应用的语言、主题和路由策略适配 Vela。模型家族包含共享编码器、任务分类器、Embedding 和 Reranker。需要推理时，先使用已发布的任务模型；评测发现能力缺口后，再进行训练。

部署请参阅[使用 Vela 模型](../tutorials/global/vela-models.md)。本节介绍如何训练和评测自己的任务模型。

## 选择训练流程 {#choose-a-training-workflow}

| 目标 | 流程 | 训练输出 |
| --- | --- | --- |
| 改善语义搜索或请求相似度 | [Embedding](./mmbert-32k-models#embedding-model-bi-encoder) | 查询和文档向量 |
| 改善检索候选的顺序 | [Reranking](./mmbert-32k-models#reranking-model-cross-encoder) | 每个查询与文档对的相关性分数 |
| 适配领域、反馈、提示词攻击、事实核查或输出模态检测 | [分类器](./classifier-models) | 请求标签 |
| 检测个人信息 | [PII](./classifier-models#pii-detector) | 实体片段 |
| 应用内容风险策略 | [Safety 和 Hazard](./mmbert-safety-classifier) | 安全判断和风险类别 |

[Vela 模型目录](./model-catalog)列出了全部 11 个模型。Vela 1.0 接受文本输入；[多模态嵌入](./multimodal-embeddings)属于独立家族。[后端模型评测](./model-performance-eval)和[学习式模型选择](./ml-model-selection)用于选择下游 LLM。

## 选择起始模型 {#record-the-base-and-task-lineage}

训练新任务时，使用 [Vela Encoder](https://huggingface.co/llm-semantic-router/Vela-1.0-Encoder-307M)。它是分类器、Embedding 和 Reranker 的共同基座。下载明确的 Hub revision，并保持权重、tokenizer 和配置一致。

改进已有任务时，从完整的 Vela 任务 checkpoint 继续训练，保留已训练的 head。除非要训练新的输出契约，否则保持标签和预处理一致。各任务指南分别说明新任务初始化和继续训练。

## 准备应用数据 {#prepare-examples-from-your-application}

从真实请求及预期行为开始。覆盖应用会接收的语言和长度、应当不触发信号的普通请求，以及接近决策边界的样本。

把相关对话、文档和翻译保留在同一数据分区。训练集用于更新权重，开发集用于选择 checkpoint 和阈值，独立测试集用于最终对比。任务指南提供数据格式和现有构建工具。

## 训练并对比 {#train-and-compare}

先运行小规模训练，检查数据和输出标签。然后用相同输入和推理设置，对比训练后的模型与已发布任务模型。

| 任务 | 评测指标 |
| --- | --- |
| 分类器 | 各类别 precision、recall、F1 和误报 |
| PII | 实体级 precision、recall 和 F1 |
| Embedding | 检索与语义相似度质量 |
| Reranker | 固定候选列表上的排序质量 |
| Safety 和 Hazard | 漏检风险、安全请求误报和类别覆盖 |

分别检查语言和长度分组。长上下文应用需要同时覆盖短请求和真实长文档，并把相关内容放在开头、中间和末尾。在实际服务长度下测量延迟。

## 接入 Router {#use-the-trained-model-in-the-router}

导出包含 tokenizer 和标签的完整 checkpoint。在[本地运行模型](../installation/runtime/in-process.md)中选择支持的引擎和输入预算，然后验证配置：

```bash
vllm-sr config validate --config config.yaml
```

使用[路由预览](../installation/runtime/lifecycle-diagnostics.md)，在部署到真实流量前检查代表性请求的信号、决策和延迟。

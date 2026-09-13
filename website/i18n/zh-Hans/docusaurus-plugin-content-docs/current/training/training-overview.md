---
title: 训练并评测 Router 模型
sidebar_label: 概览
translation:
  source_commit: "e56591a9cb24f073bf159927e87116ba6d278741"
  source_file: "docs/training/training-overview.md"
  outdated: false
---

# 训练并评测 Router 模型 {#train-and-evaluate-router-models}

Semantic Router 在把请求发给 LLM 之前使用小型、任务专用的模型。这些模型产生路由信号：嵌入请求、为候选排序、分类意图或风险，或预测应由哪个 provider 模型作答。它们不生成最终响应。

按三步使用本节：

1. 选择所需的路由决策。
2. 了解该模型家族的架构和训练目标。
3. 用 Router 将使用的同一输入输出契约训练、评测并导出产物。

## 按路由决策选择模型 {#choose-the-model-by-routing-decision}

| 你需要 | 从这里开始 | 输出 |
| --- | --- | --- |
| 高效比较查询和文档 | [mmBERT-32K embedder](./mmbert-32k-models#embedding-model-bi-encoder) | 每个输入一个规范化向量 |
| 以更高准确率重打分短列表 | [mmBERT-32K reranker](./mmbert-32k-models#reranking-model-cross-encoder) | 每个查询-文档对的相关性分数 |
| 将文本、图像和音频放入同一向量空间 | [多模态嵌入](./multimodal-embeddings) | 规范化的跨模态向量 |
| 检测意图、越狱、反馈、模态、事实核查需求或 PII | [分类器模型](./classifier-models) | 类别、概率分布或 token 标签 |
| 应用分层提示词安全策略 | [安全分类器](./mmbert-safety-classifier) | `safe`/`unsafe`，随后是危害类别 |
| 学习应由哪个 provider 模型作答 | [基于 ML 的模型选择](./ml-model-selection) | 一个 provider 模型选择 |
| 比较 provider 池中已有模型 | [模型性能评测](./model-performance-eval) | 按模型和按类别的分数 |

[模型目录](./model-catalog)列出当前 MoM 多语言嵌入和分类器集合中的每一个产物，并将发布变体映射到其训练工作流。

## 理解三种常见架构 {#understand-the-three-common-architectures}

本节中大多数 Router 模型使用以下模式之一：

| 模式 | 如何处理输入 | 最适合 |
| --- | --- | --- |
| Bi-encoder | 独立编码每个输入，然后比较向量 | 大规模检索和语义缓存查找 |
| Cross-encoder | 联合编码一对并预测一个分数 | 对小候选集做准确重排序 |
| Encoder plus task head | 编码一个请求，然后预测序列或 token 标签 | 在线路由和策略信号 |

多模态模型用独立的文本、图像和音频塔扩展 bi-encoder 模式，其输出被投影到共享空间。目录和家族页面说明精确的塔、维度、标签和目标。

## Adapter 与合并模型 {#adapter-versus-merged-model}

若干分类器条目同时有 `-lora` 和 `-merged` 产物。它们是同一逻辑模型的两种发布形态：

- **LoRA adapter** 存储训练好的低秩更新和分类头。它很小，但推理还需要兼容的基础模型。
- **合并模型** 将 adapter 折入基础权重。它更大，可由受支持的运行时作为独立分类器加载。

选择你的推理后端支持的形态。不要把这两个名称当作独立训练的架构来比较。

## 遵循训练生命周期 {#follow-the-training-lifecycle}

### 1. 定义路由契约 {#1-define-the-routing-contract}

指定标签或分数、该输出如何改变路由、支持的语言和请求长度、延迟预算以及回退行为。标签只有映射到可观察的 Router 决策或策略时才有用。

### 2. 准备版本化数据 {#2-prepare-versioned-data}

保持训练、验证和测试划分分开。记录数据集修订、许可证、预处理、标签定义和合成数据规则。
划分前先去重，避免近乎相同的样本泄漏到评测中。

### 3. 从冒烟运行开始 {#3-start-with-a-smoke-run}

以已核对配置或脚本的 `--help` 输出作为事实来源。显式解析路径，运行小样本，并在分配完整训练运行前检查标签计数、损失和验证输出。

### 4. 评测路由行为 {#4-evaluate-routing-behavior}

将指标与决策匹配：

| 任务 | 最低有用评测 |
| --- | --- |
| 序列分类 | 每类精确率、召回率、F1 和混淆矩阵 |
| PII token 分类 | 实体级精确率、召回率和 F1 |
| 安全检测 | 假阴性和假阳性率，以及每危害 F1 |
| 嵌入检索 | Recall@k、排序质量、语言/领域切片和延迟 |
| 模型选择 | 端到端回答质量、成本、延迟，以及相对 oracle 的 regret |

始终保留留出测试集。按语言、领域、输入长度以及部署关心的失败模式切片结果。

### 5. 导出并集成 {#5-export-and-integrate}

导出分词器、模型或 adapter、标签映射，以及运行时所需的任何架构元数据。然后校验完整的 Router 配置：

```bash
vllm-sr config validate --config config.yaml
```

最后，通过完整 Router 路径发送代表性请求。这能捕获离线训练器无法检测的标签顺序、预处理、维度或产物形态不匹配。

## 推荐阅读路径 {#recommended-reading-path}

若你刚接触这些模型，按此顺序阅读页面：

1. [当前模型目录](./model-catalog)
2. [mmBERT-32K 基础、embedder 和 reranker](./mmbert-32k-models)
3. [小型和大型多模态嵌入](./multimodal-embeddings)
4. [mmBERT-32K 分类器模型](./classifier-models)
5. [两级安全分类器](./mmbert-safety-classifier)
6. [模型性能评测](./model-performance-eval)

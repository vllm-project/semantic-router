---
is_mtpe: false
title: 训练 Vela 路由模型
sidebar_label: 概述
translation:
  source_commit: "118fb72eb67a0e7193697e350d7b494dc016631e"
  source_file: "docs/training/training-overview.md"
  outdated: false
---

# 训练 Router 的模型 {#train-the-routers-models}

Router 会询问两类模型，而对大多数应用来说，只有其中一类值得训练。

- **决策模型**零样本回答每个路由判断：应用信号、你自己的问题、System One 选择器、防护和评审。新的路由问题无需训练：把它写成一条 [`systemone` 规则](../tutorials/signal/systemone)，再调优它的 predicate（[编写自定义问题](../system-one/custom-questions)）。当某个问题需要更高准确率时，下一步是换用更大的[决策模型](../system-one/deployments)。
- **嵌入模型**只用于召回：响应缓存、记忆和 RAG 用它查找候选，没有任何决策读取它。当召回在你的数据上漏掉相关内容时，训练一个嵌入模型。

## 选择训练流程 {#choose-a-training-workflow}

| 你的目标 | 流程 | 训练内容 |
| --- | --- | --- |
| 改进语义搜索、缓存召回或记忆召回 | [嵌入](./mmbert-32k-models#embedding-model-bi-encoder) | 查询和文档的向量 |
| 选择路由要调用的生成模型 | [提供方模型评测](./model-performance-eval) | 不训练：评测你的候选模型 |

## 选择起始 checkpoint {#record-the-base-and-task-lineage}

从 [Vela Encoder](https://huggingface.co/vllm-sr/Vela-1.0-Encoder-307M) 或已发布的 [Vela Embedding](https://huggingface.co/vllm-sr/Vela-1.0-Encoder-307M-Embedding) 开始。下载明确的 Hub revision，并把它的 tokenizer 和配置与权重放在一起。

## 用你的应用准备示例 {#prepare-examples-from-your-application}

从真实请求和你期望它们召回的内容开始。包括你的应用会收到的语言和输入长度，以及不应匹配的相近样本对。

把相关的对话、文档和译文放在同一个划分中。用训练数据更新权重，用开发数据选择 checkpoint，并用单独的测试集做最终比较。

## 训练并比较 {#train-and-compare}

先运行一次小规模训练，检查数据。然后在相同的输入和推理设置下，把你训练的模型与已发布的嵌入模型比较：分别在不同语言和长度的切片上比较检索和语义相似度质量。按你计划服务的输入长度测量延迟。

## 在 Router 中使用训练好的模型 {#use-the-trained-model-in-the-router}

导出包含 tokenizer 的完整 checkpoint。把它声明为嵌入部署，并在召回使用它的地方指定它（[嵌入](../model-runtime/guides/embeddings.md)）。用其他模型建立索引的存储必须重新建立索引，或继续指定建立索引的那个模型。然后校验你的配置：

```bash
vllm-sr config validate --config config.yaml
```

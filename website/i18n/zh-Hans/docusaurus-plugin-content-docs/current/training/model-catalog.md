---
title: Vela 模型目录
sidebar_label: 模型目录
translation:
  source_commit: "19c03bfd28b6c53df9b682bbfc3e5fc1d372c803"
  source_file: "docs/training/model-catalog.md"
  outdated: false
---

# Vela 模型目录 {#vela-model-catalog}

[Vela 1.0](https://huggingface.co/collections/llm-semantic-router/vela-10-router-models-6aa555ba70cc6997d6d67798)是面向智能路由的模型家族。全部 12 个模型共享 307M 参数的 Vela Encoder 基座，覆盖请求理解、安全、检索和重排序。

## 选择模型 {#choose-a-model}

| 模型 | 用途 |
| --- | --- |
| [Encoder](https://huggingface.co/llm-semantic-router/Vela-1.0-Encoder-307M) | 在共享基座上训练新任务 |
| [Domain](https://huggingface.co/llm-semantic-router/Vela-1.0-Encoder-307M-Domain) | 将请求分为 14 个主题领域 |
| [Guard](https://huggingface.co/llm-semantic-router/Vela-1.0-Encoder-307M-Guard) | 检测提示词注入和越狱攻击 |
| [Safety](https://huggingface.co/llm-semantic-router/Vela-1.0-Encoder-307M-Safety) | 检测不安全内容 |
| [Shield](https://huggingface.co/llm-semantic-router/Vela-1.0-Encoder-307M-Shield) | 检测不安全请求；可替代 Safety |
| [Hazard](https://huggingface.co/llm-semantic-router/Vela-1.0-Encoder-307M-Hazard) | 识别 12 类内容风险 |
| [PII](https://huggingface.co/llm-semantic-router/Vela-1.0-Encoder-307M-PII) | 定位 17 类个人信息 |
| [FactCheck](https://huggingface.co/llm-semantic-router/Vela-1.0-Encoder-307M-FactCheck) | 判断回答是否需要事实核查 |
| [Feedback](https://huggingface.co/llm-semantic-router/Vela-1.0-Encoder-307M-Feedback) | 识别满意、澄清、纠错、修改或无反馈 |
| [Modality](https://huggingface.co/llm-semantic-router/Vela-1.0-Encoder-307M-Modality) | 选择文本、图像或组合输出 |
| [Embedding](https://huggingface.co/llm-semantic-router/Vela-1.0-Encoder-307M-Embedding) | 比较请求并检索相关文档 |
| [Reranker](https://huggingface.co/llm-semantic-router/Vela-1.0-Encoder-307M-Reranker) | 按相关性重新排列候选文档 |

Guard 检测改变指令执行的攻击；Safety 检测内容风险，Hazard 识别风险类别。应用同时需要提示词攻击防护和内容策略时，可以组合使用。Shield 是单独训练的模型，与 Safety 使用相同的 `safe`/`unsafe` 接口；Safety 仍是默认模型，改用 Shield 的方法见 [Safety 信号指南](../tutorials/signal/learned/safety.md#select-vela-shield)。

Embedding 和 Reranker 提供四种编码器深度和五种维度，方便权衡质量、延迟和内存。选择方法见 [Embedding 和 Reranking](./mmbert-32k-models)。

## 部署或定制 {#deploy-or-customize}

运行已发布模型，请参阅[使用 Vela 模型](../tutorials/global/vela-models.md)，了解默认下载和 Router 配置。Model card 提供独立使用的 quickstart 和评测结果。

使用自己的数据适配模型，请从[训练概览](./training-overview)开始。共享 Encoder 用于训练；路由信号使用对应的任务 checkpoint。

## 早期版本和多模态模型 {#earlier-releases-and-multimodal-models}

早期 [mmBERT 分类器集合](https://huggingface.co/collections/llm-semantic-router/mom-multilingual-class)和 [mmBERT 嵌入集合](https://huggingface.co/collections/llm-semantic-router/mom-multilingual-embed)仍可用于现有集成和对比。[产物索引](https://github.com/vllm-project/semantic-router/blob/main/src/training/model_artifacts.json)保留了它们的训练入口。

Vela 1.0 包含文本模型。图像和音频嵌入模型使用独立的[多模态训练指南](./multimodal-embeddings)。

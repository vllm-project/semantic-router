---
title: 支持的模型
sidebar_label: 支持的模型
description: 内置运行时提供的每个 System One 模型，按家族列出，包括其规模、问题类型、输入和显存。
is_mtpe: false
translation:
  source_commit: "bc5c6bfb747ffe9c5f35d98b203427dada8c4c56"
  source_file: "docs/model-runtime/supported-models/overview.md"
  outdated: false
---

# 支持的模型 {#supported-models}

这些是内置模型运行时提供的决策模型：Router 可以向其提出 System One 问题的每一个模型。每个模型都有自己的页面，由 Router 和运行时使用的目录生成，因此其问题类型、规模和固定的修订版本总是与代码一致。

:::tip 选哪一个？
从默认的 **vela3-fast** 开始，在 CPU 上运行。想要更高准确率并且内置问题仍由训练过的适配器回答，就换更大的 Vela 3.0 规格；想就图像和视频提出你自己的问题，就换 Decision 3.0。[选择模型](../get-started/choose-a-model.md)对它们做了比较。
:::

## 家族 {#families}

| 家族 | 用途 | Span | 运行于 |
| --- | --- | --- | --- |
| [Decision 3.0](./decision-3/overview.md) | 零样本回答你关于文本、图像和视频的问题，0.6B 到 27B | 否 | GPU |
| [Decision 2.0](./decision-2/overview.md) | 零样本回答你关于文本的问题，0.6B 到 27B | 否 | CPU（Kai-0.6B）或 GPU |
| [Decision 1.0](./decision-1/overview.md) | 第一代决策模型，0.6B 到 9B 的编码器和解码器 | 否 | CPU（0.6B）或 GPU |
| [Vela 3.0](./vela-3/overview.md) | Router 自己的模型：针对内置问题训练过，支持 span；默认模型 | 是 | CPU（vela3-fast）或 GPU |
| [Vela 2.0](./vela-2/overview.md) | 上一代 Router 模型，支持 set 和 span 问题 | 是 | CPU（0.3B）或 GPU |

每个模型都回答 `choice` 和 `noul` 问题；其余类型见下表。不支持 span 的模型仍可作为决策模型：把 `pii` 和 `hallucination` 绑定到支持 span 的部署（[Span 能力](../../system-one/get-started/concepts.md#span-capability)）。

## 全部模型 {#all-models}

<!-- BEGIN GENERATED MODEL TABLE -->

### Decision 3.0

| 模型 | 参数量 | 回答的问题类型 | Span | 输入 | 显存 |
| --- | --- | --- | --- | --- | --- |
| [d3-edge](./decision-3/d3-edge) | 0.595B | choice, noul, score | — | text, image, video | 4 GiB |
| [d3-lite](./decision-3/d3-lite) | 0.853B | choice, noul, score | — | text, image, video | 4 GiB |
| [d3-nano](./decision-3/d3-nano) | 2.21B | choice, noul, score | — | text, image, video | 8 GiB |
| [d3-mini](./decision-3/d3-mini) | 4.54B | choice, noul, score | — | text, image, video | 16 GiB |
| [d3-flash](./decision-3/d3-flash) | 8.39B | choice, noul, score | — | text, image, video | 32 GiB |
| [d3](./decision-3/d3) | 26.1B | choice, noul, score | — | text, image, video | 72 GiB |
| [d3-turbo](./decision-3/d3-turbo) | 26.1B | choice, noul, score | — | text, image, video | 72 GiB |
| [d3-ultra](./decision-3/d3-ultra) | 26.1B | choice, noul, score | — | text, image, video | 72 GiB |

### Decision 2.0

| 模型 | 参数量 | 回答的问题类型 | Span | 输入 | 显存 |
| --- | --- | --- | --- | --- | --- |
| [Decision-2.0-Kai-0.6B](./decision-2/kai-0-6b) | 0.597B | choice, noul, score | — | text | 4 GiB |
| [Decision-2.0-Eos-0.8B](./decision-2/eos-0-8b) | 0.753B | choice, noul, score | — | text | 4 GiB |
| [Decision-2.0-Sol-2B](./decision-2/sol-2b) | 1.88B | choice, noul, score | — | text | 8 GiB |
| [Decision-2.0-Nox-4B](./decision-2/nox-4b) | 4.21B | choice, noul, score | — | text | 16 GiB |
| [Decision-2.0-Lux-9B](./decision-2/lux-9b) | 7.94B | choice, noul, score | — | text | 24 GiB |
| [Decision-2.0-Vega-27B](./decision-2/vega-27b) | 29.4B | choice, noul, score | — | text | 72 GiB |

### Decision 1.0

| 模型 | 参数量 | 回答的问题类型 | Span | 输入 | 显存 |
| --- | --- | --- | --- | --- | --- |
| [Decision-1.0-Kai-0.6B](./decision-1/kai-0-6b) | 0.572B | choice, noul, score | — | text | 4 GiB |
| [Decision-1.0-Lex-0.6B](./decision-1/lex-0-6b) | 0.572B | choice, noul, score | — | text | 4 GiB |
| [Decision-1.0-Route-0.6B](./decision-1/route-0-6b) | 0.572B | choice, noul, score | — | text | 4 GiB |
| [Decision-1.0-Eos-0.8B](./decision-1/eos-0-8b) | 0.753B | choice, noul, score | — | text | 4 GiB |
| [Decision-1.0-Sol-2B](./decision-1/sol-2b) | 1.88B | choice, noul, score | — | text | 8 GiB |
| [Decision-1.0-Nox-4B](./decision-1/nox-4b) | 4.21B | choice, noul, score | — | text | 16 GiB |
| [Decision-1.0-Lux-9B](./decision-1/lux-9b) | 7.94B | choice, noul, score | — | text | 24 GiB |

### Vela 3.0

| 模型 | 参数量 | 回答的问题类型 | Span | 输入 | 显存 |
| --- | --- | --- | --- | --- | --- |
| [vela3-fast](./vela-3/vela3-fast) | 1.52B | choice, noul, span | router | text | 12 GiB |
| [vela3-lite](./vela-3/vela3-lite) | 0.903B | choice, noul, score, span | router | text, image, video | 4 GiB |
| [vela3-mini](./vela-3/vela3-mini) | 4.79B | choice, noul, score, span | router | text, image, video | 16 GiB |
| [vela3-flash](./vela-3/vela3-flash) | 8.83B | choice, noul, score, span | router | text, image, video | 32 GiB |
| [vela3-turbo](./vela-3/vela3-turbo) | 27.6B | choice, noul, score, span | router | text, image, video | 72 GiB |

### Vela 2.0

| 模型 | 参数量 | 回答的问题类型 | Span | 输入 | 显存 |
| --- | --- | --- | --- | --- | --- |
| [Vela-2.0-0.3B](./vela-2/vela-2-0-3b) | 0.309B | choice, noul, score, set, span | router | text | 4 GiB |
| [Vela-2.0-0.8B](./vela-2/vela-2-0-8b) | 0.756B | choice, noul, score, set, span | router, broad | text | 4 GiB |
| [Vela-2.0-4B](./vela-2/vela-2-4b) | 4.21B | choice, noul, score, set, span | router, broad | text | 16 GiB |
| [Vela-2.0-9B](./vela-2/vela-2-9b) | 7.95B | choice, noul, score, set, span | router, broad | text | 24 GiB |

<!-- END GENERATED MODEL TABLE -->

“显存”是目录为模型所在设备给出的最低值；请用你的输入长度和并发量测量实际峰值。

:::note 嵌入模型
召回功能使用的嵌入模型 Vela 1.0 Embedding、Qwen3-Embedding-0.6B 和 Vela 1.0 Omni 不是 System One 模型，不在这里列出。参见[嵌入](../guides/embeddings.md)和[图像与音频](../guides/multimodal.md)。
:::

## 这些页面如何生成 {#how-these-pages-are-made}

模型页面由 `tools/docs/generate_model_pages.py` 根据 Router 的决策目录、运行时的注册表、记录的参考答案和 Router 的校准阈值生成。更改内置模型后运行 `make docs-model-pages`，在此之前 `make docs-generated-check` 会失败。家族页面和本页是手写的，生成的表格嵌在其中。模型页面只有英文版本，中文站点显示英文内容。

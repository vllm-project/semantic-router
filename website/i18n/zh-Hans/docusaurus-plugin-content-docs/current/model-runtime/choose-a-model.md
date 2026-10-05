---
title: 选择模型、规模和硬件
sidebar_label: 选择模型
description: 每个任务该用哪个模型，决策模型需要多大，以及用什么硬件运行。
translation:
  source_commit: "73936aeacee7fbbf670416aa6a75eaeacd436a03"
  source_file: "docs/model-runtime/choose-a-model.md"
  outdated: false
---

# 选择模型、规模和硬件

从任务出发。下面每个内置模型都固定到确切的 Hugging Face revision，因此同一个名字始终加载相同的文件。

## 按任务选择 {#by-task}

| 你想要 | 模型 | 规模 | 说明 |
| --- | --- | --- | --- |
| 按主题路由（数学、法律、代码……） | `vllm-sr/Vela-1.0-Encoder-307M-Domain` | 307M | 14 个领域 |
| 发现需要事实核查的请求 | `vllm-sr/Vela-1.0-Encoder-307M-FactCheck` | 307M | 它标出需要核查，但不核查事实 |
| 读取用户对上一个回答的反应 | `vllm-sr/Vela-1.0-Encoder-307M-Feedback` | 307M | 满意、需要澄清、回答错误、想要别的、无反馈 |
| 区分文本请求和图片请求 | `vllm-sr/Vela-1.0-Encoder-307M-Modality` | 307M | 只读取书面请求 |
| 发现个人信息 | `vllm-sr/Vela-1.0-Encoder-307M-PII` | 307M | 17 种实体类型，给出精确的字符片段 |
| 拦截提示词注入和越狱 | `vllm-sr/Vela-1.0-Encoder-307M-Guard` | 307M | |
| 标记不安全内容 | `vllm-sr/Vela-1.0-Encoder-307M-Safety` 或 `-Shield` | 307M | Shield 是另一种安全模型 |
| 指出风险类别 | `vllm-sr/Vela-1.0-Encoder-307M-Hazard` | 307M | 12 个独立的危害类别，带已发布的阈值 |
| 对照来源检查回答 | `vllm-sr/Vela-1.0-Encoder-307M-Halu` | 307M | 标出回答中无依据的片段 |
| 用于缓存、记忆、RAG 和工具的 embedding | `vllm-sr/Vela-1.0-Encoder-307M-Embedding` | 307M | 更小的维度和更少的层以质量换速度 |
| 更大或带指令的文本 embedding | `Qwen/Qwen3-Embedding-0.6B` | 0.6B | 1,024 维 |
| 对检索到的文档重排序 | `vllm-sr/Vela-1.0-Encoder-307M-Reranker` | 307M | |
| 把文本、图片和音频放进同一向量空间 | `vllm-sr/Vela-1.0-Omni-Nano` 或 `-Mini` | 164M / 1.36B | Mini 更准确，并接受更长的文本 |
| 用自然语言提出你自己的问题 | 决策模型（见下一节） | 0.6B 到 27B | |

任务模型在 CPU 上都运行良好：在 16 个核上，Vela Domain 请求的中位耗时约 12 ms，
比早期版本使用的原生绑定快三倍
（[测量记录](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/vela1-performance.md)）。
它们大多最多读取 32,768 个 token；更长或更短的上限列在每个模型卡片和 `GET /v1/models` 中。

## 决策模型 {#decision-models}

决策模型回答你自己写的问题，例如“这需要逐步推理吗？”或“这些模型中哪个应该回答？”。
选择对你的问题足够准确的最小模型。

| 模型 | 规模 | 适合运行在 | 适合 |
| --- | --- | --- | --- |
| `vllm-sr/Decision-2.0-Kai-0.6B` | 0.6B | CPU（16 核上两个问题约 0.2 秒）或任意 GPU | 快速、简单的路由问题；入门的默认选择 |
| `vllm-sr/Decision-2.0-Eos-0.8B` | 0.8B | CPU 或任意 GPU | 稍难的问题，成本相近 |
| `vllm-sr/Decision-2.0-Sol-2B` | 2B | GPU；低流量时也可用 CPU | 需要更多判断的问题 |
| `vllm-sr/Decision-2.0-Nox-4B` | 4B | GPU | 细致的问题和较多选项 |
| `vllm-sr/Decision-2.0-Lux-9B` | 9B | GPU（24 GB 或以上） | 以适中成本获得最高准确度 |
| `vllm-sr/Decision-2.0-Vega-27B` | 27B | 一块 64 GB 或以上的 GPU | 整体最准确 |

Decision 1.0 模型（`vllm-sr/Decision-1.0-Kai-0.6B`、`-Lex-0.6B`、`-Route-0.6B`、`-Eos-0.8B`、
`-Sol-2B`、`-Nox-4B`、`-Lux-9B`）也已内置，回答同类问题。Vela 2.0（`vllm-sr/Vela-2.0-0.3B`、
`-0.8B`、`-4B`、`-9B`）支持选择多个标签或标出文本片段的问题；它是私有预览，需要具备访问权限的 Hugging Face token。
在 CPU 上运行 0.3B。在 GPU 上，较大的几档可读取最多 16,384 个 token 的输入（0.3B 为 8,192）：其中 0.8B 成本最低，4B 和 9B 最准确。

`vllm-sr-runtime models` 会列出每个内置模型及其固定的 revision。

## 硬件 {#hardware}

| 硬件 | 状态 | 用法 |
| --- | --- | --- |
| CPU | 已验证 | 每个路由器镜像都能开箱即用地在 CPU 上运行模型。 |
| AMD Instinct MI300X、MI325X | 已验证 | 设置 `device: rocm:0`。`vllm-sr serve --platform amd` 和 `extproc-rocm` 镜像自带 ROCm 版 PyTorch。 |
| NVIDIA GPU | 可用，尚未验证 | 设置 `device: cuda:0`。`vllm-sr serve --platform nvidia` 自带 CUDA 版 PyTorch。 |
| Intel GPU | 可用，尚未验证 | 设置 `device: xpu:0`，并把运行时安装在 XPU 版 PyTorch 旁边。 |
| Apple 芯片 | 可用，尚未验证 | 设置 `device: mps`，并在 macOS 上安装运行时。 |

在 AMD GPU 上，路由器镜像自带运行时经过验证的软件栈：ROCm 7.2 版 PyTorch 2.12、FLA 0.5.2，以及为 ROCm 构建的 `causal-conv1d` 1.7.0。
其中的 `causal-conv1d` 也包含 MI200 和 MI350 GPU 的代码，因此用到它的模型在这些 GPU 上也能运行，但只有 MI300X 和 MI325X 经过验证。
内置模型的 GPU 参考答案已在该栈上核验，每个模型加载时都会与它们对照自检。换用其他 PyTorch、ROCm 或内核构建时，
模型可能无法通过自检，或报告 [`unverified`](model-runtime/troubleshooting.md#a-ready-models-self-check-says-unverified)。
如果某个模型的参考答案需要在该栈上重新记录，其模型族的[记录](https://github.com/vllm-project/semantic-router/tree/main/src/model-runtime/docs/records)
会说明这一点，并给出它与发布版答案的一致率。

`device: auto`（默认）选择第一个有足够空闲显存的已验证 GPU，否则使用 CPU。
你显式指定的 GPU 必须存在，否则模型会带着明确的原因加载失败，而不是悄悄在 CPU 上运行。

### 需要多少内存 {#how-much-memory}

按 CPU 上每个参数约 4 字节、GPU 上每个参数约 2 字节估算，再为请求留出余量：
一个 307M 的任务模型在 CPU 上约需 1.3 GB，Decision 2.0 Lux-9B 在 GPU 上约需 18 GB。
运行时会拒绝加载放不进设备的模型并说明原因。要让大模型彼此隔离，给它们各自的 process
（见[与路由器一起运行](model-runtime/deploy.md#group-models-into-processes)）。

## 你自己的模型 {#your-own-models}

Hugging Face 的 ModernBERT 和 mmBERT 分类器、token 分类器和 embedding 模型，加载方式与内置 Vela 模型相同：
提供带 `revision` 的 Hub 仓库，或本地副本的绝对路径。其他架构的模型需要家族插件；
见[添加你自己的模型家族](model-runtime/plugins.md)。

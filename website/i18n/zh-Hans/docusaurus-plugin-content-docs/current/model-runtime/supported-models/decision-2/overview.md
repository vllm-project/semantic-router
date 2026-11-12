---
title: Decision 2.0
sidebar_label: Decision 2.0
description: 六个 Decision 2.0 规格，从 CPU 上的 Kai-0.6B 到一块大 GPU 上的 Vega-27B，零样本回答关于文本的 choice、noul 和 score 问题。
is_mtpe: false
translation:
  source_commit: "bc5c6bfb747ffe9c5f35d98b203427dada8c4c56"
  source_file: "docs/model-runtime/supported-models/decision-2/overview.md"
  outdated: false
---

# Decision 2.0 {#decision-20}

Decision 2.0 零样本回答你关于文本的 `choice`、`noul` 和 `score` 问题。选择对你的问题足够准确的最小规格：Kai-0.6B 可以在 CPU 上运行，更大的规格运行在 GPU 上。

```bash title="Start the smallest size on a CPU"
vllm-sr serve vllm-sr/Decision-2.0-Kai-0.6B --engine --platform cpu
```

## 规格 {#sizes}

<!-- BEGIN GENERATED MODEL TABLE -->

| 模型 | 参数量 | 回答的问题类型 | Span | 输入 | 显存 |
| --- | --- | --- | --- | --- | --- |
| [Decision-2.0-Kai-0.6B](./kai-0-6b) | 0.597B | choice, noul, score | — | text | 4 GiB |
| [Decision-2.0-Eos-0.8B](./eos-0-8b) | 0.753B | choice, noul, score | — | text | 4 GiB |
| [Decision-2.0-Sol-2B](./sol-2b) | 1.88B | choice, noul, score | — | text | 8 GiB |
| [Decision-2.0-Nox-4B](./nox-4b) | 4.21B | choice, noul, score | — | text | 16 GiB |
| [Decision-2.0-Lux-9B](./lux-9b) | 7.94B | choice, noul, score | — | text | 24 GiB |
| [Decision-2.0-Vega-27B](./vega-27b) | 29.4B | choice, noul, score | — | text | 72 GiB |

<!-- END GENERATED MODEL TABLE -->

| 模型 | 适合运行于 | 适合 |
| --- | --- | --- |
| Kai-0.6B | CPU（16 核上两个问题约 0.2 秒）或任意 GPU | 快速、简单的路由问题；入门的默认选择 |
| Eos-0.8B | CPU 或任意 GPU | 成本相近、稍难一些的问题 |
| Sol-2B | GPU；低流量时也可用 CPU | 需要更多判断的问题 |
| Nox-4B | GPU | 细致的问题和选项众多的问题 |
| Lux-9B | GPU（24 GB 或以上） | 成本适中下最准确 |
| Vega-27B | 一块 64 GB 或以上的 GPU | 总体最准确 |

:::warning 不支持 span
Decision 2.0 不回答 span 问题。当某个 Decision 2.0 规格作为决策模型时，把 `pii` 和 `hallucination` 绑定到支持 span 的部署；每个模型页面都给出了相应配置。
:::

## 运行方式 {#how-it-runs}

- **精度：** GPU 上为 BF16，CPU 上为 FP32。
- **共享上下文默认开启。** 对同一输入提出很多问题的请求只计算一次输入，在 Vega-27B 上对 128 个问题最多快 3.4 倍；其答案与发布的运行时一致（[引擎参数](../../deploy/engine-args.md#shared-context)）。
- **批处理**会合并并发请求。在 ROCm 上，合并的行改变的决策数，与把同样的问题分开而不是一起提问时一样多（[记录](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/rocm-mi325x-batching.md)）。
- **NVIDIA 上更快的内核：** 在 Ampere 及更新的 GPU 上，有 Triton 时它运行融合的逐元素内核，快 11% 到 48%，且没有改变任何决策（[记录](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/cuda-fused-approximate.md)）。
- Qwen3.5 规格（除 Kai-0.6B 外的全部）需要 CPU LAPACK，而 ROCm 镜像的 PyTorch 没有：在 CPU 上请用 CPU 镜像运行它们。

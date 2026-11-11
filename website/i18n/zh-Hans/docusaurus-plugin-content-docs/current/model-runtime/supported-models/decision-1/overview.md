---
title: Decision 1.0
sidebar_label: Decision 1.0
description: 七个 Decision 1.0 模型，三个 0.6B 编码器和四个 Qwen3.5 解码器，零样本回答关于文本的 choice、noul 和 score 问题。
is_mtpe: false
translation:
  source_commit: "bc5c6bfb747ffe9c5f35d98b203427dada8c4c56"
  source_file: "docs/model-runtime/supported-models/decision-1/overview.md"
  outdated: false
---

# Decision 1.0 {#decision-10}

Decision 1.0 是第一代决策模型：三个 0.6B 编码器（Kai、Lex 和 Route，基于 Vela 编码器骨干网络），以及四个基于 Qwen3.5 的解码器（Eos-0.8B、Sol-2B、Nox-4B 和 Lux-9B）。它们零样本回答你关于文本的 `choice`、`noul` 和 `score` 问题。

```bash title="Start an encoder on a CPU"
vllm-sr serve vllm-sr/Decision-1.0-Kai-0.6B --engine --platform cpu
```

## 规格 {#sizes}

<!-- BEGIN GENERATED MODEL TABLE -->

| 模型 | 参数量 | 回答的问题类型 | Span | 输入 | 显存 |
| --- | --- | --- | --- | --- | --- |
| [Decision-1.0-Kai-0.6B](./kai-0-6b) | 0.572B | choice, noul, score | — | text | 4 GiB |
| [Decision-1.0-Lex-0.6B](./lex-0-6b) | 0.572B | choice, noul, score | — | text | 4 GiB |
| [Decision-1.0-Route-0.6B](./route-0-6b) | 0.572B | choice, noul, score | — | text | 4 GiB |
| [Decision-1.0-Eos-0.8B](./eos-0-8b) | 0.753B | choice, noul, score | — | text | 4 GiB |
| [Decision-1.0-Sol-2B](./sol-2b) | 1.88B | choice, noul, score | — | text | 8 GiB |
| [Decision-1.0-Nox-4B](./nox-4b) | 4.21B | choice, noul, score | — | text | 16 GiB |
| [Decision-1.0-Lux-9B](./lux-9b) | 7.94B | choice, noul, score | — | text | 24 GiB |

<!-- END GENERATED MODEL TABLE -->

:::tip
新的配置请优先使用 [Decision 2.0](../decision-2/overview.md) 或 [Decision 3.0](../decision-3/overview.md)。Decision 1.0 仍然内置，因此引用它的配置可以继续工作。
:::

## 运行方式 {#how-it-runs}

- **与发布的运行时逐字节一致。** 在每个请求单独运行时，该家族在 Router 的 ROCm 镜像上对全部七个包的回答都与各包自带的运行时逐字节一致（[一致性记录](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/decision1-parity.md)）。
- **精度：** 编码器在所有设备上都以 FP32 运行；解码器在 GPU 上为 BF16，在 CPU 上为 FP32。
- **x86 CPU 上更快的编码器：** Kai、Lex 和 Route 用 oneDNN 在预先打包的 FP32 线性层副本上运行合并的前向计算，快 1.5–1.7 倍，且没有改变任何决策。
- 对解码器**关闭共享上下文**：它们的记录显示决策会发生变化，因此 `--enable-shared-context` 需要显式开启（[引擎参数](../../deploy/engine-args.md#shared-context)）。
- **不支持 span。** 当 Decision 1.0 模型作为决策模型时，把 `pii` 和 `hallucination` 绑定到支持 span 的部署。

---
title: Vela 2.0
sidebar_label: Vela 2.0
description: 上一代 Router 模型，从 CPU 上的 0.3B 到 GPU 上的 9B，回答 set 和 span 问题并带有校准阈值。
is_mtpe: false
translation:
  source_commit: "bc5c6bfb747ffe9c5f35d98b203427dada8c4c56"
  source_file: "docs/model-runtime/supported-models/vela-2/overview.md"
  outdated: false
---

# Vela 2.0 {#vela-20-family}

Vela 2.0 是上一代 Router 模型家族。除了 `choice`、`noul` 和 `score`，它还回答选取多个标签的问题（`set`）和标记文本 span 的问题（`span`），Router 可以依据这两种答案路由。每个规格都为内置问题提供校准阈值。

```bash title="Restore the previous default on a CPU"
vllm-sr serve vllm-sr/Vela-2.0-0.3B --platform cpu
```

## 规格 {#sizes}

<!-- BEGIN GENERATED MODEL TABLE -->

| 模型 | 参数量 | 回答的问题类型 | Span | 输入 | 显存 |
| --- | --- | --- | --- | --- | --- |
| [Vela-2.0-0.3B](./vela-2-0-3b) | 0.309B | choice, noul, score, set, span | router | text | 4 GiB |
| [Vela-2.0-0.8B](./vela-2-0-8b) | 0.756B | choice, noul, score, set, span | router, broad | text | 4 GiB |
| [Vela-2.0-4B](./vela-2-4b) | 4.21B | choice, noul, score, set, span | router, broad | text | 16 GiB |
| [Vela-2.0-9B](./vela-2-9b) | 7.95B | choice, noul, score, set, span | router, broad | text | 24 GiB |

<!-- END GENERATED MODEL TABLE -->

- **在 CPU 上请运行 0.3B**，一个最多读取 8,192 个 token 的 ModernBERT 编码器。0.8B 是解码器：在 CPU 上一个请求需要数秒。
- **在 GPU 上**，更大的规格读取最多 16,384 个 token 的输入：其中 0.8B 成本最低，4B 和 9B 最准确。
- **Span head：** 每个规格都有 router span head，它用自己现成的问题回答 [`pii`](../../../tutorials/signal/application/pii.md) 和 [`hallucination`](../../../tutorials/signal/application/hallucination.md) 信号；解码器规格还增加了 broad head。

[选择模型](../../get-started/choose-a-model.md#choose-a-size)给出了每个规格的准确率、延迟和校准阈值。

## Vela 2.0 0.3B，之前的默认模型 {#vela-20}

把 `vllm-sr/Vela-2.0-0.3B` 设为 `primary` 部署的制品即可恢复它，校准阈值也随之切换（jailbreak 0.75、domain 0.28、PII 0.01、fact check 0.93、user feedback 0.37）。

当它成为默认模型时，维护者选择了 0.3B，尽管它没有达到为它设定的两个目标（[#4639](https://github.com/vllm-project/semantic-router/issues/4639)）：每个信号的准确率持平或更好，以及 CPU 上的延迟持平或更好。通过 Router 在 [router signal suite](https://huggingface.co/datasets/vllm-sr/router-signal-suite) 上测量（[A/B 记录](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/vela2-router-signals.md)）：

- **领先：** prompt guard（留出集 AUC +0.026；在 E2E 攻击样例上它拦截全部六个攻击，Vela 1.0 Guard 拦截五个）和 safety（留出集 +0.052，并在每个集合上领先）。这些信号共享一个模型。
- **持平：** 留出集和新文件上的 PII 与 hallucination。
- **落后最多：** modality（留出集 AUC −0.180；0.3B 漏掉了大多数要求生成新图像的请求）和 user feedback（准确率留出集 −0.038，新文件 −0.178）。
- **落后：** domain（准确率留出集 −0.037，新文件 −0.088）和 fact check（留出集 AUC −0.101）。
- **CPU 时间：** 每个请求要把问题、其选项和 17 个 PII 标签（至少 560 个 token）送入 307M 参数的模型，而每个 Vela 1.0 模型只读取请求本身。在 12 个 CPU 核上，对于[延迟记录](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/router-latency-cpu.md)中的五个请求信号，请求的中位耗时约为原来的 4.9 倍：

| 12 个 CPU 核上的 Router | p50 | p95 | 每秒请求数 | 并发 16 时 |
| --- | ---: | ---: | ---: | ---: |
| Vela 1.0 专用模型（已移除） | 16 ms | 58 ms | 38.9 | 51.8 |
| Vela 2.0 0.3B（之前的默认模型） | 79 ms | 100 ms | 11.9 | 12.8 |

[#4668](https://github.com/vllm-project/semantic-router/issues/4668) 正在改进 CPU 延迟。在 GPU 上，0.3B 在一块 AMD Instinct MI325X 上回答同样问题的中位耗时约 7 ms（[测量](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/vela2-performance.md#against-the-vela-10-path)）。

:::note
Vela 1.0 专用分类器不回答任何 System One 问题，因此绑定不能指定它们；每个应用信号都询问决策模型。
:::

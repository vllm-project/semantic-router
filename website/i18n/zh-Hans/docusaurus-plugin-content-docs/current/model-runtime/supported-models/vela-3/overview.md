---
title: Vela 3.0
sidebar_label: Vela 3.0
description: Router 自己的决策模型，针对每个内置问题训练过并支持 span；vela3-fast 是默认模型。
is_mtpe: false
translation:
  source_commit: "bc5c6bfb747ffe9c5f35d98b203427dada8c4c56"
  source_file: "docs/model-runtime/supported-models/vela-3/overview.md"
  outdated: false
---

# Vela 3.0 {#vela-30}

Vela 3.0 是 Router 自己的决策模型家族。每个规格都用训练过的适配器回答内置路由问题，并为 PII 和幻觉检查返回 span。**vela3-fast** 是默认决策模型，在 CPU 上运行；更大的规格在 GPU 上运行一个带门控适配器的 Decision 3.0 包。

```yaml title="The default, which you can leave out"
global:
  model_catalog:
    system:
      decision_model: {deployment: primary}  # vllm-sr/vela3-fast on the CPU
```

## 规格 {#sizes}

<!-- BEGIN GENERATED MODEL TABLE -->

| 模型 | 参数量 | 回答的问题类型 | Span | 输入 | 显存 |
| --- | --- | --- | --- | --- | --- |
| [vela3-fast](./vela3-fast) | 1.52B | choice, noul, span | router | text | 12 GiB |
| [vela3-lite](./vela3-lite) | 0.903B | choice, noul, score, span | router | text, image, video | 4 GiB |
| [vela3-mini](./vela3-mini) | 4.79B | choice, noul, score, span | router | text, image, video | 16 GiB |
| [vela3-flash](./vela3-flash) | 8.83B | choice, noul, score, span | router | text, image, video | 32 GiB |
| [vela3-turbo](./vela3-turbo) | 27.6B | choice, noul, score, span | router | text, image, video | 72 GiB |
| [vela3.1-pico-150m](./vela3-1-pico-150m) | 0.153B | choice, noul, span | router | text | 2 GiB |
| [vela3.1-pico-380m](./vela3-1-pico-380m) | 0.381B | choice, noul, span | router | text | 3 GiB |

<!-- END GENERATED MODEL TABLE -->

这些模型都在 [Vela 3.0 合集](https://huggingface.co/collections/vllm-sr/vela-30-6ad1c430eb1718eb4754edcd)中。

## vela3-fast {#vela3-fast}

`vllm-sr/vela3-fast` 是一个 380M 的 UltraBERT 编码器，加上问题所运行的适配器共 0.4B。它的十个适配器带有各自的顶层和 head，因此部署加载 1.5B 个参数。

- **训练过的适配器**回答 Router 的内置问题；你自己的 `choice` 和 `noul` 问题运行在其决策骨干上。
- **Span** 运行在按标签集合和措辞路由的 span 适配器上：个人数据、无依据的陈述、证据、关系和实体，并提供 `pii` 和 `halu` 预设。
- **它不回答 `score` 问题。** `noul` 问题只按其说明读取，因此描述了 `false` 或 `true` 的 `noul` 问题无效。`set` 问题采用其训练过的路由文本之一。
- **精度与内存：** 在每种设备上都是 FP32，加载后约占 7 GB；给 Router 分配 12 GiB。
- 它在 CPU 上的答案与包自带的运行时逐位一致（[一致性记录](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/vela3-parity.md)）。

[选择模型](../../get-started/choose-a-model.md#built-in-signals)给出了它在内置信号上的实测准确率和校准阈值。

## 更大的规格 {#the-larger-sizes}

`vllm-sr/vela3-lite`、`-mini`、`-flash` 和 `-turbo` 在一个固定的 Decision 3.0 包（d3-lite、d3-mini、d3-flash 和 d3-turbo，使用其适配器训练时的修订版本）之上运行门控 LoRA 适配器：

- Router 以这些包训练时的 ID 提出其内置问题，因此它们运行在训练过的适配器上。危害类别以每个类别一个 `hazard:` noul 的形式提问，只有 vela3-lite 的危害适配器能回答。
- 其他每个 `choice`、`noul` 和 `score` 问题都按对应的 d3 包作答。
- 它们回答 `span` 问题，与 vela3-fast 一样按标签集合和措辞路由。一个 span 问题分窗口读取最多 32,768 个 token 的目标；更长的目标会让该问题以 `scan_budget_exceeded` 失败。
- vela3-turbo 先在 d3-turbo 的只读一次状态路径上回答每个请求，再开启适配器重新回答路由问题。
- 带图像的请求在所有适配器关闭的情况下运行，并拒绝 span 问题。
- vela3-mini 和 vela3-flash 按名称运行没有描述的开放 span 标签，而它们的包会先用基座为其写出定义（[一致性记录](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/vela3-d3-parity.md)）。

:::note
只有 vela3-fast 和 Vela 2.0 各规格有校准阈值。更大的 Vela 3.0 规格对没有 `predicate` 的内置问题使用 0.5，并给出 `uncalibrated_preset` 警告。
:::

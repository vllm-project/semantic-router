---
title: Decision 3.0
sidebar_label: Decision 3.0
description: d3 家族零样本回答关于文本、图像和视频的 choice、noul 和 score 问题，规模从 0.6B 到 27B。
is_mtpe: false
translation:
  source_commit: "bc5c6bfb747ffe9c5f35d98b203427dada8c4c56"
  source_file: "docs/model-runtime/supported-models/decision-3/overview.md"
  outdated: false
---

# Decision 3.0 {#decision-30}

Decision 3.0（d3）零样本回答你关于文本、图像和视频的 `choice`、`noul` 和 `score` 问题。一个请求中的每个问题看到的是同样的图像和视频。这个家族运行在 GPU 上，不返回 span。

```bash title="Start the smallest size on an AMD Instinct MI325X"
vllm-sr serve vllm-sr/d3-lite --engine --platform rocm
```

## 规格 {#sizes}

<!-- BEGIN GENERATED MODEL TABLE -->

| 模型 | 参数量 | 回答的问题类型 | Span | 输入 | 显存 |
| --- | --- | --- | --- | --- | --- |
| [d3-edge](./d3-edge) | 0.595B | choice, noul, score | — | text, image, video | 4 GiB |
| [d3-lite](./d3-lite) | 0.853B | choice, noul, score | — | text, image, video | 4 GiB |
| [d3-nano](./d3-nano) | 2.21B | choice, noul, score | — | text, image, video | 8 GiB |
| [d3-mini](./d3-mini) | 4.54B | choice, noul, score | — | text, image, video | 16 GiB |
| [d3-flash](./d3-flash) | 8.39B | choice, noul, score | — | text, image, video | 32 GiB |
| [d3](./d3) | 26.1B | choice, noul, score | — | text, image, video | 72 GiB |
| [d3-turbo](./d3-turbo) | 26.1B | choice, noul, score | — | text, image, video | 72 GiB |
| [d3-ultra](./d3-ultra) | 26.1B | choice, noul, score | — | text, image, video | 72 GiB |

<!-- END GENERATED MODEL TABLE -->

- **d3-edge 到 d3** 是普通规格。它们的答案与包自带的运行时逐位一致（[一致性记录](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/decision3-parity.md)）。
- **d3-turbo 和 d3-ultra** 是 27B 的只读一次状态（state-once）模型，见下文。
- d3-edge 是剪枝后的 d3-lite。d3-lite、d3-mini、d3-flash 和 d3-turbo 也是更大的 [Vela 3.0](../vela-3/overview.md) 规格的基座。

:::warning 不支持 span
d3 不回答 span 问题。当某个 d3 规格作为决策模型时，把 `pii` 和 `hallucination` 绑定到支持 span 的部署，例如 vela3-fast；每个模型页面都给出了相应配置。
:::

## 只读一次状态的模型：d3-turbo 和 d3-ultra {#state-once-models-d3-turbo-and-d3-ultra}

`vllm-sr/d3-turbo` 只读取一次请求的输入，然后从这次读取中回答每个问题，因此对同一输入问很多问题的成本只比问一个多一点（[记录](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/state-once.md)）：

- 除了 `choice`、`noul` 和 `score`，它还在 [System One API](../../../system-one/use/api.md) 上回答 `box` 问题（某物在图像中的位置，或在视频中 `at` 给定那一秒的位置）和 `track` 问题（整个视频每个时刻的一个框，每秒 1 或 2 个）。
- 问题或选项可以带有自己的图像（`assets`）。
- 在一块 AMD Instinct MI325X 上，每个问题默认最多读取 32,768 个 token，使用 `options.long_context` 时最多 262,144 个。

`vllm-sr/d3-ultra` 同样只读取一次请求的输入，回答与 d3-turbo 相同的问题类型（[记录](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/d3-ultra.md)）：

- 纯文本请求会把其问题列表放在状态之前读取，因此每个问题都能看到还问了什么；带图像或视频的请求按 d3-turbo 的方式读取。
- 只有一个问题的请求按包为它设置的温度作答。
- 在一块 AMD Instinct MI325X 上，每个问题默认最多读取 131,072 个 token。

## 量化包 {#quantized-packages}

带 compressed-tensors 格式的 FP8、int4（GPTQ 或 AWQ）或 NVFP4 权重以及 `MODEL_MANIFEST.json` 的 d3 包，以同样的方式提供服务：从本地路径，或从其仓库加上要加载的提交。`vllm-sr/d3` 已发布的包有 `vllm-sr/d3-FP8`、`vllm-sr/d3-W4A16-AWQ`、`vllm-sr/d3-W4A16-GPTQ`、`vllm-sr/d3-NVFP4` 和 `vllm-sr/d3-NVFP4-AWQ`。

| 包 | 与 BF16 答案的一致率 | 显存（BF16：52.2 GB） |
| --- | ---: | ---: |
| FP8 | 99.08% | 31.9 GB |
| GPTQ int4 | 97.98% | 16.4 GB |
| AWQ int4 | 97.76% | 16.5 GB |
| NVFP4 AWQ | 96.81% | 17.2 GB |
| NVFP4 | 96.64% | 17.2 GB |

[量化](../../quantization/overview.md)介绍各种格式、每种格式需要的硬件，以及如何提供服务。

## 精度与硬件 {#precision-and-hardware}

- d3 在每种设备上都以 BF16 运行，即其发布时的精度。
- 它在 AMD Instinct MI300X 和 MI325X 上经过验证；模型页面列出了每个修订版本记录了参考答案的设备。
- ROCm 镜像的 PyTorch 没有 CPU LAPACK，而 Qwen3.5 骨干网络在 CPU 上需要它：请在 GPU 上运行 d3，或用 CPU 镜像在 CPU 上运行。
- 在 GPU 上，d3 把文本行和图像或视频行放在不同的通道中批处理，并把一次合并的前向计算控制在约 0.3 秒的设备时间（[引擎参数](../../deploy/engine-args.md#batching)）。

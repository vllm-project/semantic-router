---
title: 量化
sidebar_label: 量化
description: 模型运行时加载的量化包格式、每种格式在一致率上的代价和在显存与延迟上的收益，以及如何选择。
is_mtpe: false
translation:
  source_commit: "bc5c6bfb747ffe9c5f35d98b203427dada8c4c56"
  source_file: "docs/model-runtime/quantization/overview.md"
  outdated: false
---

# 量化 {#quantization}

量化包用更少的位数保存模型的权重，因此模型占用更少的显存；在具有相应矩阵单元的 GPU 上，回答也更快。模型运行时可以加载三种 compressed-tensors 格式的 Decision 3.0 包。量化包是一个独立的模型：它的答案来自它自己的权重，与 BF16 包的答案接近但并不相同。

```bash title="Serve the FP8 package of d3 on an AMD Instinct MI300X or MI325X"
vllm-sr serve vllm-sr/d3-FP8 --revision 2fb271adbdaf6dc969ca1ca746b61a6a5127ed62 --engine --platform rocm
```

## 格式 {#the-formats}

| 格式 | 权重 | 激活 | d3 的显存 | 与 BF16 答案的一致率 | 需要 |
| --- | --- | --- | ---: | ---: | --- |
| BF16（发布的包） | BF16 | BF16 | 52.2 GB | 100% | 任意 GPU，或 CPU |
| **FP8**（W8A8） | FP8 E4M3，每个输出通道一个 scale | 每个 token 的 FP8，在每次调用时量化 | 31.9 GB | 99.08% | FP8 矩阵单元：AMD Instinct MI300 或 MI350 系列，或计算能力 8.9 及以上的 NVIDIA GPU |
| **int4**（W4A16），GPTQ 或 AWQ | 每 128 个输入列一组的 4 位整数 | BF16 | 16.4 GB（GPTQ）、16.5 GB（AWQ） | 97.98%（GPTQ）、97.76%（AWQ） | 该模型能运行的任意 GPU |
| **NVFP4**（W4A4），普通或 AWQ | 每 16 个一组的 FP4 E2M1，带 FP8 块 scale | FP4，在每次调用时按块量化 | 17.2 GB | 96.64%（NVFP4）、96.81%（NVFP4 AWQ） | FP4 张量核心：NVIDIA Blackwell GPU（计算能力 10.x 或 12.x） |

一致率在[量化记录](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/decision3-quantized.md)的 14,031 个测试请求上测量：即首选选项（或 Noul 的取值方向）与 BF16 包相同的答案所占的比例。在 MI300X 上，准确率从 0.8325（BF16）变为 0.8324（FP8）、0.8310（GPTQ）和 0.8309（AWQ）；在 Blackwell GPU 上，从 0.8323 变为 0.8271（NVFP4）和 0.8292（NVFP4 AWQ）。

## 权衡 {#the-trade-offs}

- **FP8 保留的答案最多**，并节省 39% 的显存。使用 graph 时它比 BF16 快（在 MI300X 上每个请求 36.8 ms 对 46.4 ms）；不使用 graph 时，其逐层启动花费的主机时间比节省的更多。
- **int4 把 FP8 的显存再减半**，但它节省的是显存而不是时间：其层在每次调用时把权重解包为 BF16，因此一个请求比 BF16 稍慢（使用 graph 时 62.0 ms 对 48.2 ms）。在 NVIDIA GPU 上，`--dtype auto` 对最多 256 行的输入运行融合的 int4 内核，缩小这一差距。
- **NVFP4 最快、最小**，前提是 GPU 有 FP4 张量核心：在 RTX PRO 6000 Blackwell 上每个请求 23.2 ms，BF16 为 62.6 ms，占用 17.2 GB。它在三者中改变的答案最多。

:::note
在 `--dtype auto` 下，量化包还会运行其记录所允许的近似融合内核（norm、残差加法和门控）；BF16 包在任何 dtype 下都保持精确内核。显式的 `--dtype` 会让量化包只使用精确内核运行。
:::

## 如何选择 {#how-to-choose}

1. **BF16 包放得下并且足够快？** 保留它：它给出发布时的答案。
2. **放不下，或者你想在 AMD Instinct MI300 或 MI350、NVIDIA Ada、Hopper 或 Blackwell GPU 上留出更多余量？** 使用 **FP8**。
3. **你需要在其他 GPU 上占用最小？** 使用 **int4**，GPTQ 或 AWQ 均可；二者的一致率相差不到 0.25 个百分点。
4. **你在 NVIDIA Blackwell 上提供服务，且最看重延迟？** 使用 **NVFP4**，并先在你自己的问题上测量其一致率。

无论选择哪种，上线前都要在有代表性的请求上把量化包与 BF16 包进行比较，并重新检查规则的谓词：量化包的概率会略有变化，在阈值附近决策可能翻转。

## 什么不是量化 {#what-quantization-is-not}

- **`--dtype`** 以另一种精度（`float32`、`bfloat16` 或 `float16`）、用参考内核运行同样的权重；它不会加载量化包。参见[引擎参数](../deploy/engine-args.md#precision)。
- **降精度副本**是模型线性层的更快副本，由合并请求的批次使用，前提是记录显示没有任何决策改变：在 x86 CPU 上，Vela 2.0 0.3B 和 Decision 1.0 编码器使用预先打包的 FP32 层。它们默认开启，不需要任何包。

## 下一步 {#next}

- [按硬件和模型的支持情况](./support.md)：哪种设备运行哪种格式，哪些模型有量化包。
- [提供量化包服务](./enable.md)：配置、检查，以及你可能看到的错误。

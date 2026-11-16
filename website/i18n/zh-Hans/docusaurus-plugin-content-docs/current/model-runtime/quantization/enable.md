---
title: 提供量化包服务
sidebar_label: 提供量化包服务
sidebar_position: 2
description: 从仓库或本地副本为 Decision 3.0 模型的 FP8、int4 或 NVFP4 包提供服务，并检查它是否加载成功。
is_mtpe: false
translation:
  source_commit: "bc5c6bfb747ffe9c5f35d98b203427dada8c4c56"
  source_file: "docs/model-runtime/quantization/enable.md"
  outdated: false
---

import Tabs from '@theme/Tabs'
import TabItem from '@theme/TabItem'

# 提供量化包服务 {#serve-a-quantized-package}

量化包与其他模型一样提供服务：指定其仓库和要加载的提交，或本地副本的绝对路径。没有需要打开的开关；运行时从包的 `config.json` 读取格式，并检查设备能否运行它。

## 为你的 GPU 选择包 {#pick-the-package-for-your-gpu}

<Tabs groupId="quantization">
<TabItem value="fp8" label="FP8" default>

适用于 AMD Instinct MI300 或 MI350 系列 GPU，或计算能力 8.9 及以上的 NVIDIA GPU：

```bash
vllm-sr serve vllm-sr/d3-FP8 --revision 2fb271adbdaf6dc969ca1ca746b61a6a5127ed62 --engine --platform rocm
```

</TabItem>
<TabItem value="int4" label="int4">

适用于任何能运行 Decision 3.0 的 GPU：

```bash
vllm-sr serve vllm-sr/d3-W4A16-GPTQ --revision 93aa927c5bbf184770ecd144972bed9a1450a4bd --engine --platform cuda
```

</TabItem>
<TabItem value="nvfp4" label="NVFP4">

适用于 NVIDIA Blackwell GPU（计算能力 10.x 或 12.x）：

```bash
vllm-sr serve vllm-sr/d3-NVFP4 --revision 07fa5a04859155eb99a034e2e424d423dd50ce1d --engine --platform cuda
```

</TabItem>
</Tabs>

`--revision` 接受完整提交、分支或标签；CLI 在启动前把它解析一次，得到存入当前部署的提交。[按硬件和模型的支持情况](./support.md)列出了每个已发布的包及其记录测量时的修订版本。

## 在 Router 中使用 {#use-it-from-the-router}

把包声明为一个部署，带上要加载的 40 字符提交，然后在它回答的地方引用它：

```yaml title="config.yaml"
global:
  model_catalog:
    deployments:
      d3-fp8:
        provider: model_runtime
        artifact: vllm-sr/d3-FP8
        revision: 2fb271adbdaf6dc969ca1ca746b61a6a5127ed62
        device: rocm:0
      spans:
        provider: model_runtime
        artifact: vllm-sr/vela3-fast
    bindings:
      pii: spans
      hallucination: spans
    system:
      decision_model: {deployment: d3-fp8}
```

- 非内置的包需要 `revision`，即一个 40 字符的提交；Router 拒绝在 YAML 中使用分支或标签。
- Decision 3.0 不回答 span 问题，因此 `pii` 和 `hallucination` 保留在支持 span 的部署上。
- `device: auto` 会跳过缺少该格式矩阵单元的设备。指定 GPU（`rocm:0`、`cuda:0`）则让无法运行该包的设备立即失败并给出原因。

要改为提供本地副本，把包目录的绝对路径作为 `artifact`，并省略 `revision`。你自己运行的运行时接受同样的包：

```bash
vllm-srun serve vllm-sr/d3-FP8 --revision 2fb271adbdaf6dc969ca1ca746b61a6a5127ed62 --device rocm:0 --host 0.0.0.0 --port 8100
```

## 检查是否加载成功 {#check-that-it-loaded}

- worker 上的 `GET /v1/models` 显示包的修订版本、标识（`model_sha256`）及其自检，自检报告 `unverified`：已发布的包没有记录的参考答案。
- worker 的 `/metrics` 上的 `vllm_srun_model_memory_bytes` 显示其已加载权重占用的字节数：FP8 约 31.9 GB，BF16 为 52.2 GB。
- 在切换流量之前，就几十个有代表性的请求问它你的规则所问的问题，并把答案与 BF16 包进行比较。

## 如果被拒绝 {#if-it-is-refused}

| 原因中写着 | 这样做 |
| --- | --- |
| FP8 weights need an NVIDIA GPU of compute capability 8.9 or later or an AMD Instinct MI300 or MI350 series GPU | 在这样的 GPU 上提供 FP8 包，或改为提供 int4 或 BF16 包。运行时不会重试。 |
| NVFP4 packages need FP4 tensor cores; serve NVFP4 packages on a Blackwell GPU | 设备是 CPU、ROCm GPU 或没有 FP4 张量核心的 NVIDIA GPU，或者其 PyTorch 不支持 FP4 张量、缺少 Triton；原因会说明是哪一项。在带有较新 PyTorch 和 Triton 的 Blackwell GPU 上提供它，或改为提供 FP8、int4 或 BF16 包。 |
| only compressed-tensors int4 (pack-quantized), FP8 (float-quantized) and NVFP4 (nvfp4-pack-quantized) weights are supported | 包的 `quantization_config` 是其他方法或格式。用其中一种格式、一个权重组重新量化。 |
| needs about X GiB, Y GiB free | GPU 的可用显存少于包的文件加上激活预留。释放该 GPU 的显存、换一块 GPU，或使用更小的包。 |

[故障排查](../deploy/troubleshooting.md#the-runtime-reports-failed)列出了其他加载错误。

## 使用你自己的包 {#bring-your-own-package}

运行时可以加载这样的 Decision 3.0 包：其 `config.json` 带有一个只包含一个权重组的 compressed-tensors `quantization_config`，并采用以下方案之一：

- **int4：** `pack-quantized`，按输入列分组（组大小为 8 的倍数）的 4 位整数权重，对称（GPTQ）或带零点的非对称（AWQ），激活为 BF16。
- **FP8：** `float-quantized`，每个输出通道一个 scale 的 FP8 E4M3 权重，激活在每次调用时按 token 量化为 FP8（`FP8_DYNAMIC`）。
- **NVFP4：** `nvfp4-pack-quantized`，权重和激活都是每 16 个一组的 FP4 E2M1，带张量全局 scale 下的 FP8 E4M3 块 scale，权重为静态，激活按块动态量化。

包还需要其 `MODEL_MANIFEST.json`，就像已发布的包那样。其他任何形式都会被拒绝，而不会以 BF16 加载。

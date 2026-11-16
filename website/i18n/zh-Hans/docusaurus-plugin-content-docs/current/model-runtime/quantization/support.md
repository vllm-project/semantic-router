---
title: 按硬件和模型的量化支持
sidebar_label: 按硬件和模型的支持
sidebar_position: 1
description: 哪些设备运行 FP8、int4 和 NVFP4 包、它们在那里使用哪些内核，以及哪些模型有量化包。
is_mtpe: false
translation:
  source_commit: "bc5c6bfb747ffe9c5f35d98b203427dada8c4c56"
  source_file: "docs/model-runtime/quantization/support.md"
  outdated: false
---

# 按硬件和模型的量化支持 {#quantization-support-by-hardware-and-model}

每种格式都需要设备可能不具备的矩阵单元，运行时会在加载任何权重之前进行检查：设备无法运行的包会被拒绝并给出原因，绝不会改为以 BF16 加载。只有 Decision 3.0 包可以包含量化权重。

## 按硬件 {#by-hardware}

| 设备 | FP8（W8A8） | int4（W4A16） | NVFP4（W4A4） |
| --- | --- | --- | --- |
| AMD Instinct MI300X、MI325X（gfx942） | 是（E4M3 FNUZ） | 是 | 否 |
| AMD Instinct MI350 系列（gfx950） | 是（E4M3） | 是 | 否 |
| 其他支持 BF16 的 AMD GPU | 否 | 是，未验证 | 否 |
| NVIDIA Ampere（计算能力 8.0、8.6） | 否 | 是，使用融合内核 | 否 |
| NVIDIA Ada、Hopper（计算能力 8.9、9.0） | 是 | 是，使用融合内核 | 否 |
| NVIDIA Blackwell（计算能力 10.x、12.x） | 是 | 是，使用融合内核 | 是，需要 Triton |
| CPU | 否 | 未测量 | 否 |

Decision 3.0 在 GPU 上以 BF16 保存其权重，因此不支持 BF16 的 GPU 无法运行它的任何包。

- **FP8** 在 PyTorch 提供带 scale 的 FP8 乘法的地方以 FP8 相乘：ROCm 上的 gfx942 和 gfx950，以及 CUDA 上计算能力 8.9 及以上的 GPU。其他任何设备都会拒绝 FP8 包，并报告“FP8 weights need an NVIDIA GPU of compute capability 8.9 or later or an AMD Instinct MI300 or MI350 series GPU”。
- **int4** 不需要特殊单元：其层把权重解包为 BF16 后以 BF16 相乘。在 NVIDIA GPU 上，`--dtype auto` 对最多 256 行的输入用融合的 Triton 内核直接从打包的权重相乘；ROCm 保持解包后的乘法，那在 ROCm 上更快。
- **NVFP4** 需要 FP4 张量核心（只有 NVIDIA Blackwell GPU 具备）、支持 FP4 张量的 PyTorch，以及用于量化激活的 Triton。其他任何设备都会在加载权重之前拒绝该包，并指出缺少的是哪一项。
- ROCm 镜像的 PyTorch 没有 CPU LAPACK，而 Decision 3.0 的 Qwen3.5 骨干网络在 CPU 上需要它，因此任何 Decision 3.0 包（无论是否量化）都无法在 ROCm 镜像的 CPU 上运行。

<details>
<summary>记录中的内核细节</summary>

- 在 Ada 上，带逐行 scale 的 `torch._scaled_mm` 从 256 行开始会使用较慢的内核。在那里，`--dtype auto` 对 256 行及以上的 FP8 乘积不带 scale 地乘到 FP32 中，再应用 scale；显式的 `--dtype` 保留逐行乘积。
- 在计算能力 12.x 上，最多 512 行的 graph 捕获使用 FlashInfer 的 FP4 GEMM 进行 NVFP4 乘法，这需要 CUDA 12.9 或更高版本的工具包以及 Ninja 来构建其计时内核；没有它们时，运行时保留 cuBLAS 并说明原因。CUDA 镜像保留 cuBLAS。
- 使用 autotune 缓存（`--autotune-cache`）时，第一个进程记录每种形状的 tile，之后的进程直接加载而不再计时，因此它们的答案一致。

[量化记录](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/decision3-quantized.md)包含相应的测量。

</details>

## 按模型 {#by-model}

| 家族 | 量化包 | 说明 |
| --- | --- | --- |
| [Decision 3.0](../supported-models/decision-3/overview.md) | 是：compressed-tensors 格式的 FP8、int4（GPTQ、AWQ）和 NVFP4 | 该家族的每个规格都能加载它们；`vllm-sr/d3` 有已发布的包 |
| [Vela 3.0](../supported-models/vela-3/overview.md) | 否 | 较大的规格运行一个固定的 BF16 Decision 3.0 包加适配器 |
| [Decision 2.0](../supported-models/decision-2/overview.md)、[Decision 1.0](../supported-models/decision-1/overview.md) | 否 | 以其发布时的精度运行 |
| [Vela 2.0](../supported-models/vela-2/overview.md) | 否 | Vela 2.0 0.3B 在 x86 CPU 上运行预先打包的 FP32 副本 |

`vllm-sr/d3`（27B）已发布的包，以及记录测量时的修订版本：

| 包 | 格式 | 修订版本 |
| --- | --- | --- |
| [`vllm-sr/d3-FP8`](https://huggingface.co/vllm-sr/d3-FP8) | FP8 | `2fb271adbdaf6dc969ca1ca746b61a6a5127ed62` |
| [`vllm-sr/d3-W4A16-GPTQ`](https://huggingface.co/vllm-sr/d3-W4A16-GPTQ) | int4，GPTQ（对称，每组 128） | `93aa927c5bbf184770ecd144972bed9a1450a4bd` |
| [`vllm-sr/d3-W4A16-AWQ`](https://huggingface.co/vllm-sr/d3-W4A16-AWQ) | int4，AWQ（非对称，每组 128） | `39ffacd3644dad729b6885ac986e740e8db7e22d` |
| [`vllm-sr/d3-NVFP4`](https://huggingface.co/vllm-sr/d3-NVFP4) | NVFP4 | `07fa5a04859155eb99a034e2e424d423dd50ce1d` |
| [`vllm-sr/d3-NVFP4-AWQ`](https://huggingface.co/vllm-sr/d3-NVFP4-AWQ) | 带 AWQ 平滑的 NVFP4 | `27dbfdc48957f9cbc3b83ac49a58d3434451a0a9` |

:::note
这些包不是内置的，因此没有记录的参考答案：它们的自检报告 `unverified`。每个包的标识会依据其自己的 `MODEL_MANIFEST.json` 进行检查。
:::

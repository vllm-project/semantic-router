---
title: 选择模型、规模和硬件
sidebar_label: 选择模型
sidebar_position: 3
description: 每个任务用哪个模型、需要多大的决策模型，以及在什么硬件上运行它。
is_mtpe: false
translation:
  source_commit: "bc5c6bfb747ffe9c5f35d98b203427dada8c4c56"
  source_file: "docs/model-runtime/get-started/choose-a-model.md"
  outdated: false
---

# 选择模型、规模和硬件 {#choose-a-model-size-and-hardware}

先从任务出发，再选出足够准确的最小模型，以及你手头的硬件。每个内置模型都固定到确切的 Hugging Face 修订版本，因此同一个名称总是加载同样的文件。

:::tip 简短的答案
在测出需要更换的理由之前，保持默认的 **vela3-fast**，在 CPU 上运行。它用训练过的适配器和校准阈值回答每一个内置问题，并为 PII 和幻觉检查返回 span。
:::

## 按任务 {#by-task}

| 你想要 | 默认模型 | 说明 |
| --- | --- | --- |
| 回答路由信号、你自己的问题以及其他每个 System One 判断 | Vela 3.0 vela3-fast | [决策模型](../../system-one/get-started/overview.md)；下面的规格在延迟和准确率之间权衡 |
| 为响应缓存、memory 和 RAG 提供嵌入 | `vllm-sr/Vela-1.0-Encoder-307M-Embedding` | 只用于召回；更小的维度和更少的层用质量换速度 |
| 更大的或带指令的文本嵌入 | `Qwen/Qwen3-Embedding-0.6B` | 0.6B，1,024 维 |

没有配置任何模型时，每个 System One 问题都询问同一个部署 `primary`，它在每个路由阶段内合并兼容的问题。召回功能使用 `vela-embedding` 进行嵌入，它只在其中某个功能需要时才启动。

## 决策模型 {#decision-models}

每个家族都能回答你自己写的问题。它们的区别在于规模、能回答的问题类型，以及是否针对 Router 的内置问题训练过：

| 家族 | 规格 | 回答 | 针对内置问题训练过 | 运行于 |
| --- | --- | --- | --- | --- |
| [Vela 3.0](../supported-models/vela-3/overview.md) | vela3-fast（每个问题 0.4B）、lite 0.8B、mini 4B、flash 9B、turbo 27B | choice、noul、span；较大的规格还回答 score 并读取图像 | 是 | CPU（vela3-fast）或 GPU |
| [Decision 3.0](../supported-models/decision-3/overview.md) | edge 0.6B 到 d3 27B，d3-turbo 和 d3-ultra 27B | 关于文本、图像和视频的 choice、noul、score | 否（零样本） | GPU |
| [Vela 2.0](../supported-models/vela-2/overview.md) | 0.3B、0.8B、4B、9B | choice、noul、score、set、span | 是 | CPU（0.3B）或 GPU |
| [Decision 2.0](../supported-models/decision-2/overview.md) | Kai 0.6B 到 Vega 27B | choice、noul、score | 否（零样本） | CPU（Kai）或 GPU |
| [Decision 1.0](../supported-models/decision-1/overview.md) | Kai、Lex、Route 0.6B 到 Lux 9B | choice、noul、score | 否（零样本） | CPU（0.6B）或 GPU |

在预算有限时回答你自己的问题，Decision 2.0 Kai-0.6B 在 16 个 CPU 核上回答两个问题大约需要 0.2 秒。关于图像和视频的问题，在 GPU 上使用 Decision 3.0 的某个规格。[支持的模型](../supported-models/overview.md)为每个模型提供一页，列出其显存、问题类型和部署片段。

:::warning
Decision 1.0、2.0 和 3.0 不回答 span 问题。当其中一个作为默认判断部署时，把 `pii` 和 `hallucination` 绑定到支持 span 的部署；参见 [Span 能力](../../system-one/get-started/concepts.md#span-capability)。
:::

## 内置信号运行在 vela3-fast 上 {#built-in-signals}

每个应用信号默认使用 `vllm-sr/vela3-fast`（[合集](https://huggingface.co/collections/vllm-sr/vela-30-6ad1c430eb1718eb4754edcd)）。它们共享一个部署 `primary`，该部署合并来自同一路由阶段的兼容问题。它们跟随决策模型；[为信号选择模型](../../installation/choose-a-model-for-a-signal.md)展示了如何为某个信号、某个 recipe 或某条规则指定另一个部署。

- **问题：** 每个信号都以模型训练时的形式提出其预设，因此 domain、fact check、feedback、jailbreak、safety、refusal 和 modality 问题运行在训练过的适配器上；其余问题以零样本方式回答。PII 和幻觉使用 span-set 适配器，因此其 span 保持精确的字符偏移。[支持矩阵](../../system-one/reference/support-matrix.md)列出了每个预设。
- **精度：** 在每种设备上都是 FP32，与该包评估时一致。
- **内存：** 部署加载后约占 7 GB。给 Router 分配 12 GiB，即 Helm chart 的默认上限；更少的内存可能导致模型加载时因内存不足被终止。
- **输入：** 普通的路由判断可能截断为一个 8,192 token 的窗口，证据 span 截断为一个 32,768 token 的窗口。jailbreak、safety、PII 和幻觉需要完整输入；窗口扫描把覆盖范围扩展到扫描预算。覆盖不完整时答案为未知，但 `jailbreak` 或 `pii` 规则会把它计为匹配。参见[长输入](../reference/http-api.md#long-inputs)。
- **阈值：** 没有 `predicate` 的规则采用为 vela3-fast 的分数校准的阈值（见下文）。

在 [router signal suite](https://huggingface.co/datasets/vllm-sr/router-signal-suite) 的测试文件上、以每个信号的校准阈值测量（[记录](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/vela3-router-signals.md)），vela3-fast 在 prompt guard（真正率 0.714 对 0.638）、PII（0.953 对 0.888）、domain、user feedback（平衡准确率 0.807 对 0.592）和 modality（0.858 对 0.539）上领先 Vela 2.0 0.3B，在 safety（0.830 对 0.865）和 fact check（0.542 对 0.698）上落后。它的端到端 CPU 延迟没有测量。

### 阈值 {#thresholds}

每个校准阈值都保持 Vela 1.0 专用模型在该测试集 dev 划分上的工作点：其误报率，或对于置信度下限而言，低于该下限的请求比例。在 vela3-fast 上，它们是 jailbreak 0.88、domain 0.41、PII 0.18、fact check 0.9999、user feedback 0.47、safety 0.52 和 modality 0.63。

- **PII：** span-set 适配器在返回 span 之前会应用它自己按长度设置的阈值，因此 0.18 会接受它返回的每个 span。
- **Fact check：** fact-check 适配器的概率集中在 0 和 1 附近，因此其工作点接近 1。
- **其他模型：** 对某个预设没有校准阈值的模型使用 0.5，配置加载时会报告 `uncalibrated_preset` 警告。
- **你自己的谓词**由你决定。记录把每个 Vela 1.0 的取值映射到 vela3-fast：prompt guard 0.3–0.9 → 0.87–0.91，PII → 0.18，safety 0.5 → 0.52，fact check 0.65–0.95 → 0.9997–0.9999，modality 0.7 → 0.63。

## 选择规模 {#choose-a-size}

决策模型是回答 Router 判断任务以及没有被任何绑定或规则分配给其他部署的 [`systemone` 问题](../../tutorials/signal/systemone.md)的部署。部署只需声明一次，然后引用其确切的键。没有覆盖时，内置默认值是 vela3-fast。

```bash title="Pass a model to change the default deployment"
vllm-sr serve vllm-sr/vela3-lite --platform rocm
```

```yaml title="The same change in config.yaml"
global:
  model_catalog:
    deployments:
      primary:
        provider: model_runtime
        artifact: vllm-sr/vela3-lite
        device: rocm:0
    system:
      decision_model:
        deployment: primary
```

- CLI 把这次部署选择作为新版本保存到当前配置中，`vllm-sr config versions` 会列出它，`vllm-sr config rollback` 可以撤销；之后的启动会保留它，`vllm-sr status` 会显示它。
- Helm chart 的 `decisionModel` 值和 operator 的 `spec.config.decision_model` 设置的是同一个决策模型。
- 部署键是精确且区分大小写的。模型标识、设备和引擎参数属于部署，而不属于 `decision_model`。
- `--platform` 选择运行时平台，而显式的部署放置仍然具有决定权。

通过 Router 在 router signal suite 上与 Vela 1.0 专用模型对比，并针对延迟记录中的五个请求信号测量（[记录](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/vela2-decision-model-sizes.md)）：

| 决策模型 | 硬件 | 相对 Vela 1.0 的留出集准确率 | GPU 上的 p50 | 12 个 CPU 核上的 p50 |
| --- | --- | --- | ---: | ---: |
| `vela3-fast`（默认） | CPU 或 GPU | 在 prompt guard、domain、modality 和 feedback 上领先；在 safety 和 fact check 上落后（[记录](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/vela3-router-signals.md)） | 未测量 | 未测量 |
| `Vela-2.0-0.3B` | CPU 或 GPU | 在 prompt guard 和 safety 上领先，在 domain、modality 和 feedback 上落后 | 6.6 ms | 79 ms |
| `Vela-2.0-0.8B` | CPU 或 GPU | 在 domain、prompt guard、safety、modality 和 hallucination 上领先；在 PII 上落后 | 40.1 ms | 约 3 秒 |
| `Vela-2.0-4B` | GPU，约 17 GB | 除 fact check 外每个信号都领先 | 55.2 ms | 未测量 |
| `Vela-2.0-9B` | GPU，约 32 GB | 每个信号都领先 | 76.5 ms | 未测量 |

- **GPU：** 一块 AMD Instinct MI325X，顺序请求。在并发 16 时，GPU 每秒大约处理 154（0.3B）、25（0.8B）、18（4B）和 13（9B）个请求。
- **在这一工作负载下，4B 和 9B 请使用 GPU。** 这份记录没有测量它们的 CPU 延迟。放置遵循部署和运行时的要求，而不是 CLI 根据模型名称进行的禁止；显式的 GPU 设备必须与所选平台匹配，并且在主机上可用。
- **CPU 上的 0.8B** 是解码器：如表所示，一个请求需要数秒。请在 GPU 上运行它，或在 CPU 上保留 0.3B。
- **每个规格**在 user feedback 的新文件（CrossWOZ）和分布内的 PII 上都落后于 Vela 1.0。带区间的逐信号数据见记录。

每个规格都有自己的校准阈值，切换时，没有 `predicate` 的规则会采用它们：

| 决策模型 | Jailbreak | Domain | PII | Fact check | User feedback |
| --- | ---: | ---: | ---: | ---: | ---: |
| `vela3-fast` | 0.88 | 0.41 | 0.18 | 0.9999 | 0.47 |
| `Vela-2.0-0.3B` | 0.75 | 0.28 | 0.01 | 0.93 | 0.37 |
| `Vela-2.0-0.8B` | 0.71 | 0.38 | 0.07 | 0.994 | 0.34 |
| `Vela-2.0-4B` | 0.63 | 0.45 | 0.05 | 0.9984 | 0.33 |
| `Vela-2.0-9B` | 0.42 | 0.46 | 0.14 | 0.998 | 0.35 |

配置中设置的谓词（例如内置 recipe 的谓词）会被保留；记录给出了它们在每个规格上的对应值（例如 `mom-v1` 的 `prompt_attack` 0.75 在 0.8B 上是 0.71，在 4B 上是 0.63，在 9B 上是 0.42）。绑定可以让某个信号保留在自己的模型上。之前的默认模型 Vela 2.0 0.3B 与 Vela 1.0 专用模型的对比见 [Vela 2.0](../supported-models/vela-2/overview.md#vela-20)。

## 硬件 {#hardware}

| 硬件 | 状态 | 用法 |
| --- | --- | --- |
| CPU | 已验证 | 每个 Router 镜像开箱即可在 CPU 上运行模型。 |
| AMD Instinct MI300X、MI325X | 已验证 | 设置 `device: rocm:0`。`vllm-sr serve --platform rocm` 和 `vllm-sr-rocm` 镜像带有 ROCm 版 PyTorch。 |
| NVIDIA GPU | 已验证 | 设置 `device: cuda:0`。`vllm-sr serve --platform cuda` 带有 CUDA 版 PyTorch。记录：[索引](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/cuda-h200.md)、[镜像冒烟测试](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/cuda-router-image.md)、[黄金答案](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/cuda-golden-answers.md)。没有逐位一致证据的融合内核不进入精确配置。 |
| Intel GPU | 可用，尚未验证 | `device: xpu:0`，需要把运行时安装在 XPU 版 PyTorch 旁边。 |
| 昇腾 NPU（910B、950） | 可用，尚未验证 | `device: npu:0`，需要把运行时安装在 torch_npu 和主机 CANN 工具包旁边。参见[昇腾 NPU](../../installation/ascend-npu.md)。 |
| Apple 芯片 | 本版本仅支持 CPU | 在 macOS 上，docker 目标运行 CPU 镜像，因为 Docker 的 Linux 虚拟机无法使用 GPU。主机 GPU 支持在 [#4636](https://github.com/vllm-project/semantic-router/issues/4636) 中跟踪。 |

`device: auto`（默认）选择第一块有足够可用显存的已验证 GPU，否则选择 CPU：

- 你显式指定的 GPU 必须存在，否则模型会以明确的原因加载失败，而不是悄悄在 CPU 上运行。
- 尚未验证的加速器（`xpu`、`npu`）永远不会被 `auto` 选中；在有这类设备的主机上，请显式指定设备来使用它。

<details>
<summary>模型在其上接受检查的 AMD 软件栈</summary>

在 AMD GPU 上，Router 镜像带有运行时经过验证的软件栈：ROCm 7.2 版 PyTorch 2.12、FLA 0.5.2，以及为 ROCm 构建的 `causal-conv1d` 1.7.0。其 `causal-conv1d` 还包含 MI200 和 MI350 GPU 的代码，因此使用它的模型也能在这些 GPU 上运行，但只有 MI300X 和 MI325X 经过验证。

内置模型的 GPU 参考答案在这个软件栈上进行检查，每个模型在加载时都会与它们比较。使用其他 PyTorch、ROCm 或内核构建时，模型可能无法通过该检查，或报告 [`unverified`](../deploy/troubleshooting.md#a-ready-models-self-check-says-unverified)。如果某个模型的参考答案必须在这个软件栈上重新记录，其家族的[记录](https://github.com/vllm-project/semantic-router/tree/main/src/model-runtime/docs/records)会说明这一点，并给出它与发布答案的一致程度。

</details>

### 需要多少内存 {#how-much-memory}

内存取决于模型家族及其精度（`dtype`），而不仅仅取决于设备：

- FP32 权重每个参数约占 4 字节，BF16 约 2 字节，另外还需要输入、激活和并发所需的内存。一个 307M 的 FP32 任务模型仅权重就需要约 1.3 GB。
- 每个副本都有自己的进程和权重副本；把副本放在合适的设备上（参见[放置与扩展 worker](../deploy/placement-and-scaling.md#place-and-scale-replicas)）。
- Decision 3.0 的[量化包](../quantization/overview.md)用大约三分之一到五分之三的显存容纳同一个模型。

查看模型页面和记录，并针对你的工作负载测量峰值内存。

## 你自己的模型 {#your-own-models}

Hugging Face 的 ModernBERT 和 mmBERT 嵌入模型与内置的 Vela Embedding 加载方式相同：给出带 `revision` 的 Hub 仓库，或本地副本的绝对路径。其他架构的决策模型需要一个能回答 System One 问题的家族插件；参见[添加你自己的模型家族](../extend/plugins.md)。

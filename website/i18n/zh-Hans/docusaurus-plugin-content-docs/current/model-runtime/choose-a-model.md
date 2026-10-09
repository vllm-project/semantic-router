---
title: 选择模型、规模和硬件
sidebar_label: 选择模型
description: 每个任务该用哪个模型，决策模型需要多大，以及用什么硬件运行。
translation:
  source_commit: "83b848c3b6a8a5cf17e488ca84975103a2855b46"
  source_file: "docs/model-runtime/choose-a-model.md"
  outdated: false
---

# 选择模型、规模和硬件

从任务出发。下面每个内置模型都固定到确切的 Hugging Face revision，因此同一个名字始终加载相同的文件。

## 按任务选择 {#by-task}

| 你想要 | 默认模型 | Vela 1.0 专用模型 | 说明 |
| --- | --- | --- | --- |
| 按主题路由（数学、法律、代码……） | Vela 2.0 0.3B | `vllm-sr/Vela-1.0-Encoder-307M-Domain` | 14 个领域 |
| 发现需要事实核查的请求 | Vela 2.0 0.3B | `vllm-sr/Vela-1.0-Encoder-307M-FactCheck` | 它标出需要核查，但不核查事实 |
| 读取用户对上一个回答的反应 | Vela 2.0 0.3B | `vllm-sr/Vela-1.0-Encoder-307M-Feedback` | 满意、需要澄清、回答错误、想要别的、无反馈 |
| 区分文本请求和图片请求 | Vela 2.0 0.3B | `vllm-sr/Vela-1.0-Encoder-307M-Modality` | 只读取书面请求 |
| 发现个人信息 | Vela 2.0 0.3B | `vllm-sr/Vela-1.0-Encoder-307M-PII` | 17 种实体类型，给出精确的字符片段 |
| 拦截提示词注入和越狱 | Vela 2.0 0.3B | `vllm-sr/Vela-1.0-Encoder-307M-Guard` | |
| 标记不安全内容 | Vela 2.0 0.3B | `vllm-sr/Vela-1.0-Encoder-307M-Safety` 或 `-Shield` | Shield 是另一种安全模型 |
| 对照来源检查回答 | Vela 2.0 0.3B | `vllm-sr/Vela-1.0-Encoder-307M-Halu` | 标出回答中无依据的片段 |
| 指出风险类别 | `vllm-sr/Vela-1.0-Encoder-307M-Hazard` | | 12 个独立的危害类别，带已发布的阈值 |
| 用于缓存、记忆、RAG 和工具的 embedding | `vllm-sr/Vela-1.0-Encoder-307M-Embedding` | | 更小的维度和更少的层以质量换速度 |
| 更大或带指令的文本 embedding | `Qwen/Qwen3-Embedding-0.6B` | | 0.6B，1,024 维 |
| 对检索到的文档重排序 | `vllm-sr/Vela-1.0-Encoder-307M-Reranker` | | |
| 把文本、图片和音频放进同一向量空间 | `vllm-sr/Vela-1.0-Omni-Nano` 或 `-Mini` | | 164M / 1.36B；Mini 更准确，并接受更长的文本 |
| 用自然语言提出你自己的问题 | 决策模型（见下一节） | | 0.6B 到 27B |

未配置模型时，表中默认为 Vela 2.0 0.3B 的内置信号共用它的一个部署，每个请求只调用一次（[见下文](#vela-20)）。
Hazard、embedding、重排序和 Omni 使用各自的模型。Vela 1.0 专用模型仍然内置，写明它们即可恢复。
它们都是 307M 的编码器，在 CPU 上运行良好：在 16 个核上，Vela Domain 请求的中位耗时约 12 ms
（[测量记录](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/vela1-performance.md)）。
它们大多最多读取 32,768 个 token，0.3B 读取 8,192 个；上限列在每个模型卡片和 `GET /v1/models` 中。

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
`-0.8B`、`-4B`、`-9B`）支持选择多个标签（`set`）或标出文本片段（`span`）的问题，路由器可以基于这两类回答路由。
它的路由片段头（router span head）还能回答 [`pii`](tutorials/signal/learned/pii.md#vela-20) 和
[`hallucination`](tutorials/signal/learned/hallucination.md#vela-20) 信号，默认的 0.3B 部署正是这样替代了单独的 PII 和 Halu 模型。
在 CPU 上运行 0.3B。在 GPU 上，较大的几档可读取最多 16,384 个 token 的输入（0.3B 为 8,192）：其中 0.8B 成本最低，4B 和 9B 最准确。

`vllm-srun models` 会列出每个内置模型及其固定的 revision。

## 内置信号在 Vela 2.0 0.3B 上运行 {#vela-20}

domain、prompt guard、safety、fact check、user feedback、modality、PII 和 hallucination 信号默认使用
`vllm-sr/Vela-2.0-0.3B`（[合集](https://huggingface.co/collections/vllm-sr/vela-20)）。它们共用一个部署
`primary`，一个请求的所有问题在一次调用中提出。

- **问题：** 每个信号提出模型针对它训练过的问题，并沿用对应 Vela 1.0 模型的标签，因此规则和策略照旧读取答案。
  PII 和 hallucination 使用模型的片段头，片段保留精确的字符偏移。
- **CPU profile：** 在 CPU 上该部署运行 `max_speed`，使用模型权重的打包副本：答案误差约在 0.00001 以内，
  速度约为 `exact` 的 1.6 倍。
- **输入：** 普通路由判断可以按模型输入上限截断。Prompt guard、safety、PII 和 hallucination 要求完整读取输入；
  支持窗口扫描的任务可在扫描预算内覆盖更长的输入。覆盖不完整时返回错误或未知结果。
  具体限制和路由策略见[长输入](./reference.md#long-inputs)。Vela 1.0 的 Guard 和 PII 按窗口扫描最多 32K。
- **阈值：** 模块默认阈值按 0.3B 的分数校准（见下文）。

维护者选择了这个默认值，尽管它没有达到当初设定的两个目标
（[#4639](https://github.com/vllm-project/semantic-router/issues/4639)）：每个信号的准确率持平或更好，
以及 CPU 上的延迟持平或更好。在 [router signal suite](https://huggingface.co/datasets/vllm-sr/router-signal-suite)
上经由路由器测得
（[A/B 记录](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/vela2-router-signals.md)）：

- **领先：** prompt guard（留出集 AUC +0.026；在 E2E 攻击样例上它拦下全部六个攻击，Vela 1.0 Guard 拦下五个）和
  safety（留出集 +0.052，在每个数据集上都领先）。一个模型、一次调用回答所有信号。
- **持平：** PII 和 hallucination 在留出集和新留出集上持平。
- **落后最多：** modality（留出集 AUC −0.180；0.3B 漏掉了大多数要求生成新图片的请求）和 user feedback
  （准确率留出集 −0.038、新留出集 −0.178）。
- **落后：** domain（准确率留出集 −0.037、新留出集 −0.088）和 fact check（留出集 AUC −0.101）。
- **CPU 时间：** 每个请求都要把问题、选项和 17 个 PII 标签（至少 560 个 token）送进一次 3.07 亿参数的前向计算，
  而每个 Vela 1.0 模型只读取请求本身。在 12 个 CPU 核上，针对
  [延迟记录](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/router-latency-cpu.md)中的五个请求信号，
  请求的中位耗时约为原来的 4.9 倍：

| 路由器，12 个 CPU 核 | p50 | p95 | 每秒请求数 | 并发 16 时 |
| --- | ---: | ---: | ---: | ---: |
| Vela 1.0 专用模型（恢复后） | 16 ms | 58 ms | 38.9 | 51.8 |
| Vela 2.0 0.3B（默认） | 79 ms | 100 ms | 11.9 | 12.8 |

[#4668](https://github.com/vllm-project/semantic-router/issues/4668) 继续改进 CPU 延迟。在 GPU 上（`use_cpu: false`），
0.3B 在一块 AMD Instinct MI325X 上回答同样的问题，中位耗时约 7 ms
（[测量记录](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/vela2-performance.md#against-the-vela-10-path)）。

### 阈值 {#thresholds}

每个默认阈值都在该数据集的 dev 划分上保持 Vela 1.0 专用模型的工作点：二分类信号保持误报率，置信度下限保持低于下限的请求比例。
默认值为 prompt guard 0.75、domain 0.28、PII 0.01、fact check 0.93、user feedback 0.37。

- **PII：** 0.3B 的片段头在返回片段前已按各标签自己的阈值筛选，因此 0.01 会接受它返回的每个片段。
- **其他模型：** 运行其他模型且未设置阈值的模块，保持它之前的默认阈值。
- **你自己的规则阈值**（`routing.signals.jailbreak[].threshold` 等）由你设定，很可能是为 Vela 1.0 选的。
  记录给出了每个 Vela 1.0 值在 0.3B 上的对应值：prompt guard 0.3–0.9 → 0.74–0.77、PII → 0.01、safety 0.5 → 0.46、
  fact check 0.95 → 0.93、modality 的 `confidence_threshold` 0.7 → 0.51。

### 恢复 Vela 1.0 专用模型 {#restore-vela-10}

一个配置块即可恢复；你未设置的模块阈值也随之恢复为专用模型的默认值：

```yaml
global:
  model_catalog:
    system:
      safety: models/Vela-1.0-Encoder-307M-Safety
      prompt_guard: models/Vela-1.0-Encoder-307M-Guard
      domain_classifier: models/Vela-1.0-Encoder-307M-Domain
      pii_classifier: models/Vela-1.0-Encoder-307M-PII
      fact_check_classifier: models/Vela-1.0-Encoder-307M-FactCheck
      hallucination_detector: models/Vela-1.0-Encoder-307M-Halu
      feedback_detector: models/Vela-1.0-Encoder-307M-Feedback
```

modality 分类器在 `classifier.model_path` 中写明 `models/Vela-1.0-Encoder-307M-Modality`。

只想恢复一个信号时，只写它那一行。以 user feedback 为例：

```yaml
global:
  model_catalog:
    system:
      feedback_detector: models/Vela-1.0-Encoder-307M-Feedback
```

| 信号 | `global.model_catalog` 下的一行 |
| --- | --- |
| Domain | `system.domain_classifier: models/Vela-1.0-Encoder-307M-Domain` |
| Prompt guard | `system.prompt_guard: models/Vela-1.0-Encoder-307M-Guard` |
| Safety | `system.safety: models/Vela-1.0-Encoder-307M-Safety` |
| Fact check | `system.fact_check_classifier: models/Vela-1.0-Encoder-307M-FactCheck` |
| User feedback | `system.feedback_detector: models/Vela-1.0-Encoder-307M-Feedback` |
| PII | `system.pii_classifier: models/Vela-1.0-Encoder-307M-PII` |
| Hallucination | `system.hallucination_detector: models/Vela-1.0-Encoder-307M-Halu` |
| Modality | `modules.modality_detector.classifier.model_path: models/Vela-1.0-Encoder-307M-Modality` |

配置自己设置的规则阈值保持不变。内置配方的规则按 0.3B 校准，因此改回 Vela 1.0 的信号要连同它的 Vela 1.0
规则阈值一起改回。在 `mom-v1` 中，它们是 prompt guard 0.5、safety 0.5 和 PII 0.7；记录列出了每个配方的值。

专用模型通过明确的任务绑定选择；切换默认决策模型会保留这些覆盖。

## 选择规模 {#choose-a-size}

默认决策绑定引用一个已声明的 deployment，用于 Router 判断任务及未指定覆盖的
[`decision` 问题](tutorials/signal/learned/decision.md)。模型、设备和 profile 仅在该资源中声明；
省略绑定时使用内置 `primary`，即 CPU 上的 Vela 2.0 0.3B：

```bash
vllm-sr serve --platform rocm
```

```yaml
global:
  model_catalog:
    deployments:
      primary:
        provider: model_runtime
        artifact: vllm-sr/Vela-2.0-4B
        device: rocm
    system:
      decision_model:
        deployment: primary
```

`serve` 把这一行作为新版本写入当前生效的配置，`vllm-sr config versions` 会列出它，
`vllm-sr config rollback` 可以撤销；之后的启动会保留它，`vllm-sr status` 会显示它。Helm chart 的
`decisionModel` 值和 operator 的 `spec.config.decision_model` 设置相同绑定。deployment key 必须精确匹配，区分大小写。

通过 Router 在 router signal suite 上与 Vela 1.0 专用模型对比测得，延迟针对延迟记录的五个请求信号
（[记录](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/vela2-decision-model-sizes.md)）：

| 决策模型 | 硬件 | 留出集上相对 Vela 1.0 的准确度 | GPU 上的 p50 | 12 个 CPU 核上的 p50 |
| --- | --- | --- | ---: | ---: |
| `Vela-2.0-0.3B`（默认） | CPU 或 GPU | prompt guard 和 safety 领先，domain、modality 和 feedback 落后 | 6.6 ms | 79 ms |
| `Vela-2.0-0.8B` | CPU 或 GPU | domain、prompt guard、safety、modality 和 hallucination 领先；PII 落后 | 40.1 ms | 约 3 s |
| `Vela-2.0-4B` | GPU，约 17 GB | 除 fact check 外全部领先 | 55.2 ms | 仅 GPU |
| `Vela-2.0-9B` | GPU，约 32 GB | 全部领先 | 76.5 ms | 仅 GPU |
| `Vela-1.0` | CPU 或 GPU | 专用模型本身 | 不适用 | 16 ms |

- **GPU：** 一块 AMD Instinct MI325X，顺序请求。并发 16 时，一块 GPU 每秒约处理 154（0.3B）、
  25（0.8B）、18（4B）和 13（9B）个请求。
- **4B 和 9B 需要 GPU。** 在 `--platform cpu` 或没有该平台 GPU 的主机上，`vllm-sr serve` 会拒绝它们；
  在模型运行时找不到 GPU 的地方，Router 也会拒绝。在 GPU 上，无论模块的 `use_cpu` 如何设置，它们都在 GPU 上运行。
- **CPU 上的 0.8B** 是解码器：如表所示，一个请求需要数秒。请在 GPU 上运行它，或在 CPU 上继续使用 0.3B。
- **每个规模** 在 user feedback 的新留出文件（CrossWOZ）和分布内的 PII 上都落后于 Vela 1.0。带区间的逐信号数据见记录。
- **Decision 1.0 和 Decision 2.0** 都可作为默认判断模型；可用任务取决于其原生能力，不按家族名称限制。
- **专用模型** 通过任务绑定覆盖默认值，并可与默认决策模型同时运行。

每个规模都有自己的模块阈值；切换时，未设置阈值的模块会采用它们：

| 决策模型 | Prompt guard | Domain | PII | Fact check | User feedback |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Vela-2.0-0.3B` | 0.75 | 0.28 | 0.01 | 0.93 | 0.37 |
| `Vela-2.0-0.8B` | 0.71 | 0.38 | 0.07 | 0.994 | 0.34 |
| `Vela-2.0-4B` | 0.63 | 0.45 | 0.05 | 0.9984 | 0.33 |
| `Vela-2.0-9B` | 0.42 | 0.46 | 0.14 | 0.998 | 0.35 |
| `Vela-1.0` | 0.5 | 0.5 | 0.9 | 0.95 | 0.7 |

配置自己设置的规则阈值（例如内置配方的）保持不变；记录把每个阈值映射到每个规模（例如 `mom-v1` 的
`prompt_attack` 0.75 在 0.8B 上是 0.71，在 4B 上是 0.63，在 9B 上是 0.42）。`system.<module>` 行或 binding
会让该信号留在它自己的模型上。

## 硬件 {#hardware}

| 硬件 | 状态 | 用法 |
| --- | --- | --- |
| CPU | 已验证 | 每个路由器镜像都能开箱即用地在 CPU 上运行模型。 |
| AMD Instinct MI300X、MI325X | 已验证 | 设置 `device: rocm:0`。`vllm-sr serve --platform rocm` 和 `vllm-sr-rocm` 镜像自带 ROCm 版 PyTorch。 |
| NVIDIA GPU | 可用，尚未验证 | 设置 `device: cuda:0`。`vllm-sr serve --platform cuda` 自带 CUDA 版 PyTorch。 |
| Intel GPU | 可用，尚未验证 | 设置 `device: xpu:0`，并把运行时安装在 XPU 版 PyTorch 旁边。 |
| Apple 芯片 | 本版本仅支持 CPU | 在 macOS 上 docker 目标使用 CPU 镜像，因为 Docker 的 Linux 虚拟机拿不到 GPU。通过宿主机使用 GPU 的支持见 [#4636](https://github.com/vllm-project/semantic-router/issues/4636)。 |

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
运行时会拒绝加载放不进设备的模型并说明原因。每个模型副本都运行在独立的进程中，可通过 `replicas` 配置设备位置
（见[放置和扩展副本](model-runtime/deploy.md#place-and-scale-replicas)）。

## 你自己的模型 {#your-own-models}

Hugging Face 的 ModernBERT 和 mmBERT 分类器、token 分类器和 embedding 模型，加载方式与内置 Vela 模型相同：
提供带 `revision` 的 Hub 仓库，或本地副本的绝对路径。其他架构的模型需要家族插件；
见[添加你自己的模型家族](model-runtime/plugins.md)。

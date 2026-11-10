---
is_mtpe: false
title: 选择模型、规模和硬件
sidebar_label: 选择模型
description: 每个任务该用哪个模型，决策模型需要多大，以及用什么硬件运行。
translation:
  source_commit: "118fb72eb67a0e7193697e350d7b494dc016631e"
  source_file: "docs/model-runtime/choose-a-model.md"
  outdated: false
---

# 选择模型、规模和硬件 {#choose-a-model-size-and-hardware}

从任务出发。下面每个内置模型都固定到确切的 Hugging Face revision，因此同一个名字始终加载相同的文件。

## 按任务选择 {#by-task}

| 你想要 | 默认模型 | 说明 |
| --- | --- | --- |
| 回答路由信号、你自己的问题以及其他所有 System One 判断 | Vela 3.0 vela3-fast | 即[决策模型](../system-one/overview)；下面的各规格在延迟和准确率之间取舍 |
| 为响应缓存、记忆和 RAG 提供 embedding | `vllm-sr/Vela-1.0-Encoder-307M-Embedding` | 只用于召回；更小的维度和更少的层以质量换速度 |
| 更大或带指令的文本 embedding | `Qwen/Qwen3-Embedding-0.6B` | 0.6B，1,024 维 |

未配置模型时，每个 System One 问题都询问同一个部署 `primary`，它在每个路由阶段内批量处理兼容的问题（[见下文](#built-in-signals)）。召回功能使用 `vela-embedding` 生成 embedding，它只在其中某个功能需要时才启动。vela3-fast 每次决策或证据 span 读取 32,768 个 token，router 问题和其他 span 每个窗口读取 8,192 个。每个模型卡片和 `GET /v1/models` 都列出了上限。

输入上限不代表延迟保证。完整的长输入 embedding 和扫描在少核 CPU 上可能超过信号截止时间。请按实际输入长度和并发量测量，再选择足够的专用 CPU 资源或 GPU，并为该流量调整 worker 线程数。

## 决策模型 {#decision-models}

决策模型回答你自己写的问题，例如“这需要逐步推理吗？”或“这些模型中哪个应该回答？”。选择对你的问题足够准确的最小模型。

| 模型 | 规模 | 适合运行在 | 适合 |
| --- | --- | --- | --- |
| `vllm-sr/Decision-2.0-Kai-0.6B` | 0.6B | CPU（16 核上两个问题约 0.2 秒）或任意 GPU | 快速、简单的路由问题；入门的默认选择 |
| `vllm-sr/Decision-2.0-Eos-0.8B` | 0.8B | CPU 或任意 GPU | 稍难的问题，成本相近 |
| `vllm-sr/Decision-2.0-Sol-2B` | 2B | GPU；低流量时也可用 CPU | 需要更多判断的问题 |
| `vllm-sr/Decision-2.0-Nox-4B` | 4B | GPU | 细致的问题和较多选项 |
| `vllm-sr/Decision-2.0-Lux-9B` | 9B | GPU（24 GB 或以上） | 以适中成本获得最高准确度 |
| `vllm-sr/Decision-2.0-Vega-27B` | 27B | 一块 64 GB 或以上的 GPU | 整体最准确 |

Decision 3.0 模型还能读取图片和视频：请求可以携带每个问题都能看到的 `images` 和 `videos`。内置的有 `vllm-sr/d3`（27B）、`vllm-sr/d3-flash`（9B）、`vllm-sr/d3-mini`（4B）、`vllm-sr/d3-nano`（2B）、`vllm-sr/d3-lite`（0.8B）和 `vllm-sr/d3-edge`（0.6B，剪枝后的 d3-lite），运行在 GPU 上。例如在 AMD Instinct MI325X 上用 `vllm-sr serve vllm-sr/d3-lite --engine --platform rocm` 启动一个。它们回答 `choice`、`noul` 和 `score` 问题，答案与这些包自己的运行时逐位一致（[一致性记录](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/decision3-parity.md)）。带有 compressed-tensors 格式的 FP8、int4（GPTQ 或 AWQ）或 NVFP4 权重以及 `MODEL_MANIFEST.json` 的 d3 包也以同样方式提供服务，可以来自本地路径，或通过 `--revision` 和要加载的 commit 来自其仓库。这些 d3 包是 `vllm-sr/d3-FP8`、`vllm-sr/d3-W4A16-AWQ`、`vllm-sr/d3-W4A16-GPTQ`、`vllm-sr/d3-NVFP4` 和 `vllm-sr/d3-NVFP4-AWQ`。在其[记录](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/decision3-quantized.md)的测试行上，FP8 d3 用 31.9 GB（而非 52.2 GB）保留了 99.1% 的 BF16 答案（FP8 需要 AMD Instinct MI300 或 MI350 系列 GPU，或计算能力 8.9 及以上的 NVIDIA GPU）；int4 d3 用 16.5 GB 保留约 98%，略慢于 BF16；NVFP4 d3 用 17.2 GB 保留约 97%，在 RTX PRO 6000 上回答一个请求需要 23.2 ms，而 BF16 需要 62.6 ms。NVFP4 需要 FP4 张量核心，只有 NVIDIA Blackwell GPU（计算能力 10.x 或 12.x）具备；其他设备会在加载权重之前拒绝该包。

`vllm-sr/d3-turbo`（27B）是只读取一次请求输入、并基于这次读取回答所有问题的 Decision 3.0 模型，所以针对同一输入提出许多问题，几乎不比只问一个更贵（[记录](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/state-once.md)）。除了 `choice`、`noul` 和 `score`，它还回答 `box` 问题（某物在图片中的位置，或在视频中由 `at` 给定的那一秒的位置）和 `track` 问题（整个视频每个时刻的框，每秒 1 或 2 个），问题或选项还可以带有自己的图片（`assets`）。在一块 AMD Instinct MI325X 上，每个问题默认最多读取 32,768 个 token，使用 `options.long_context` 时最多 262,144 个。用 `vllm-sr serve vllm-sr/d3-turbo --engine --platform rocm` 启动它。

`vllm-sr/d3-ultra`（27B）同样只读取一次请求输入，回答与 d3-turbo 相同的问题类型。纯文本请求会把问题列表放在状态前面读取，因此每个问题都能看到还问了什么；带图片或视频的请求按 d3-turbo 的方式读取（[记录](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/d3-ultra.md)）。只有一个问题的请求按包为它设定的温度回答。在一块 AMD Instinct MI325X 上，每个问题默认最多读取 131,072 个 token。用 `vllm-sr serve vllm-sr/d3-ultra --engine --platform rocm` 启动它。

Decision 1.0 模型（`vllm-sr/Decision-1.0-Kai-0.6B`、`-Lex-0.6B`、`-Route-0.6B`、`-Eos-0.8B`、`-Sol-2B`、`-Nox-4B`、`-Lux-9B`）也已内置，回答同类问题。Vela 2.0（`vllm-sr/Vela-2.0-0.3B`、`-0.8B`、`-4B`、`-9B`）还支持选择多个标签（`set`）或标出文本片段（`span`）的问题，Router 可以基于这两类回答路由。它的 router span head 还能用自己现成的问题回答 [`pii`](../tutorials/signal/application/pii) 和 [`hallucination`](../tutorials/signal/application/hallucination) 信号。在 CPU 上请运行 0.3B。在 GPU 上，较大的规格最多读取 16,384 个 token 的输入（0.3B 读取 8,192 个）：其中 0.8B 成本最低，4B 和 9B 最准确。

Vela 3.0 的 `vllm-sr/vela3-fast`（一个 380M 的 UltraBERT 编码器，加上问题所用的适配器为 0.4B；它的十个适配器带有各自的顶层和输出头，所以部署会加载 1.5B 参数）在其决策主干上回答 `choice` 和 `noul` 问题，用训练过的适配器回答 Router 的内置问题，并用按标签集合和措辞路由的 span 适配器回答 `span` 问题（个人数据、无依据陈述、证据、关系、实体），同时提供 `pii` 和 `halu` 预设。`noul` 问题只按其说明读取，所以描述了 `false` 或 `true` 的 noul 问题无效。`set` 问题必须使用它训练过的某个 router 文本；它不回答 `score` 问题。它在 CPU 上的答案与包自己的运行时逐位一致（[一致性记录](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/vela3-parity.md)）。更大的 Vela 3.0 规格 `vllm-sr/vela3-lite`、`-mini`、`-flash` 和 `-turbo` 运行一个固定版本的 Decision 3.0 包（d3-lite、d3-mini、d3-flash 和 d3-turbo，版本为其适配器训练时的版本），并在其上加载带门控的 LoRA 适配器。Router 用这些包训练时的 ID 提出内置问题，因此它们会使用训练过的适配器；危害类别则以每个类别一个 `hazard:` Noul 的方式提出，由 vela3-lite 的危害适配器回答；其他所有 `choice`、`noul` 和 `score` 问题按对应 d3 包的方式回答。它们也回答 `span` 问题，并像 vela3-fast 一样按标签集合和措辞路由。vela3-turbo 先在 d3-turbo 的“状态只读一次”路径上回答每个请求，再打开适配器重新回答 router 问题。带图片的请求会关闭所有适配器运行，并拒绝 span 问题。span 问题分窗口读取最多 32,768 个 token 的目标；更长的目标会让该问题以 `scan_budget_exceeded` 失败。vela3-mini 和 vela3-flash 按名称运行没有描述的开放 span 标签，而它们的包会先用主干为其写出定义（[一致性记录](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/vela3-d3-parity.md)）。

`vllm-srun models` 会打印每个内置模型及其固定的 revision。

## 内置信号运行在 vela3-fast 上 {#built-in-signals}

每个应用信号默认使用 `vllm-sr/vela3-fast`（[合集](https://huggingface.co/collections/vllm-sr/vela-30-6ad1c430eb1718eb4754edcd)）。它们共用一个部署 `primary`，在同一路由阶段内批量处理兼容的问题。一次 API 调用可以携带多个问题；窗口扫描和批量上限仍可能需要多次模型前向计算，后续路由阶段也可能发起额外调用。它们跟随决策模型；[为信号选择模型](../installation/choose-a-model-for-a-signal.md)说明了如何给某个信号、某个配方或某条规则指定其他部署。

- **问题**：每个信号按模型训练时的形式提出其预设问题，因此领域、事实核查、反馈、越狱、安全、拒答和模态问题使用训练过的适配器；其他问题零样本回答。PII 和幻觉使用 span 集合适配器，因此它们的 span 保持精确的字符偏移。[支持矩阵](../system-one/reference#support-matrix)列出了每个预设。
- **精度**：在每种设备上都是 FP32，与包的评测方式一致。
- **内存**：部署加载后约占 7 GB。给 Router 分配 12 GiB（Helm chart 的默认上限）；更少的内存可能在模型加载时因内存不足被终止。
- **输入**：普通的路由判断可能截断到一个 8,192 token 的窗口，证据 span 截断到一个 32,768 token 的窗口。越狱、安全、PII 和幻觉需要完整输入；窗口扫描把覆盖范围扩展到扫描预算。覆盖不完整时答案为未知，但 `jailbreak` 或 `pii` 规则会把它算作匹配。见[长输入](./reference.md#long-inputs)。
- **阈值**：没有 `predicate` 的规则使用针对 vela3-fast 得分校准的阈值（见下文）。

在 [router signal suite](https://huggingface.co/datasets/vllm-sr/router-signal-suite) 的测试文件上、按每个信号的校准阈值测量（[记录](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/vela3-router-signals.md)），vela3-fast 在提示词防护（真阳性率 0.714 对 0.638）、PII（0.953 对 0.888）、领域、用户反馈（平衡准确率 0.807 对 0.592）和模态（0.858 对 0.539）上领先 Vela 2.0 0.3B，在安全（0.830 对 0.865）和事实核查（0.542 对 0.698）上落后。它的端到端 CPU 延迟没有测量。

### 阈值 {#thresholds}

每个校准阈值都在该套件的开发划分上保持 Vela 1.0 专用模型的工作点：它的误报率，或者对置信度下限而言，低于下限的请求比例。在 vela3-fast 上，它们是越狱 0.88、领域 0.41、PII 0.18、事实核查 0.9999、用户反馈 0.47、安全 0.52 和模态 0.63；没有 `predicate` 的规则使用它们。

- **PII**：span 集合适配器在返回 span 之前会应用自己按长度设定的阈值，所以 0.18 会接受它返回的每个 span。
- **事实核查**：事实核查适配器的概率接近 0 和 1，所以它的工作点接近 1。
- **其他模型**：没有某个预设校准阈值的模型使用 0.5，加载配置时会报告 `uncalibrated_preset` 警告。
- **你自己的 predicate** 由你决定。记录把每个 Vela 1.0 值映射到 vela3-fast：提示词防护 0.3–0.9 → 0.87–0.91，PII → 0.18，安全 0.5 → 0.52，事实核查 0.65–0.95 → 0.9997–0.9999，模态 0.7 → 0.63。

### Vela 2.0 0.3B，之前的默认模型 {#vela-20}

把 `vllm-sr/Vela-2.0-0.3B` 设为 `primary` 部署的 artifact 即可恢复它，校准阈值也随之切换（越狱 0.75、领域 0.28、PII 0.01、事实核查 0.93、用户反馈 0.37）。当它成为默认模型时，维护者选择了 0.3B，尽管它没有达到他们设定的两个目标（[#4639](https://github.com/vllm-project/semantic-router/issues/4639)）：在每个信号上准确率持平或更好，以及在 CPU 上延迟持平或更好。通过 Router 在 [router signal suite](https://huggingface.co/datasets/vllm-sr/router-signal-suite) 上测量（[A/B 记录](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/vela2-router-signals.md)）：

- **领先**：提示词防护（留出集 AUC +0.026；在 E2E 攻击样例上拦截全部六个攻击，Vela 1.0 Guard 拦截五个）和安全（留出集 +0.052，并在每个集合上领先）。这些信号共用一个模型。
- **持平**：PII 和幻觉，在留出集和新文件上均持平。
- **落后最多**：模态（留出集 AUC −0.180；0.3B 漏掉了大多数要求生成新图片的请求）和用户反馈（留出集准确率 −0.038，新文件 −0.178）。
- **落后**：领域（留出集准确率 −0.037，新文件 −0.088）和事实核查（留出集 AUC −0.101）。
- **CPU 时间**：在这次测量中，每个请求都带着问题、选项和 17 个 PII 标签（至少 560 个 token）经过这个 307M 参数的模型，而每个 Vela 1.0 模型只读取请求本身。在 12 个 CPU 核上，对于[延迟记录](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/router-latency-cpu.md)中的五个请求信号，请求的中位耗时约为原来的 4.9 倍：

| 12 个 CPU 核上的 Router | p50 | p95 | 每秒请求数 | 并发 16 时 |
| --- | ---: | ---: | ---: | ---: |
| Vela 1.0 专用模型（已移除） | 16 ms | 58 ms | 38.9 | 51.8 |
| Vela 2.0 0.3B（之前的默认模型） | 79 ms | 100 ms | 11.9 | 12.8 |

[#4668](https://github.com/vllm-project/semantic-router/issues/4668) 正在改进 CPU 延迟。在 GPU 上，0.3B 在一块 AMD Instinct MI325X 上回答同样的问题，中位耗时约 7 ms（[测量记录](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/vela2-performance.md#against-the-vela-10-path)）。

Vela 1.0 专用分类器不回答任何 System One 问题，所以绑定不能指定它们；每个应用信号都询问决策模型。

## 选择规模 {#choose-a-size}

决策模型是回答 Router 判断任务、以及没有被绑定或规则指定其他部署的 [`systemone` 问题](../tutorials/signal/systemone)的部署。部署只声明一次，然后写出它的确切键名。没有覆盖时，内置默认值是 vela3-fast。传入模型 artifact 会更新当前生效的默认部署；`--platform` 选择运行时平台，而显式的部署放置设置仍然优先。例如：

```bash
vllm-sr serve vllm-sr/Vela-2.0-4B --platform rocm
```

```yaml
global:
  model_catalog:
    deployments:
      primary:
        provider: model_runtime
        artifact: vllm-sr/Vela-2.0-4B
        device: rocm:0
    system:
      decision_model:
        deployment: primary
```

CLI 把这个部署选择作为新版本保存到当前配置中，`vllm-sr config versions` 会列出它，`vllm-sr config rollback` 可以撤销；之后的启动会保留它，`vllm-sr status` 会显示它。Helm chart 的 `decisionModel` 值和 operator 的 `spec.config.decision_model` 设置的是同一个决策模型。部署键名精确且区分大小写。模型身份、设备和引擎参数属于部署，而不属于 `decision_model`。

通过 Router 在 router signal suite 上、对照 Vela 1.0 专用模型，并针对延迟记录中的五个请求信号测量（[记录](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/vela2-decision-model-sizes.md)）：

| 决策模型 | 硬件 | 留出集准确率（对照 Vela 1.0） | GPU 上的 p50 | 12 个 CPU 核上的 p50 |
| --- | --- | --- | ---: | ---: |
| `vela3-fast`（默认） | CPU 或 GPU | 在提示词防护、领域、模态和反馈上领先；在安全和事实核查上落后（[记录](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/vela3-router-signals.md)） | 未测量 | 未测量 |
| `Vela-2.0-0.3B` | CPU 或 GPU | 在提示词防护和安全上领先，在领域、模态和反馈上落后 | 6.6 ms | 79 ms |
| `Vela-2.0-0.8B` | CPU 或 GPU | 在领域、提示词防护、安全、模态和幻觉上领先；在 PII 上落后 | 40.1 ms | 约 3 s |
| `Vela-2.0-4B` | GPU，约 17 GB | 除事实核查外在每个信号上领先 | 55.2 ms | 未测量 |
| `Vela-2.0-9B` | GPU，约 32 GB | 在每个信号上领先 | 76.5 ms | 未测量 |

- **GPU**：一块 AMD Instinct MI325X，顺序请求。并发 16 时，GPU 每秒约处理 154（0.3B）、25（0.8B）、18（4B）和 13（9B）个请求。
- **在这类负载下，4B 和 9B 请使用 GPU**。这份记录没有测量它们的 CPU 延迟。放置位置取决于部署和运行时的要求，而不是 CLI 按模型名称禁止。显式指定的 GPU 设备必须与所选平台匹配，并且在主机上可用。
- **CPU 上的 0.8B** 是解码器：如表所示，一个请求需要数秒。请在 GPU 上运行它，或在 CPU 上继续使用 0.3B。
- **每个规格**在用户反馈的新文件（CrossWOZ）和分布内的 PII 上都落后于 Vela 1.0。各信号带区间的数值见记录。
- **Decision 1.0、Decision 2.0 和 Decision 3.0** 可以作为默认的判断部署。它们不回答 span 问题，所以请把 `pii` 和 `hallucination` 绑定到支持 span 的部署；见 [Span 能力](../system-one/concepts#span-capability)。

每个规格都有自己的校准阈值，切换时没有 `predicate` 的规则会使用它们：

| 决策模型 | 越狱 | 领域 | PII | 事实核查 | 用户反馈 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `vela3-fast` | 0.88 | 0.41 | 0.18 | 0.9999 | 0.47 |
| `Vela-2.0-0.3B` | 0.75 | 0.28 | 0.01 | 0.93 | 0.37 |
| `Vela-2.0-0.8B` | 0.71 | 0.38 | 0.07 | 0.994 | 0.34 |
| `Vela-2.0-4B` | 0.63 | 0.45 | 0.05 | 0.9984 | 0.33 |
| `Vela-2.0-9B` | 0.42 | 0.46 | 0.14 | 0.998 | 0.35 |

配置自己设置的 predicate（例如内置配方的）保持不变；记录把每个值映射到各个规格（例如 `mom-v1` 的 `prompt_attack` 0.75 在 0.8B 上是 0.71，在 4B 上是 0.63，在 9B 上是 0.42）。绑定会让那一个信号留在它自己的模型上。

## 硬件 {#hardware}

| 硬件 | 状态 | 用法 |
| --- | --- | --- |
| CPU | 已验证 | 每个 Router 镜像开箱即可在 CPU 上运行模型。 |
| AMD Instinct MI300X、MI325X | 已验证 | 设置 `device: rocm:0`。`vllm-sr serve --platform rocm` 和 `vllm-sr-rocm` 镜像自带 ROCm 版 PyTorch。 |
| NVIDIA GPU | 已验证 | 设置 `device: cuda:0`。`vllm-sr serve --platform cuda` 自带 CUDA 版 PyTorch。记录：[索引](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/cuda-h200.md)、[镜像冒烟测试](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/cuda-router-image.md)、[黄金答案](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/cuda-golden-answers.md)、[D1 性能](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/cuda-decision1-performance.md)、[D2 性能](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/cuda-decision2-performance.md)、[Vela 2.0 性能](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/cuda-vela2-performance.md)、[Vela 1.0 A/B](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/cuda-vela1-performance.md)。没有逐位一致证据的融合内核不会进入精确配置。 |
| Intel GPU | 可用，尚未验证 | `device: xpu:0`，运行时需与 XPU 版 PyTorch 一起安装。 |
| 昇腾 NPU（910B、950） | 可运行，尚未验证 | `device: npu:0`，运行时需与 torch_npu 和主机上的 CANN 工具包一起安装。见[昇腾 NPU](../installation/ascend-npu.md)。 |
| Apple silicon | 本版本仅支持 CPU | 在 macOS 上，docker 目标运行 CPU 镜像，因为 Docker 的 Linux 虚拟机无法使用 GPU。主机 GPU 支持见 [#4636](https://github.com/vllm-project/semantic-router/issues/4636)。 |

在 AMD GPU 上，Router 镜像自带运行时所验证的软件栈：ROCm 7.2 版 PyTorch 2.12、FLA 0.5.2，以及为 ROCm 构建的 `causal-conv1d` 1.7.0。其中的 `causal-conv1d` 也包含 MI200 和 MI350 GPU 的代码，所以使用它的模型也能在这些 GPU 上运行，但只有 MI300X 和 MI325X 经过验证。内置模型的 GPU 参考答案在这个软件栈上检查过，每个模型加载时都会与它们比对。使用其他版本的 PyTorch、ROCm 或内核构建时，模型可能无法通过这项检查，或报告 [`unverified`](./troubleshooting.md#a-ready-models-self-check-says-unverified)。如果某个模型的参考答案必须在这个软件栈上重新记录，它所属家族的[记录](https://github.com/vllm-project/semantic-router/tree/main/src/model-runtime/docs/records)会说明这一点，并给出它与已发布答案的一致程度。

`device: auto`（默认值）选择第一个有足够空闲内存且经过验证的 GPU，否则选择 CPU。显式指定的 GPU 必须存在，否则模型会以明确的原因加载失败，而不会悄悄地在 CPU 上运行。尚未验证的加速器（`xpu`、`npu`）永远不会被 `auto` 选中——在有这类加速器的主机上，`auto` 仍然选择 CPU；要使用它们，请显式指定设备。

### 需要多少内存 {#how-much-memory}

内存取决于模型家族及其精度（`dtype`），而不仅仅取决于设备。FP32 权重每个参数约占 4 字节；较低精度的模型约占 2 字节，此外输入、激活值和并发还需要额外内存。因此，一个 307M 的 FP32 任务模型仅权重就需要约 1.3 GB。请查看模型记录，并针对你的负载测量峰值内存。每个副本都有自己的进程和权重副本；请把副本放在合适的设备上（见[与 Router 一起运行](./deploy.md#place-and-scale-replicas)）。

## 你自己的模型 {#your-own-models}

Hugging Face 的 ModernBERT 和 mmBERT embedding 模型与内置的 Vela Embedding 以相同方式加载：给出带 `revision` 的 Hub 仓库，或本地副本的绝对路径。其他架构的决策模型需要一个能回答 System One 问题的家族插件；见[添加你自己的模型家族](./plugins.md)。

---
title: 模型运行时
sidebar_label: 概述
sidebar_position: 1
slug: /model-runtime/overview
description: Router 询问的决策模型和嵌入模型运行在内置模型运行时中。从这里开始。
is_mtpe: false
translation:
  source_commit: "bc5c6bfb747ffe9c5f35d98b203427dada8c4c56"
  source_file: "docs/model-runtime/get-started/overview.md"
  outdated: false
---

# 模型运行时 {#model-runtime}

**内置模型运行时**为 Router 在处理每个请求时所咨询的模型提供服务：回答每个 System One 问题的[决策模型](../../system-one/get-started/overview.md)，以及召回功能使用的嵌入模型。你不需要单独安装它：它随 Router 镜像一起发布，由 Router 为你启动和监管。

```bash title="Serve a decision model and ask it directly"
vllm-sr serve vllm-sr/Decision-2.0-Kai-0.6B --engine --platform cpu
curl -s localhost:8899/v1/systemone/models
```

:::note
运行时并不是运行聊天模型的地方。回答用户的模型仍然留在你的提供方（vLLM、Ollama、托管 API）之后；运行时只为 Router 针对每个请求所询问的小模型提供服务。
:::

![前端、可选的决策引擎与按需加载的模型运行时](/img/architecture/system-one/01-component-composition.svg)

## 三种使用方式 {#three-ways-to-use-it}

| 你想要 | 这样做 | 阅读 |
| --- | --- | --- |
| 使用 Router 的内置功能 | 无需额外操作。Router 为你启动运行时；`--platform rocm` 或 `--platform cuda` 选择支持 GPU 的镜像 | [连接 Router](../deploy/connect-the-router.md) |
| 在多个 Router 之间共享模型，或在另一台机器上运行它们 | 自己启动运行时，并用 `endpoint` 让 Router 指向它 | [附加你自己运行的运行时](../deploy/connect-the-router.md#attach-to-a-runtime-you-run) |
| 从你自己的代码调用模型 | 运行 `vllm-sr serve ARTIFACT --engine` 并发送 HTTP 请求 | [快速开始](./quickstart.md) |

托管的 worker 以 `vllm-srun` 进程的形式运行在 Router 容器内。启动 `vllm-sr serve ARTIFACT --engine` 会保留同一个实例前端和模型管理，只是关闭 Chat 路由。Router 模式可以在处理路由的 Chat 请求的同时，处理原生 System One 请求。[组件架构](../../overview/component-architecture.md)介绍了请求路径。

## 它能提供什么 {#what-it-can-serve}

| 任务 | 内置模型 | 指南 |
| --- | --- | --- |
| 回答每个路由判断：应用信号、你自己的问题、System One 选择器、守卫和评审 | Vela 3.0（默认 vela3-fast）、Decision 3.0、Decision 2.0 和 1.0、Vela 2.0 | [支持的模型](../supported-models/overview.md) |
| 为响应缓存、memory 和 RAG 提供召回 | Vela 1.0 Embedding、Qwen3-Embedding-0.6B | [嵌入](../guides/embeddings.md) |

[选择模型](./choose-a-model.md)帮助你选择规模和硬件。

## 如何启动和失败 {#how-it-starts-and-fails}

- **只加载你用到的。** 内置功能会自动解析其默认模型（[为信号选择模型](../../installation/choose-a-model-for-a-signal.md)）。Router 只为有内容解析到的部署以及显式发布的原生模型启动托管 worker。未使用的部署不加载任何权重。
- **启动时等待所需的模型。** 附加模型的就绪遵循[就绪规则](../deploy/readiness-and-monitoring.md#when-a-model-is-not-ready)。
- **慢模型绝不会卡住请求。** 信号截止时间限定请求的等待时长；未完成的信号遵循其配置的错误或未扫描策略。

## 你可以依赖什么 {#what-you-can-rely-on}

- **固定且经过校验。** 内置模型固定到确切的 Hugging Face 修订版本。每个文件在加载前都会与其记录的 SHA-256 进行比对，并且模型仓库中附带的代码永远不会被运行。
- **吞吐优先，以发布时的精度运行。** 与 vLLM 一样，运行时默认把并发请求合并成批，并以模型发布时的精度运行每个模型。批处理后的答案可能因舍入而略有变化，记录中按模型对此进行了测量；记录无法证明准确率不变的更快设置需要显式开启。参见[引擎参数](../deploy/engine-args.md)。
- **有界的等待。** 慢或不可用的模型通过信号的截止时间和错误策略来处理。已经在运行的模型前向计算可能在调用方超时后继续，并延迟排队的工作。Router 会重启崩溃的运行时。
- **合并的调用。** 来自同一路由阶段的兼容模型工作可以在一次 API 调用中一起发送。这样的调用可能需要多次模型前向计算，后续阶段也可能发出额外的调用。硬件容量允许时，独立的 worker 可以并行作答。
- **可插拔。** 新的模型家族、引擎和硬件后端都是普通的 Python 包。参见[添加你自己的模型家族](../extend/plugins.md)。

## 硬件 {#hardware}

| 硬件 | 状态 |
| --- | --- |
| CPU | 已验证。每个 Router 镜像都能在 CPU 上运行模型 |
| AMD GPU（MI300X、MI325X） | 已验证。`vllm-sr serve --platform rocm` |
| NVIDIA GPU | 已验证。`vllm-sr serve --platform cuda` |
| Intel GPU（`xpu`）、昇腾 NPU（`npu`）、Apple GPU（`mps`） | 可用，尚未验证 |

[选择模型](./choose-a-model.md#hardware)给出了细节，[量化](../quantization/overview.md)介绍各 GPU 能运行的低精度包。

:::tip 从旧版本升级？
candle、ONNX Runtime 和 OpenVINO 后端已经移除。运行 `vllm-sr config migrate` 更新你的配置；参见[从原生绑定迁移](../deploy/migrate.md)。
:::

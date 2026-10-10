---
title: 模型运行时
sidebar_label: 概览
description: 对请求进行分类、防护、向量化和路由的模型都运行在内置模型运行时中。从这里开始。
translation:
  source_commit: "9156d5bc1ed9edff626b95a2b8260a77cb1712c5"
  source_file: "docs/model-runtime/overview.md"
  outdated: false
---

# 模型运行时

**内置模型运行时**提供路由功能需要的判断、分类、embedding、重排序和幻觉检测模型。
它可以管理本地 worker，也可以附加到独立运行的 worker。显式配置了
[外部服务](../installation/runtime/external)的功能则调用相应服务。

运行时不是你的聊天模型运行的地方。回答用户的模型留在你的提供方后面（vLLM、Ollama、托管 API）；
运行时提供路由器调用的判断与辅助模型。每个受管副本在独立的 `vllm-srun` 进程中运行。

用 `vllm-sr serve ARTIFACT --engine` 启动时仍保留同一个实例前端和模型管理，但关闭 Chat 路由。
Router 模式可以同时提供原生 System One 与经过路由的 Chat 请求。

![前端、可选决策引擎与按需模型运行时](/img/architecture/system-one/01-component-composition.svg)

请求路径，以及模型选择与副本调度的区别，见[组件架构](../overview/component-architecture)。

内置功能会自动解析模型默认值。准备配置时，Router 只为实际模型使用方和显式发布的原生模型启动受管 worker；
未使用的 deployment 不加载权重。启动会等待所需受管模型就绪；附加模型遵循
[部署就绪规则](./deploy#when-a-model-is-not-ready)。
之后如果运行时变慢或崩溃，信号截止时间会限制请求等待的时长。
未完成的信号按配置的错误策略或未扫描策略处理。

## 三种用法 {#three-ways-to-use-it}

| 你想要 | 这样做 | 阅读 |
| --- | --- | --- |
| 使用路由器的内置功能 | 不需要额外操作。路由器会替你启动并监管运行时；`--platform rocm` 或 `--platform cuda` 选择支持 GPU 的镜像，每个 worker 的设备由 deployment 决定。 | [与路由器一起运行](model-runtime/deploy.md) |
| 在多个路由器之间共享模型，或在另一台机器上运行它们 | 自己启动一个运行时，并用 `endpoint` 让路由器指向它。 | [与路由器一起运行](model-runtime/deploy.md#attach-to-a-runtime-you-run) |
| 在自己的代码里调用模型 | 运行 `vllm-sr serve ARTIFACT --engine` 并发送 HTTP 请求。 | [快速开始](model-runtime/quickstart.md) |

## 它能提供什么 {#what-it-can-serve}

| 任务 | 内置模型 | 指南 |
| --- | --- | --- |
| 识别领域、判断是否需要事实核查、读取用户反馈、识别请求的输出模态 | Vela 1.0 Domain、FactCheck、Feedback、Modality | [请求分类](model-runtime/guides/classify.md) |
| 发现个人信息 | Vela 1.0 PII | [检测 PII](model-runtime/guides/pii.md) |
| 拦截提示词攻击和不安全内容 | Vela 1.0 Guard、Safety、Shield、Hazard | [提示词攻击与不安全内容](model-runtime/guides/safety.md) |
| 对照来源检查回答 | Vela 1.0 Halu | [幻觉检查](model-runtime/guides/hallucination.md) |
| 语义缓存、记忆、RAG、工具选择、embedding 信号 | Vela 1.0 Embedding、Qwen3-Embedding-0.6B | [Embeddings](model-runtime/guides/embeddings.md) |
| 对检索到的文档重排序 | Vela 1.0 Reranker | [文档重排序](model-runtime/guides/rerank.md) |
| 按图片和音频路由 | Vela 1.0 Omni Nano 和 Mini | [图片与音频](model-runtime/guides/multimodal.md) |
| 用自然语言提出你自己的路由问题 | Decision 2.0、Decision 1.0、Vela 2.0 | [决策模型](model-runtime/guides/decisions.md) |

[选择模型](model-runtime/choose-a-model.md)帮助你挑选规模和硬件。

## 你可以依赖的特性 {#what-you-can-rely-on}

- **固定版本并经过校验。** 内置模型固定到确切的 Hugging Face revision。
  每个文件在加载前都会对照记录的 SHA-256 校验，模型仓库中附带的代码永远不会执行。
- **与发布模型相同的答案。** 默认的 `exact` profile 给出模型发布方测得的答案。
  更快的设置需要显式开启，并会说明可能改变结果。见 [Profiles](model-runtime/profiles.md)。
- **等待有上限。** 模型过慢或不可用时，按信号截止时间和错误策略处理。
  已开始的模型前向计算可能在调用方超时后继续执行，并延迟队列中的请求。
  路由器会重启崩溃的运行时。
- **批量调用。** 同一路由阶段中兼容的模型任务可以合并为一次 API 调用。
  一次调用可能需要多次模型前向计算，后续阶段也可以发起额外调用。
  硬件资源足够时，独立 worker 可以并行回答。
- **可插拔。** 新的模型家族、引擎和硬件后端都是普通的 Python 包。
  见[添加你自己的模型家族](model-runtime/plugins.md)。

## 硬件 {#hardware}

CPU 和 AMD GPU（MI300X、MI325X）已经验证。NVIDIA GPU 可用但尚未验证；
Intel GPU（`xpu`）和 Apple GPU（`mps`）可用但尚未验证。每个路由器镜像都能在 CPU 上运行模型；
AMD 和 NVIDIA 镜像（`vllm-sr serve --platform rocm` 或 `--platform cuda`）还能在 GPU 上运行它们。

## 从旧版本升级？ {#coming-from-an-older-release}

candle、ONNX Runtime 和 OpenVINO 后端已经移除。运行 `vllm-sr config migrate`
更新你的配置；见[从原生绑定迁移](model-runtime/migrate.md)。

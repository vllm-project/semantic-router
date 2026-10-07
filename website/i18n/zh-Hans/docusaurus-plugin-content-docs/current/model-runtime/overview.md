---
title: 模型运行时
sidebar_label: 概览
description: 对请求进行分类、防护、向量化和路由的模型都运行在内置模型运行时中。从这里开始。
translation:
  source_commit: "c94fff6a5d6368a2743b786db5624274053f1ae9"
  source_file: "docs/model-runtime/overview.md"
  outdated: false
---

# 模型运行时

路由器用到的每个模型都运行在**内置模型运行时**中：domain、PII、jailbreak
等信号背后的分类器，语义缓存、记忆和 RAG 背后的 embedding 模型，重排序模型，
幻觉检测器，以及回答路由问题的决策模型。

运行时不是你的聊天模型运行的地方。回答用户的模型留在你的提供方后面（vLLM、Ollama、托管 API）；
运行时提供的是路由器针对每个请求去询问的那些小模型。它以 `vllm-srun` 进程的形式运行在路由器容器内，
或者在你用 `vllm-sr serve <model>` 启动时运行在单独的容器中（engine 模式）。

通常你什么都不用做。某个功能需要模型时，路由器会下载模型、校验每个文件、
启动运行时，并把请求文本发给它。路由所需的模型加载完成后，路由器才开始提供服务。
之后如果运行时变慢或崩溃，请求仍会继续流转：该功能报告“未知”，
路由按你配置的方式回退。

## 三种用法 {#three-ways-to-use-it}

| 你想要 | 这样做 | 阅读 |
| --- | --- | --- |
| 使用路由器的内置功能 | 不需要额外操作。路由器会替你启动并监管运行时；`vllm-sr serve --platform amd` 或 `--platform nvidia` 会把它的模型放到 GPU 上。 | [与路由器一起运行](model-runtime/deploy.md) |
| 在多个路由器之间共享模型，或在另一台机器上运行它们 | 自己启动一个运行时，并用 `endpoint` 让路由器指向它。 | [与路由器一起运行](model-runtime/deploy.md#attach-to-a-runtime-you-run) |
| 在自己的代码里调用模型 | 运行 `vllm-sr serve <model>` 并发送 HTTP 请求。 | [快速开始](model-runtime/quickstart.md) |

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
- **请求从不等待故障模型。** 太慢、仍在重启或已崩溃的模型会让该请求上的对应功能变为“未知”。
  路由器会重启崩溃的运行时，并在此期间继续路由。
- **每个请求的调用很少。** 一个请求的各信号发往同一运行时进程的模型工作会合并为一次调用，
  分布在不同进程中的 CPU 模型并行回答，因此增加信号不会增加往返次数。
- **可插拔。** 新的模型家族、引擎和硬件后端都是普通的 Python 包。
  见[添加你自己的模型家族](model-runtime/plugins.md)。

## 硬件 {#hardware}

CPU 和 AMD GPU（MI300X、MI325X）已经验证。NVIDIA GPU 可用但尚未验证；
Intel GPU（`xpu`）和 Apple GPU（`mps`）可用但尚未验证。每个路由器镜像都能在 CPU 上运行模型；
AMD 和 NVIDIA 镜像（`vllm-sr serve --platform amd` 或 `--platform nvidia`）还能在 GPU 上运行它们。

## 从旧版本升级？ {#coming-from-an-older-release}

candle、ONNX Runtime 和 OpenVINO 后端已经移除。运行 `vllm-sr config migrate`
更新你的配置；见[从原生绑定迁移](model-runtime/migrate.md)。

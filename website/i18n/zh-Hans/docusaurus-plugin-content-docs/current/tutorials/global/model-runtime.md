---
translation:
  source_commit: "dc40c9a164b35316778c66982e6184d9f45cc97c"
  source_file: "docs/tutorials/global/model-runtime.md"
  outdated: false
---

# 模型运行时

## 概述

模型运行时负责运行 Router 使用的所有模型：domain、PII 和 jailbreak 等信号的分类器，语义缓存、记忆和 RAG 使用的嵌入模型，以及重排序器、幻觉检测器和决策模型。托管部署使用统一模型服务 API 下的独立 worker。Router 会启动并监督这些 worker，也可以连接到你自行运行的运行时。

## 解决什么问题？

所有需要模型的功能都通过同一机制获取模型：统一下载、校验、加载模型，并在 CPU 或 GPU 上提供服务。输入兼容的调用可以共享原生批处理。独立逻辑部署使用各自的托管 worker。

模型失败或超过截止时间时，相应证据不可用，路由由决策的未知信号策略决定。请求可能一直等待到截止时间；调用方停止等待后，已经开始的模型 forward 仍可能继续运行。请根据输入长度和并发量配置运行时资源。

## 何时使用

只要配置的功能需要模型，就会使用模型运行时，无需另行开启。你可以自行配置，将模型放到 GPU 上、固定另一个模型、调整副本的位置或数量，或者连接共享的外部 worker。

使用 `vllm-sr serve ARTIFACT --engine`，可以通过同一个持久前端提供原生 System One 请求服务。不带 `--engine` 启动时会启用保存的 recipe 路由。两种启动模式都保留前端和 Dashboard。

## 配置

自行选择的模型通过 `model_runtime` 部署声明：

```yaml
global:
  model_catalog:
    deployments:
      decision-kai:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Kai-0.6B
        device: auto
      decision-shared:
        provider: model_runtime
        endpoint: http://decision-runtime:8100
```

不设置 `endpoint` 时，Router 会为部署启动运行时，并在其退出后重新启动。设置 `endpoint` 时，Router 会连接到你自行运行的运行时。

从以下指南开始：

- [快速入门](../../model-runtime/quickstart.md)：提供模型服务并从 Router 使用它。
- [选择模型、规模和硬件](../../model-runtime/choose-a-model.md)。
- [与 Router 一起运行](../../model-runtime/deploy.md)：设备、进程、外部连接和 Kubernetes。
- [运行配置档](../../model-runtime/profiles.md)：精确答案或更快的近似设置。
- [从原生绑定迁移](../../model-runtime/migrate.md)。
- [故障排查与常见问题](../../model-runtime/troubleshooting.md)。

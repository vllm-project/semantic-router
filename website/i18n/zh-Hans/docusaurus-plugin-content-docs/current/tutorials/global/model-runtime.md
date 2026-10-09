---
translation:
  source_commit: "dc40c9a164b35316778c66982e6184d9f45cc97c"
  source_file: "docs/tutorials/global/model-runtime.md"
  outdated: false
---

# 内置模型运行时

## 概览 {#overview}

模型运行时负责运行 Router 使用的所有模型：为领域、PII 和越狱等信号提供支持的分类器，为语义缓存、记忆和 RAG 提供支持的嵌入模型，以及重排序器、幻觉检测器和判断模型。受管部署使用独立的 worker，通过统一的模型服务 API 提供能力。Router 会启动并监管这些 worker，也可以连接你自行运行的运行时。

## 解决什么问题？ {#what-problem-does-it-solve}

所有依赖模型的功能都通过同一种方式获取模型：由统一的运行时下载、校验、加载模型，并在 CPU 或 GPU 上提供服务。模型输入兼容的调用可以共享原生批次。独立的逻辑部署使用各自的受管 worker。

模型失败或超过截止时间时，相关证据变为不可用，由决策的未知信号策略决定如何路由。请求可能一直等待到截止时间；调用方停止等待后，已经开始的模型 forward 仍可能继续。应根据输入长度和并发量规划运行时资源。

## 何时使用 {#when-to-use}

只要配置的功能需要模型，就会使用模型运行时，无需额外启用。需要将模型放到 GPU、固定其他模型、放置或扩展副本，或连接共享的外部 worker 时，可以自行配置。

使用 `vllm-sr serve ARTIFACT --engine`，可通过同一个持久前端提供原生 System One 请求服务。启动时不加 `--engine`，即可启用已保存配方的路由。两种启动模式都保留前端和 Dashboard。

## 配置 {#configuration}

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

不设置 `endpoint` 时，Router 会为该部署启动运行时，并在其退出后重启。设置 `endpoint` 时，Router 会连接你自行运行的运行时。

从以下文档开始：

- [快速开始](../../model-runtime/quickstart.md)：启动模型并在 Router 中使用。
- [选择模型、规模和硬件](../../model-runtime/choose-a-model.md)。
- [与 Router 一起运行](/docs/model-runtime/deploy)：设备、进程、外部运行时和 Kubernetes。
- [运行配置](/docs/model-runtime/profiles)：精确答案或更快的近似设置。
- [从原生绑定迁移](../../model-runtime/migrate.md)。
- [故障排查与常见问题](../../model-runtime/troubleshooting.md)。

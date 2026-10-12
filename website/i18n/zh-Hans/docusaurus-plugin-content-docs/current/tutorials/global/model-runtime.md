---
translation:
  source_commit: "dc40c9a164b35316778c66982e6184d9f45cc97c"
  source_file: "docs/tutorials/global/model-runtime.md"
  outdated: true
---

# 内置模型运行时

## 概述

模型运行时承载 Router 使用的所有模型，包括领域、PII 和越狱等信号背后的分类器，语义缓存、记忆和 RAG 使用的嵌入模型，以及重排、幻觉检测和判断模型。托管部署使用独立的工作进程，通过统一的模型服务 API 提供能力。Router 可以启动并监督这些进程，也可以连接你自行运行的运行时。

## 解决什么问题？

需要模型的功能使用同一条加载路径，统一完成下载、验证、加载以及 CPU 或 GPU 推理。模型输入兼容的调用可以共享原生批处理；独立的逻辑部署则有各自的托管工作进程。

模型失败或超过截止时间时，Router 会得到不可用的证据，再按 decision 的未知信号策略决定路由。请求可能等待至截止时间，而已开始的模型前向计算在调用方停止等待后仍可能继续。请根据输入长度和并发量规划运行时容量。

## 何时使用

只要配置中的功能需要模型，运行时就会参与，无需额外开关。需要指定 GPU、固定模型、安排或扩展副本，或连接外部共享工作进程时，再显式配置部署。

使用 `vllm-sr serve ARTIFACT --engine`，可以通过同一个持久运行的前端暴露原生 System One 请求。不加 `--engine` 启动时会启用已保存的 recipe 路由。两种启动模式下前端和 Dashboard 都保持可用。

## 配置

通过 `model_runtime` 部署声明你选择的模型：

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

未配置 `endpoint` 时，Router 会启动该部署的运行时，并在它退出后重启。配置 `endpoint` 时，Router 连接你自行运行的运行时。

可以从以下指南开始：

- [快速开始](/model-runtime/quickstart.md)：启动模型并在 Router 中使用。
- [选择模型、规模和硬件](/model-runtime/choose-a-model.md)。
- [与 Router 一起运行](/model-runtime/deploy.md)：设备、进程、连接外部运行时和 Kubernetes。
- [执行配置](/model-runtime/profiles.md)：精确答案或更快的近似设置。
- [从原生绑定迁移](/model-runtime/migrate.md)。
- [排障与常见问题](/model-runtime/troubleshooting.md)。

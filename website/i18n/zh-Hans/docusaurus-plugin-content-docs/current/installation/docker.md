---
is_mtpe: false
title: Docker 部署
description: 将 Semantic Router 作为本地或单主机容器栈运行，并连接你单独运维的模型后端。
translation:
  source_commit: "9be5c38a1a746675535001e9fa45712e93a2cc7f"
  source_file: "docs/installation/docker.md"
  outdated: false
---

# 使用 Docker 部署

Docker 是从 Semantic Router 配置到运行中栈的最短路径。它适合评估、开发、CI、边缘主机，以及不需要 Kubernetes 调度或故障转移的单主机部署。

CLI 会管理 Router、控制面板，以及所选配置所需的支持服务；使用 `--gateway extproc` 时还会在 Router 前面放一个 Envoy 容器（见[网关模式](gateway-modes)）。Router 会为自己的分类器和嵌入模型启动[模型运行时](../model-runtime/overview.md)。模型服务器保持独立：健康的 Router 栈并不意味着其 provider 端点已安装、正在运行或能够生成。

## 启动栈

完成[快速开始](/zh-Hans/docs/installation)以安装 CLI 并创建配置，或从现有 canonical YAML 文件开始：

```bash
vllm-sr config validate --config config.yaml
vllm-sr serve --config config.yaml
```

稳定版 CLI 拉取以自身发布版本为 tag 的 Router 和控制面板镜像；开发版本的 CLI 拉取从它自己的 `main` commit 发布、以该 commit SHA 为 tag 的镜像。源码检出使用 `:latest`。需要其他构建时，用 `--image` 和文档中的镜像环境变量覆盖。

未指定 `--config` 时，`vllm-sr serve` 使用当前目录中的 `config.yaml`，或在控制面板中打开首次运行设置。设置期间该命令保持运行：你在控制面板中激活配置后，它用这份配置启动 Router 并等待其就绪。如果你提前停止它，下一次 `vllm-sr serve` 会启动 Router，`vllm-sr status` 会提示设置是否已完成。控制面板不会拿到容器运行时的 socket。默认本地端点为：

| 端点 | 默认值 | 用途 |
| --- | --- | --- |
| 控制面板 | `http://localhost:8700` | 配置并检查栈。 |
| 已路由监听器 | `http://localhost:8899` | 发送 OpenAI 兼容的模型请求。 |
| 管理 API | `http://localhost:8080` | 校验配置，并使用评估、回放或向量存储 API。 |
| Router 指标 | `http://localhost:9190/metrics` | Prometheus 指标，包括模型运行时的 `vsr_model_runtime_*`。 |

端口可能随活动配置或栈端口偏移而变化。不确定哪些端点处于活动状态时，使用 `vllm-sr status`。

容器启动后，CLI 默认最多等待 1800 秒，直到 Router 就绪；首次设置模式下则等待控制面板就绪。如果模型加载或 GPU 编译需要不同的时限，可将 `--startup-timeout SECONDS` 设为正整数：

```bash
vllm-sr serve --config config.yaml --startup-timeout 7200
```

根据模型和硬件的实际启动时间选择时限。此选项适用于本地 Docker 部署，不改变推理请求的时限。等待超时后，CLI 报错退出，容器继续运行，可使用 `vllm-sr status` 和 `vllm-sr logs router` 检查。需要停止时，运行 `vllm-sr stop`。

## 连接模型后端

根据模型服务器的运行位置选择连接方式：

| 模型位置 | 用以下方式配置后端 |
| --- | --- |
| 在 Docker 主机上 | `host.docker.internal:<port>`；CLI 会添加 host-gateway 映射。 |
| 在同一网络的容器中 | 模型容器的 DNS 名称和服务端口。 |
| 在另一主机或托管服务上 | 其可到达的 HTTPS base URL，以及由环境变量提供的凭据。 |

对于小型本地模型，请遵循[使用 Ollama 配置模型](ollama)。对于 GPU 支持的 vLLM 服务器，在 **Hardware** 下选择指南。无论哪种情况，都要在调试路由之前直接验证模型端点。

## 运维本地栈

```bash
vllm-sr status
vllm-sr logs router -f
vllm-sr dashboard
vllm-sr stop
```

使用 `--gateway extproc` 时，`vllm-sr logs envoy` 还能查看 Envoy 容器的日志；standalone 模式下没有 Envoy 容器，该命令会说明这一点。

使用 `--minimal` 在不启动控制面板和可观测性栈（Jaeger、Prometheus、Grafana）的情况下运行。使用 `--readonly` 保持控制面板可用但不允许更改配置。在将监听器暴露到受信任主机之外之前，固定镜像并复核[安全加固](security-hardening)。

### 让栈在崩溃和重启后继续运行

`vllm-sr serve` 创建的容器默认没有重启策略：崩溃的容器会保持停止状态，主机重启后栈也不会恢复。在服务器上，请带上重启策略启动：

```bash
vllm-sr serve --config config.yaml --restart-policy unless-stopped
```

`unless-stopped` 和 `always` 会重启退出的容器，并在 Docker 守护进程启动时重新启动栈；`on-failure` 只重启失败的容器。`vllm-sr stop` 仍然会停止栈。策略只作用于本次启动创建的容器，因此每次 `serve` 都要传入。使用 Podman 时，还需要启用它的重启服务（`systemctl enable --now podman-restart.service`），带策略的容器才会在重启后恢复。

使用 `--gateway extproc` 时，Envoy 默认使用安全的 `info` 日志级别。若要临时排查问题，在启动栈之前设置 `VLLM_SR_ENVOY_LOG_LEVEL=debug`，完成后取消设置：debug 日志可能在日志中暴露转发的请求标头，包括 provider 的 `Authorization` 标头。

## 何时迁移到 Kubernetes

Docker 不提供多节点调度、滚动部署控制或集群级恢复。当你需要跨节点 Router 副本、声明式发布、网关集成或平台管理的模型发现时，迁移到 Kubernetes 路径。同一份 canonical 配置可以通过 CLI 和 Helm 部署，或通过 Semantic Router Operator 管理。

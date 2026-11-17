---
is_mtpe: false
title: Docker 部署
description: 将 Semantic Router 作为本地或单主机容器栈运行，并连接你单独运维的模型后端。
translation:
  source_commit: "bc5c6bfb747ffe9c5f35d98b203427dada8c4c56"
  source_file: "docs/installation/docker.md"
  outdated: false
---

# 使用 Docker 部署 {#deploy-with-docker}

Docker 是从 Semantic Router 配置到运行中服务的最短路径。它适合评估、开发、CI、边缘主机，以及不需要 Kubernetes 调度或故障转移的单主机部署。

| 组件 | 由谁运行 |
| --- | --- |
| Router、Dashboard 以及配置所需的服务 | CLI |
| Router 前面的 Envoy 容器 | CLI，使用 `--gateway extproc` 时（见[网关模式](gateway-modes)） |
| Router 的决策模型和嵌入模型 | [模型运行时](../model-runtime/get-started/overview.md)，由 Router 启动 |
| 回答用户的模型服务器 | 你 |

:::note
Router 服务健康，并不代表它的 provider 端点已安装、正在运行或能够生成。请直接测试这些端点。
:::

## 启动服务 {#start-the-stack}

完成[快速开始](installation.md)以安装 CLI 并创建配置，或从现有的 canonical YAML 文件开始：

```bash
vllm-sr config validate --config config.yaml
vllm-sr serve --config config.yaml
```

| CLI | 拉取的 Router 和 Dashboard 镜像 tag |
| --- | --- |
| 稳定版 | 它自己的发布版本 |
| 开发版本 | 构建它的 `main` commit 的 SHA |
| 源码检出 | `:latest` |

需要其他构建时，用 `--image` 和文档中的镜像环境变量覆盖。

未指定 `--config` 时，`vllm-sr serve` 使用当前目录中的 `config.yaml`，或在 Dashboard 中打开首次运行设置：

- **设置期间命令保持运行。** 你在 Dashboard 中激活配置后，它用这份配置启动 Router，并等待其就绪。
- **如果你提前停止了它，**下一次 `vllm-sr serve` 会启动 Router，`vllm-sr status` 会提示设置何时完成。
- **Dashboard 从不获得容器运行时的 socket。**

默认的本地端点为：

| 端点 | 默认值 | 用途 |
| --- | --- | --- |
| Dashboard | `http://localhost:8700` | 配置并检查服务。 |
| 路由 listener | `http://localhost:8899` | 发送 OpenAI 兼容的模型请求。 |
| 管理 API | `http://localhost:8080` | 校验配置，并使用评估、回放或向量存储 API。 |
| Router 指标 | `http://localhost:9190/metrics` | Prometheus 指标，包括模型运行时的 `vsr_model_runtime_*`。 |

端口可能随活动配置或服务的端口偏移而变化。不确定哪些端点处于活动状态时，使用 `vllm-sr status`。

Dashboard 默认绑定 `127.0.0.1`，首次运行的管理员注册仍可在本机完成。在远程主机上，用 `ssh -L 8700:127.0.0.1:8700 <host>` 转发端口，再打开本地 Dashboard 地址。

:::warning 在所有网络接口上发布 Dashboard
设置 `VLLM_SR_DASHBOARD_HOST_BIND=0.0.0.0`，并且要么同时提供 `DASHBOARD_ADMIN_EMAIL` 和 `DASHBOARD_ADMIN_PASSWORD`，要么在确实打算允许该网络上注册首个管理员时设置 `DASHBOARD_ALLOW_OPEN_BOOTSTRAP=true`。两者都没有配置时，CLI 会拒绝通配地址绑定。
:::

容器启动后，CLI 最多等待 1800 秒让 Router 就绪；首次设置期间则等待 Dashboard 就绪。如果模型加载或 GPU 编译需要不同的时限，请设置 `--startup-timeout SECONDS`：

```bash
vllm-sr serve --config config.yaml --startup-timeout 7200
```

- **先测量。** 根据模型和硬件的实际启动时间选择时限。
- **只影响启动。** 它适用于本地 Docker 目标，不改变推理请求的时限。
- **超时时，**CLI 报错退出，并保留容器供 `vllm-sr status` 和 `vllm-sr logs router` 检查。需要停止时，运行 `vllm-sr stop`。

## 连接模型后端 {#connect-model-backends}

根据模型服务器的运行位置选择连接方式：

| 模型位置 | 用以下方式配置后端 |
| --- | --- |
| 在 Docker 主机上 | `host.docker.internal:<port>`；CLI 会添加 host-gateway 映射。 |
| 在同一网络的容器中 | 模型容器的 DNS 名称和服务端口。 |
| 在另一主机或托管服务上 | 其可到达的 HTTPS base URL，以及由环境变量提供的凭据。 |

对于小型本地模型，请按[使用 Ollama 配置模型](ollama)操作。对于 GPU 支持的 vLLM 服务器，在 **Hardware** 下选择指南。无论哪种情况，都要在调试路由之前直接验证模型端点。

## 运维本地服务 {#operate-a-local-stack}

```bash
vllm-sr status
vllm-sr logs router -f
vllm-sr dashboard
vllm-sr stop
```

使用 `--gateway extproc` 时，`vllm-sr logs envoy` 还能查看 Envoy 容器的日志；standalone 模式下没有 Envoy 容器，该命令会说明这一点。

使用 `--minimal` 可以不启动 Dashboard 和可观测性栈（Jaeger、Prometheus、Grafana）。使用 `--readonly` 可以保留 Dashboard，但不允许更改配置。在把 listener 暴露到受信任主机之外之前，请固定镜像并复核[安全加固](security-hardening)。

### 让服务在崩溃和重启后继续运行 {#keep-the-stack-running-across-crashes-and-reboots}

`vllm-sr serve` 创建的容器默认没有重启策略：崩溃的容器会保持停止状态，主机重启后服务也不会恢复。在服务器上，请带上重启策略启动：

```bash
vllm-sr serve --config config.yaml --restart-policy unless-stopped
```

| 策略 | 行为 |
| --- | --- |
| `unless-stopped`、`always` | 重启退出的容器，并在 Docker 守护进程启动时重新启动服务。 |
| `on-failure` | 只重启失败的容器。 |

- `vllm-sr stop` 仍然会停止服务。
- 策略只作用于 `serve` 创建的容器，因此每次 `serve` 都要传入。
- 使用 Podman 时，还需启用它的重启服务（`systemctl enable --now podman-restart.service`），容器才会在主机重启后恢复。

:::warning Envoy 的 debug 日志
使用 `--gateway extproc` 时，Envoy 默认使用安全的 `info` 日志级别。排查问题时，在启动服务之前设置 `VLLM_SR_ENVOY_LOG_LEVEL=debug`，完成后取消设置：debug 日志可能暴露转发的请求头，包括 provider 的 `Authorization` 头。
:::

## 通过反向代理暴露 Dashboard {#expose-the-dashboard-through-a-reverse-proxy}

Grafana Live 会校验建立 WebSocket 时使用的 `Origin` 头。当 HTTPS 反向代理在公共主机名上暴露 Dashboard 时，请在启动服务之前允许该确切的 origin：

```bash
export GF_LIVE_ALLOWED_ORIGINS='https://dashboard.example.com'
vllm-sr serve --config config.yaml
```

- 多个 origin 用逗号分隔，格式为 `scheme://host[:port]`，不带路径、查询、片段或结尾斜杠。
- Grafana 支持通配模式，但对公共部署而言，确切的 origin 更安全。
- `vllm-sr serve` 会把该值写入生成的 `.vllm-sr/grafana/grafana.serve.ini`；修改后请重启服务。

## 何时迁移到 Kubernetes {#when-to-move-to-kubernetes}

Docker 不提供多节点调度、滚动部署控制或集群级恢复。当你需要跨节点的 Router 副本、声明式发布、网关集成或平台管理的模型发现时，请迁移到 Kubernetes 路径。同一份 canonical 配置可以通过 CLI 和 Helm 部署，或通过 Semantic Router Operator 管理。

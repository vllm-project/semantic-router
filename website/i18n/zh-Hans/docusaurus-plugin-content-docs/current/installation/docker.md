---
title: Docker 部署
description: 将 Semantic Router 作为本地或单主机容器栈运行，并连接你单独运维的模型后端。
translation:
  source_commit: "e56591a9cb24f073bf159927e87116ba6d278741"
  source_file: "docs/installation/docker.md"
  outdated: false
---

# 使用 Docker 部署

Docker 是从 Semantic Router 配置到运行中栈的最短路径。它适合评估、开发、CI、边缘主机，以及不需要 Kubernetes 调度或故障转移的单主机部署。

CLI 会管理 Router、Envoy、控制面板，以及所选配置所需的支持服务。模型服务器保持独立：健康的 Router 栈并不意味着其 provider 端点已安装、正在运行或能够生成。

## 启动栈

完成[快速开始](/zh-Hans/docs/installation)以安装 CLI 并创建配置，或从现有 canonical YAML 文件开始：

```bash
vllm-sr config validate --config config.yaml
vllm-sr serve --config config.yaml
```

未指定 `--config` 时，`vllm-sr serve` 使用当前目录中的 `config.yaml`，或在控制面板中打开首次运行设置。默认本地端点为：

| 端点 | 默认值 | 用途 |
| --- | --- | --- |
| 控制面板 | `http://localhost:8700` | 配置并检查栈。 |
| 已路由监听器 | `http://localhost:8899` | 发送 OpenAI 兼容的模型请求。 |
| 管理 API | `http://localhost:8080` | 校验配置，并使用评估、回放或向量存储 API。 |

端口可能随活动配置或栈端口偏移而变化。不确定哪些端点处于活动状态时，使用 `vllm-sr status`。

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
vllm-sr logs router
vllm-sr logs envoy -f
vllm-sr dashboard
vllm-sr stop
```

使用 `--minimal` 仅运行 Router 和 Envoy。使用 `--readonly` 保持控制面板可用但不允许更改配置。在将监听器暴露到受信任主机之外之前，固定镜像并复核[安全加固](security-hardening)。

Envoy 默认使用安全的 `info` 日志级别。若要临时排查问题，在启动栈之前设置 `VLLM_SR_ENVOY_LOG_LEVEL=debug`，完成后取消设置：debug 日志可能在日志中暴露转发的请求标头，包括 provider 的 `Authorization` 标头。

## 何时迁移到 Kubernetes

Docker 不提供多节点调度、滚动部署控制或集群级恢复。当你需要副本、声明式发布、网关集成或平台管理的模型发现时，迁移到 Kubernetes 路径。同一份 canonical 配置可以通过 CLI 和 Helm 部署，或通过 Semantic Router Operator 管理。

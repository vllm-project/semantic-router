---
translation:
  source_commit: "8e37ee0"
  source_file: "docs/installation/docker-compose.md"
  outdated: false
sidebar_position: 3
---

# 使用 Docker Compose 安装

:::warning 已弃用
仓库内置的 `deploy/docker-compose/docker-compose.yml` 已被移除，Docker Compose
不再是本地运行 vLLM Semantic Router 的受支持方式。

如需使用 Docker 进行本地开发，请改用 `vllm-sr` CLI —— `vllm-sr serve` 会为你
启动 Router、Envoy 与 Dashboard。当前流程请参阅 **[快速开始](./installation.md)**。
:::

## 为什么有此变更

早期版本在 `deploy/docker-compose/` 下提供 `docker-compose.yml`。现在本地 Docker
编排统一由 `vllm-sr` CLI 处理，它会让容器接线、配置初始化与 Dashboard 始终与
每个版本保持一致。

## 使用 CLI 在本地运行

```bash
# 安装 CLI（前提条件请参阅快速开始）
pip install --pre vllm-sr

# 在 Docker 中启动 Router、Envoy 与 Dashboard
vllm-sr serve
```

**[快速开始](./installation.md)** 涵盖了前提条件、模型下载，以及完整的 `vllm-sr`
命令集（`vllm-sr status`、`vllm-sr logs`、`vllm-sr stop`）。

## Kubernetes 与生产环境

集群部署请使用 **[Operator](k8s/operator)** 或 `deploy/helm/` 下的 Helm Chart。

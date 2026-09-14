---
title: 受限网络环境
sidebar_label: 受限网络
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/troubleshooting/network-tips.md"
  outdated: false
---

# 受限网络环境

Semantic Router 可能因三种不同原因需要网络访问：

1. 容器运行时拉取 Router、控制面板、Envoy 和支持镜像；
2. Router 下载分类器或嵌入产物；以及
3. 被路由的请求调用你配置的模型提供方。

在更改镜像或代理设置之前，先确认失败的是哪一层。仓库超时、Hugging Face 超时和不可达的提供方端点需要不同的修复。

## 诊断失败层

启动协议栈，并检查其状态和组件日志：

```bash
vllm-sr serve --config config.yaml
vllm-sr status
vllm-sr logs router
vllm-sr logs envoy
```

| 现象 | 可能的层 |
|---------|--------------|
| 镜像拉取或仓库身份验证错误 | 容器仓库 |
| Router 已启动，但在加载模型产物时等待 | Hugging Face 或本地模型路径 |
| Router 和 Envoy 已就绪，但 completions 返回连接错误 | 提供方端点或防火墙 |
| Kubernetes Pod 停留在 `ImagePullBackOff` | 集群节点访问仓库 |

## 容器镜像

从能到达其仓库的网络预先拉取镜像，或把它们镜像到部署环境可用的仓库。本地开发时，在使用不拉取策略之前确认每个所需镜像都已存在：

```bash
vllm-sr serve --config config.yaml --image-pull-policy never
```

`never` 不会下载缺失镜像；镜像不存在时启动失败。当本地镜像应被复用、但缺失的仍可拉取时，使用 `ifnotpresent`。

从源码构建项目时，通过组织已批准的代理或镜像配置包管理器和容器运行时。避免把区域端点、凭据或本地代理地址提交到仓库。

## Hugging Face 下载

本地 CLI 会把 `HF_ENDPOINT`、`HF_TOKEN`、`HF_HOME` 和 `HF_HUB_CACHE` 转发到 Router 容器。只设置环境需要的值：

```bash
export HF_ENDPOINT=https://your-approved-hugging-face-mirror.example
export HF_TOKEN=your_token_if_required
vllm-sr serve --config config.yaml
```

把令牌放在环境或外部密钥管理器中。CLI 会在日志中遮蔽敏感的透传值。

离线部署时，预先下载所需产物并放到工作区模型目录：

- 普通 YAML 工作区把 `models/` 挂到 `/app/models`；
- 受管配方把可变模型状态放在 `.vllm-sr/models/`，并挂到同一容器路径。

在 Router 配置中使用 `/app/models/...`，然后确认文件存在于运行时内，且格式与所选信号或嵌入实现匹配。

## 提供方端点

提供方 URL 必须从 Router/Envoy 网络可达，而不仅是从宿主机 shell。不要把运行在另一容器或宿主机上的后端写成 `localhost`；在容器内，`localhost` 指向该容器自身。

使用以下模式之一：

- 同一容器网络上的服务名；
- 容器运行时可达的主机地址或主机网关名；
- Kubernetes Service DNS 名；或
- 可路由的私有或公共提供方端点。

逐步的端点和防火墙检查清单见[容器连通性](./container-connectivity)。

## Kubernetes 镜像拉取

Kubernetes 节点使用自己的容器运行时，不会继承运行 `kubectl` 的机器上的镜像缓存或代理设置。

对于受限集群：

- 把所需镜像镜像到每个节点可达的仓库；
- 为需要身份验证的仓库配置 `imagePullSecrets`；
- 镜像就位后使用合适的拉取策略；
- 对于开发集群，用集群工具支持的命令预加载镜像；以及
- 检查 Pod 事件，以区分 DNS、身份验证、速率限制和缺失镜像错误。

```bash
kubectl describe pod <pod-name> -n <namespace>
kubectl get events -n <namespace> --sort-by=.lastTimestamp
```

## 不要做的事

- 不要提交 API 令牌、仓库凭据、代理密码或私有镜像地址。
- 不要把禁用 TLS 验证当作长期变通办法。
- 不要假定宿主机侧一次成功的 `curl` 就能证明容器或 Pod 可达。
- 不要用环境特定副本替换已入库的 Dockerfile；把组织特定的构建配置放在源码树之外。

## 相关指南

- [容器连通性](./container-connectivity)
- [安全加固](../installation/security-hardening)
- [快速开始](/zh-Hans/docs/installation)

---
title: 容器连通性
sidebar_label: 容器连通性
translation:
  source_commit: "867155c924b6527d6a412e1412ce712a9e5cc9b8"
  source_file: "docs/troubleshooting/container-connectivity.md"
  outdated: false
---

# 容器连通性

当本地协议栈已启动但无法到达模型后端，或宿主机无法到达 Router、Envoy、控制面板或指标端点时，使用本指南。

## 从失败的那一跳开始

一次路由请求会跨越多个网络边界：

```text
client -> Envoy -> Router -> selected provider backend
```

按该顺序检查每一跳：

```bash
vllm-sr status
vllm-sr logs envoy
vllm-sr logs router
curl -sS http://localhost:8899/v1/models
```

若 `/v1/models` 不可用，先关注本地协议栈和已发布端口。若它成功但 completions 失败，在响应/日志中检查所选提供方名称，并从 Router 的网络测试该提供方。

## `localhost` 表示当前容器

常见配置错误是把提供方指向 `localhost`，而推理服务器实际运行在宿主机或另一个容器中。从 Router 容器看，`127.0.0.1` 和 `localhost` 指向 Router 容器自身。

配置运行时网络可达的地址：

```yaml
providers:
  models:
    - name: local-model
      provider_model_id: local-model
      api_format: openai
      backend_refs:
        - name: local-vllm
          endpoint: model-server:8000
          protocol: http
          provider: vllm
```

当两个服务共享同一网络时，使用容器 DNS 名。对于直接跑在宿主机上的模型服务器，使用容器运行时支持的主机网关名，或容器可达的宿主机 IP。在 Kubernetes 上使用 Service DNS 名，而不是 Pod IP。

## 让后端监听回环以外的接口

模型服务器必须绑定到客户端可达的接口。例如，宿主机上的 vLLM 服务器通常需要 `--host 0.0.0.0`：

```bash
vllm serve <model-id> \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name local-model
```

绑定到 `0.0.0.0` 不提供身份验证。用防火墙或私有网络限制端口，并在流量跨越信任边界时使用提供方的身份验证支持。

## 从同一网络命名空间测试

宿主机侧请求成功只证明宿主机可达。请从同一运行时网络上的临时容器测试，或用容器运行时的检查工具从 Router 容器测试。查询后端的 OpenAI 兼容模型端点：

```bash
curl -sS http://<reachable-backend>:8000/v1/models
```

对于 Kubernetes，从 Service 和 endpoints 开始：

```bash
kubectl get service,endpoints -n <namespace>
kubectl run network-check \
  --rm -i --restart=Never \
  --image=curlimages/curl \
  -n <namespace> -- \
  curl -sS http://<service-name>:8000/v1/models
```

按集群策略删除或限制临时诊断 Pod。

## 检查防火墙和安全规则

若宿主机能到达后端而运行时不能，检查：

- 后端端口的宿主机防火墙规则；
- 云安全组或网络 ACL；
- Kubernetes NetworkPolicies；
- 只拦截部分地址范围的企业代理；以及
- 容器或 Pod 内的 DNS 解析。

只开放部署需要的源网络和端口。不要仅为诊断内部路由而把提供方端点公开。

## 已发布的本地端口

默认本地协议栈发布这些面向用户的端点：

| 端点 | 默认地址 |
|----------|-----------------|
| 经 Envoy 的 OpenAI 兼容监听器 | `http://localhost:8899` |
| 控制面板 | `http://localhost:8700` |
| Router 管理 API | `http://localhost:8080` |
| Router 指标 | `http://localhost:9190/metrics` |

管理端点可能需要身份验证，取决于活动配置。非零的 `VLLM_SR_PORT_OFFSET` 会平移每个已发布的宿主机端口；使用 `vllm-sr status` 和 `vllm-sr dashboard`，不要假定默认值。

若端口已被占用，停止冲突进程，或运行隔离协议栈：

```bash
VLLM_SR_STACK_NAME=lane-b \
VLLM_SR_PORT_OFFSET=200 \
vllm-sr serve --config config.yaml
```

对该协议栈的 `status`、`logs`、`dashboard` 和 `stop` 使用同样的两个环境变量。

## Grafana 没有数据

仅在启用可观测性时才有 Grafana 和 Prometheus。在调试面板之前先检查指标源：

```bash
curl -sS http://localhost:9190/metrics | head
```

然后确认 Prometheus 能抓取 Router 和 Envoy 目标。在没有请求走过对应路径时，空面板可以是正确的；例如，拒绝和缓存指标在策略拒绝或缓存处理请求之前会保持为空。

同时验证：

- 所选时间范围包含最近流量；
- 活动协议栈的端口偏移已反映在本地 URL 中；
- 直方图面板用合适的 rate 窗口查询 bucket 指标；以及
- 复制来的 dashboard 中的标签与此版本发出的指标匹配。

## 快速检查清单

- 本地协议栈正在运行，且 `vllm-sr status` 能识别失败组件。
- 客户端可以到达 Envoy 的 `/v1/models` 端点。
- 提供方端点不是 `localhost`，除非它确实运行在同一容器中。
- 后端监听可达接口，并暴露 `/v1/models`。
- DNS、防火墙、安全组和 NetworkPolicy 规则允许所需路径。
- Router 配置中的模型名与提供方服务的名称匹配。
- 在调查 Grafana 查询之前，指标已经发出。

仓库或产物下载失败见[受限网络环境](./network-tips)。

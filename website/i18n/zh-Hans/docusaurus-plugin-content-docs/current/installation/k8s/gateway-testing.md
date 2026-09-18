---
title: 测试 Kubernetes Gateway 部署
sidebar_label: 测试 Gateway 部署
description: 验证网关可达性、直接模型请求、语义路由、响应头和后端选择。
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/installation/k8s/gateway-testing.md"
  outdated: false
---

# 测试 Kubernetes Gateway 部署

在 Istio、Envoy Gateway、Gateway API Inference Extension 或其他受支持网关后面安装 Semantic Router 后，使用此清单。

## 1. 解析真实的网关地址

不要从另一个集群复制 IP 或 NodePort。检查为 Gateway 创建的 Service：

```bash
kubectl get gateway -A
kubectl get service -A | grep -i gateway
```

对于 Minikube，辅助命令可以打印可达 URL：

```bash
export GATEWAY_URL="$(minikube service inference-gateway-istio --url | head -n 1)"
```

对于 LoadBalancer Service，使用其分配的主机名或 IP。对于仅本地集群，对 Gateway Service 做 port-forward，并将 `GATEWAY_URL` 设为转发地址。

```bash
test -n "$GATEWAY_URL" && printf 'Gateway: %s\n' "$GATEWAY_URL"
```

## 2. 检查 Gateway API 状态

```bash
kubectl get gateway,httproute -A
kubectl describe httproute <route-name> -n <namespace>
```

路由应被接受，其后端引用应已解析。在 LLM-D 部署中，还要检查每条路由所选的 `InferencePool` 和 EPP 调度器。

## 3. 列出已暴露的模型

```bash
curl -fsS "$GATEWAY_URL/v1/models"
```

确认响应包含当前 Router 配置所期望的物理或虚拟模型名称。

## 4. 发送直接模型请求

将 `physical-model` 替换为配置所暴露的提供商模型：

```bash
curl -fsS -D /tmp/direct-headers.txt \
  "$GATEWAY_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "physical-model",
    "messages": [{"role": "user", "content": "Reply with one short sentence."}],
    "max_tokens": 64,
    "temperature": 0
  }'
```

直接提供商名称应到达该提供商，而不运行配方信号、决策、路由插件、缓存、学习或会话路由。

## 5. 发送已路由请求

将 `virtual-model` 替换为该部署配置的入口或自动别名：

```bash
curl -fsS -D /tmp/routed-headers.txt \
  "$GATEWAY_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "virtual-model",
    "messages": [{"role": "user", "content": "Explain why 2 + 2 equals 4."}],
    "max_tokens": 128,
    "temperature": 0
  }'
```

检查响应以及部署所支持的路由头。确认所选提供商对匹配的决策有效；不要仅根据提示词措辞假定特定类别或后端。

## 6. 验证后端路径

将请求与 Gateway、Router 和提供商日志关联。仅有成功的 HTTP 响应，并不能证明预期路由或调度器处理了该请求。

```bash
kubectl logs -n <router-namespace> deployment/<router-deployment> --since=5m
kubectl logs -n <gateway-namespace> deployment/<gateway-deployment> --since=5m
```

当请求和响应体可能包含敏感数据时，不要将其放入共享日志。

## 故障指南

| 现象 | 先检查 |
|---------|-------------|
| 无外部连接 | Gateway Service 地址、LoadBalancer/NodePort、防火墙 |
| HTTPRoute 未被接受 | 父引用、监听器主机名、允许的路由 |
| 后端引用未解析 | Service/InferencePool 名称、命名空间、端口 |
| `/v1/models` 可用但 completions 失败 | 提供商就绪状态、所服务的模型名称、凭证 |
| 直接请求可用但虚拟模型失败 | 入口、配方、信号/决策、默认模型 |
| 响应成功但经过错误的池 | 路由匹配、所选模型头、EPP/Gateway 日志 |

检查后如果临时头文件可能包含敏感元数据，请删除它们。

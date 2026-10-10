---
title: 与 NVIDIA Dynamo 集成
sidebar_label: NVIDIA Dynamo
description: 将 Semantic Router 放在 Kubernetes 上由 Dynamo 管理的推理集群前面。
translation:
  source_commit: "867155c924b6527d6a412e1412ce712a9e5cc9b8"
  source_file: "docs/installation/k8s/dynamo.md"
  outdated: false
---

# 与 NVIDIA Dynamo 集成

当 NVIDIA Dynamo 已经负责模型服务，而你希望 Semantic Router 在 Dynamo 将请求调度到推理 worker 之前选择模型或策略时，使用此集成。

两层路由在不同层次做决策：

| 层次 | 职责 |
|-------|----------------|
| Semantic Router | 解析入口，评估语义信号和策略，运行请求插件，并选择提供商模型。 |
| Dynamo | 协调推理图，暴露其 OpenAI 兼容前端，并按其服务拓扑和缓存感知路由选择 worker。 |

Semantic Router 必须选择 Dynamo 前端所服务的模型名称。随后 Dynamo 可以在该模型的 worker 之间选择。

```text
Client
  -> Envoy Gateway
     -> Semantic Router (ExtProc)
        -> Dynamo frontend
           -> Dynamo router and inference workers
```

语义响应缓存与 Dynamo KV-cache 路由相互独立。Semantic Router 缓存可以复用先前响应；Dynamo 的路由器在服务新请求时复用或预测 token 级前缀状态。

## 前置条件

需要：

- Kubernetes `1.33`–`1.36`，且节点带 NVIDIA GPU。这是本集成所固定的 Envoy Gateway `v1.9.0` 的支持范围。
- Gateway API `v1.6.1` CRD
- 与集群支持的版本偏差范围内的 `kubectl`
- Helm 3
- NVIDIA GPU Operator，除非集群提供方已提供等效的 GPU 驱动和容器运行时集成
- 所选模型需要认证时，准备 Hugging Face token Secret

集群准备、支持的加速器和可选调度器，见 [NVIDIA Dynamo Kubernetes Quickstart](https://docs.nvidia.com/dynamo/dev/kubernetes/getting-started/quickstart)。

## 1. 安装 Dynamo 平台

本指南固定 Dynamo 1.4.0。安装或升级前，请在 NVIDIA 的 [Dynamo 发布产物](https://docs.nvidia.com/dynamo/dev/reference/release-artifacts) 中确认版本及其兼容矩阵。

```bash
export DYNAMO_NAMESPACE=dynamo-system
export DYNAMO_VERSION=1.4.0

helm upgrade --install dynamo-platform \
  "https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-${DYNAMO_VERSION}.tgz" \
  --namespace "$DYNAMO_NAMESPACE" \
  --create-namespace \
  --wait \
  --timeout 10m
```

部署模型前先检查 Operator 和平台服务：

```bash
helm status dynamo-platform --namespace "$DYNAMO_NAMESPACE"
kubectl get pods --namespace "$DYNAMO_NAMESPACE"
kubectl get crd | grep -i dynamo
```

该平台 chart 安装 Dynamo 控制面。它本身不会定义将服务你请求的模型拓扑。

## 2. 用 Dynamo 部署模型

按 NVIDIA 的[模型部署概览](https://docs.nvidia.com/dynamo/dev/kubernetes/model-deployment/introduction) 选择调优过的配方，用 `DynamoGraphDeploymentRequest` 生成部署，或应用已知可用的 `DynamoGraphDeployment`。这些 API 和运行时镜像随 Dynamo 演进，因此本指南不复制其清单。

部署后，识别前端 Service 并确认所服务的模型名称：

```bash
kubectl get dynamographdeployments,dynamocomponentdeployments \
  --namespace "$DYNAMO_NAMESPACE"
kubectl get services --namespace "$DYNAMO_NAMESPACE"
```

记录 Semantic Router 将使用的值：

```bash
export DYNAMO_FRONTEND_SERVICE=your-frontend-service
export DYNAMO_FRONTEND_PORT=8000
export DYNAMO_MODEL=your-served-model-name
```

在加入另一层路由之前，在单独终端对前端 Service 做 port-forward，并发送直接的 OpenAI 兼容请求。这样可以把 Dynamo 部署问题与网关或 Semantic Router 问题分开。

```bash
kubectl port-forward \
  --namespace "$DYNAMO_NAMESPACE" \
  "service/$DYNAMO_FRONTEND_SERVICE" \
  8000:"$DYNAMO_FRONTEND_PORT"
```

使用所选 Dynamo 部署中的请求示例。同时验证模型标识，以及针对 `http://localhost:8000` 的 chat completion。

```bash
curl -fsS http://localhost:8000/v1/models
```

:::caution 仓库示例的兼容性

本仓库中的 `deploy/kubernetes/dynamo/helm-chart` 和 `dynamo-resources/dynamo-graph-deployment.yaml` 示例面向较旧的 Dynamo API 和运行时。不要原样与 1.4.0 平台组合使用。Dynamo 模型资源请使用 NVIDIA 当前的部署指南；下面的仓库文件仅用于 Semantic Router 和网关集成。

:::

## 3. 为 Dynamo 前端配置 Semantic Router

下载集成 values，安装 chart 前先编辑：

```bash
curl -fsSL \
  https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/dynamo/semantic-router-values/values.yaml \
  -o semantic-router-dynamo-values.yaml
```

用你的 Dynamo 部署中的值替换示例模型和端点。提供商端点应使用集群 DNS：

```text
<frontend-service>.<dynamo-namespace>.svc.cluster.local:<frontend-port>
```

同时更新所有引用示例模型的 `modelRefs[].model` 以及 `providers.defaults.model`。这些名称必须匹配 Dynamo 前端返回的条目。

开发环境可安装持续发布的 Semantic Router chart：

```bash
export SEMANTIC_ROUTER_NAMESPACE=vllm-semantic-router-system

helm upgrade --install semantic-router \
  oci://ghcr.io/vllm-project/charts/semantic-router \
  --version 0.0.0-latest \
  --namespace "$SEMANTIC_ROUTER_NAMESPACE" \
  --create-namespace \
  --values semantic-router-dynamo-values.yaml \
  --wait
```

`0.0.0-latest` 跟随 main 分支。生产环境请固定经过测试的 chart 版本和明确的 Semantic Router 镜像标签。

## 4. 连接 Envoy Gateway

仓库集成使用 `EnvoyPatchPolicy` 将 Semantic Router 插入为 ExtProc 过滤器。安装 Envoy Gateway 时启用该扩展：

```bash
export ENVOY_GATEWAY_VERSION=v1.9.0

helm upgrade --install envoy-gateway \
  oci://docker.io/envoyproxy/gateway-helm \
  --version "$ENVOY_GATEWAY_VERSION" \
  --namespace envoy-gateway-system \
  --create-namespace \
  --values https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/dynamo/dynamo-resources/envoy-gateway-values.yaml \
  --wait
```

若集群已有 Envoy Gateway，升级前请确认其版本和 Gateway API CRD 兼容。`EnvoyPatchPolicy` 对版本敏感，并可能改变网关行为；限制创建或修改这些资源的权限。见 [Envoy Gateway 安装指南](https://gateway.envoyproxy.io/docs/install/install-helm/) 和 [EnvoyPatchPolicy 安全指引](https://gateway.envoyproxy.io/docs/tasks/extensibility/envoy-patch-policy/)。

下载 Gateway API 集成清单：

```bash
curl -fsSL \
  https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/dynamo/dynamo-resources/gwapi-resources.yaml \
  -o semantic-router-dynamo-gateway.yaml
```

应用前，更新这些环境相关引用：

- `HTTPRoute` 后端 Service 的名称、命名空间和端口
- 当 Dynamo 前端不在 `dynamo-system` 时，更新 `ReferenceGrant` 命名空间
- 若更改了 Semantic Router 的 release 名称或命名空间，更新其 Service 地址
- 当 `default` 不合适时，更新 Gateway 和路由的命名空间

然后应用并检查资源状态：

```bash
kubectl apply --filename semantic-router-dynamo-gateway.yaml
kubectl get gateway,httproute --all-namespaces
kubectl describe envoypatchpolicy semantic-router-extproc-patch-policy \
  --namespace default
```

等到 Gateway 和路由被接受、patch 策略已生效后再继续。

## 5. 验证完整请求路径

解析集群中真实的 Gateway 地址并设置 `GATEWAY_URL`。[网关测试清单](./gateway-testing) 覆盖 LoadBalancer、本地集群、路由状态和日志检查。

通过 Gateway 发送请求，使用 Semantic Router 中配置的入口。示例 values 使用默认自动入口：

```bash
curl -fsS -D - "$GATEWAY_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "auto",
    "messages": [
      {"role": "user", "content": "Explain why prefix caching can reduce inference latency."}
    ],
    "max_tokens": 128,
    "temperature": 0
  }'
```

在 Gateway、Semantic Router 和 Dynamo 前端日志中关联同一请求。仅有成功的 HTTP 响应，并不能证明请求经过了预期路由或到达了所选的 Dynamo 部署。

## 故障排查

| 现象 | 先检查 |
|---------|-------------|
| 直接前端请求失败 | Dynamo 部署状态、GPU 分配、模型凭证和前端日志 |
| 直接前端可用但 Gateway 失败 | `HTTPRoute` 后端名称和端口、`ReferenceGrant` 和 Gateway 地址 |
| patch 策略未被接受 | Envoy Gateway 扩展设置、patch 目标命名空间和发行兼容性 |
| 物理模型可用但 `auto` 失败 | Semantic Router 入口、提供商模型名称、决策和分类器就绪状态 |
| 请求到达错误的 Dynamo 部署 | 所选模型头、提供商端点、Dynamo 前端模型列表，以及两层路由的日志 |

## 清理

只删除为此集成创建的资源。先移除 Gateway 资源，避免拆除过程中仍有新流量进入：

```bash
kubectl delete --filename semantic-router-dynamo-gateway.yaml \
  --ignore-not-found
helm uninstall semantic-router \
  --namespace "$SEMANTIC_ROUTER_NAMESPACE" \
  --ignore-not-found
```

用 NVIDIA 部署流程中的名称删除 DGD 或 DGDR。若这是专用的 Dynamo 安装，最后移除平台：

```bash
helm uninstall dynamo-platform \
  --namespace "$DYNAMO_NAMESPACE" \
  --ignore-not-found
```

仅当 Envoy Gateway 是专为此设置安装时才卸载。日常应用清理不要删除共享命名空间或 CRD。

## 延伸阅读

- [NVIDIA Dynamo Kubernetes Quickstart](https://docs.nvidia.com/dynamo/dev/kubernetes/getting-started/quickstart)
- [NVIDIA Dynamo 模型部署概览](https://docs.nvidia.com/dynamo/dev/kubernetes/model-deployment/introduction)
- [NVIDIA Dynamo 发布产物](https://docs.nvidia.com/dynamo/dev/reference/release-artifacts)
- [Semantic Router Dynamo 集成文件](https://github.com/vllm-project/semantic-router/tree/main/deploy/kubernetes/dynamo)

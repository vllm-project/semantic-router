---
title: 使用 Envoy AI Gateway 部署
description: 由 Semantic Router 负责模型选择或 Responses 状态，由 Envoy AI Gateway 负责提供商和网关策略。
translation:
  source_commit: "29e20acc0914caf09f92de44be45473db8fbf5a5"
  source_file: "docs/installation/k8s/ai-gateway.md"
  outdated: false
---

# 使用 Envoy AI Gateway 部署

当 Envoy AI Gateway 已经负责南北向流量和提供商集成时，使用此拓扑。Semantic Router 可以根据请求含义选择模型，也可以仅作为 ExtProc 服务运行，用于 OpenAI Responses 状态和协议转换。Envoy AI Gateway 仍负责 Gateway API 资源、提供商凭证、速率限制和流量策略。

对于大型请求体或 Semantic Router 的流式立即响应，另见 [Streamed ExtProc 与立即响应](./streamed-extproc)。该指南说明如何将 ExtProc 过滤器的请求体从 `BUFFERED` 切换到 `STREAMED`，以及流式 Chat Completions 客户端如何接收 looper 或 `fast_response` 立即响应。

## 职责划分

部署包含：

- **Semantic Router** 评估所选配方，并选择逻辑模型或提供商别名。
- **Envoy Gateway** 提供 Kubernetes Gateway API 数据面。
- **Envoy AI Gateway** 转换提供商 API，并应用网关侧的认证、速率限制和流量策略。
- **模型提供商** 服务所选模型。本指南使用演示后端，不安装生产推理容量。

提供商支持独立于 Semantic Router 变化。请使用 [Envoy AI Gateway 提供商文档](https://aigateway.envoyproxy.io/docs/capabilities/llm-integrations/supported-providers/) 选择 `AIServiceBackend` 和凭证策略，再将提供商名称绑定到 Semantic Router 配置所用的别名。

## 仅提供 Responses 状态、不选择模型

当外部网关已经选择路由和后端，但客户端需要 Semantic Router 的 OpenAI Responses 实现时，使用 [`responses-state.yaml`](https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/ai-gateway/semantic-router-values/responses-state.yaml) 配置。该配置启用 `POST /v1/responses`、`GET /v1/responses/{id}` 和 `previous_response_id` 会话串联。它将 Responses 请求转换为声明的上游线格式，再将上游响应转换回 Responses 对象。

所有权边界如下：

| 关注点 | 所有者 |
| --- | --- |
| 公共监听器、Gateway/路由、提供商后端、凭证、速率限制和流量策略 | 外部网关 |
| Responses 对象存储、`previous_response_id` 展开，以及请求/响应协议转换 | Semantic Router ExtProc |

此配置有意不包含 Semantic Router 监听器、后端引用、凭证、路由信号、决策或可选插件。模型条目只是协议元数据。请使其名称和 `api_format` 值与外部网关接受的模型保持一致。Semantic Router 在协议转换时改写上游 API 路径，但不清除 Envoy 的路由缓存，因此网关所选路由仍然有效。

安装 chart 时，用仅状态配置替换步骤 2 中的模型选择 values：

```bash
helm install semantic-router oci://ghcr.io/vllm-project/charts/semantic-router \
  --version 0.0.0-latest \
  --namespace vllm-semantic-router-system \
  --create-namespace \
  -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/ai-gateway/semantic-router-values/responses-state.yaml
```

内置的内存响应存储面向单副本演示或本地验证；重启后状态会丢失。持久或复制的生产部署请配置 Response API Redis 后端。外部网关的 ExtProc 过滤器必须向 Semantic Router 发送缓冲的请求和响应头及请求体，其对 `/v1/responses` 的路由必须在请求体转换运行之前选定目标提供商后端。

## 前置条件

需要：

- Kubernetes `1.32` 或更高，以匹配固定的 Envoy AI Gateway `v1.0.x` 与 Envoy Gateway `v1.8.x` 兼容集；演示可用 [kind](https://kind.sigs.k8s.io/docs/user/quick-start/#installation)；
- Gateway API `v1.5.x` CRD。下面默认的 Envoy Gateway Helm 安装会安装兼容集合；若平台自行管理这些 CRD，安装 chart 前请核对版本；
- [kubectl](https://kubernetes.io/docs/tasks/tools/)；
- [Helm](https://helm.sh/docs/intro/install/)；以及
- 配置所用每个提供商的凭证和网络访问。

## 步骤 1：创建 Kind 集群（可选）

创建针对 Semantic Router 工作负载优化的本地 Kubernetes 集群：

```bash
kind create cluster --name semantic-router-cluster

# Verify cluster is ready
kubectl wait --for=condition=Ready nodes --all --timeout=300s
```

## 步骤 2：部署 vLLM Semantic Router

使用集成 values 部署 Semantic Router：

```bash
# Install with custom values from GHCR OCI registry
# (Optional) If you use a registry mirror/proxy, append: --set global.imageRegistry=<your-registry>
helm install semantic-router oci://ghcr.io/vllm-project/charts/semantic-router \
  --version 0.0.0-latest \
  --namespace vllm-semantic-router-system \
  --create-namespace \
  -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/ai-gateway/semantic-router-values/values.yaml

# Wait for deployment to be ready (this may take several minutes for model downloads)
kubectl wait --for=condition=Available deployment/semantic-router -n vllm-semantic-router-system --timeout=600s

# Verify deployment status
kubectl get pods -n vllm-semantic-router-system
```

**说明**：values 文件包含 Semantic Router 的模型 binding、信号、决策和路由规则。在适配真实提供商池之前，请下载并审阅 [values.yaml](https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/ai-gateway/semantic-router-values/values.yaml)。

## 步骤 3：安装 Envoy Gateway

安装 Envoy AI Gateway `v1.0.0` 所支持的 Envoy Gateway 版本：

```bash
export AIGW_VERSION=v1.0.0
export ENVOY_GATEWAY_VERSION=v1.8.1

helm upgrade -i eg oci://docker.io/envoyproxy/gateway-helm \
  --version "${ENVOY_GATEWAY_VERSION}" \
  --namespace envoy-gateway-system \
  --create-namespace \
  -f "https://raw.githubusercontent.com/envoyproxy/ai-gateway/${AIGW_VERSION}/manifests/envoy-gateway-values.yaml"

kubectl wait --timeout=2m -n envoy-gateway-system deployment/envoy-gateway --for=condition=Available
```

## 步骤 4：安装 Envoy AI Gateway

先安装 AI Gateway CRD，再安装控制器。这些版本遵循上游 [`v1.0.x` 兼容矩阵](https://aigateway.envoyproxy.io/docs/compatibility/)。

```bash
# Install Envoy AI Gateway CRDs
helm upgrade -i aieg-crd oci://docker.io/envoyproxy/ai-gateway-crds-helm \
  --version "${AIGW_VERSION}" \
  --namespace envoy-ai-gateway-system \
  --create-namespace

# Install the controller
helm upgrade -i aieg oci://docker.io/envoyproxy/ai-gateway-helm \
  --version "${AIGW_VERSION}" \
  --namespace envoy-ai-gateway-system

# Wait for AI Gateway Controller to be ready
kubectl wait --timeout=300s -n envoy-ai-gateway-system deployment/ai-gateway-controller --for=condition=Available
```

## 步骤 5：部署演示 LLM

创建演示 LLM，作为 Semantic Router 的后端：

```bash
# Deploy demo LLM
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/ai-gateway/aigw-resources/base-model.yaml
```

## 步骤 6：创建 Gateway API 资源

为 AI gateway 创建所需的 Gateway API 资源：

```bash
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/ai-gateway/aigw-resources/gwapi-resources.yaml
```

## 测试部署

### 方法 1：Port Forwarding（推荐用于本地测试）

设置 port forwarding，以便在本地访问网关：

```bash
# Get the Envoy service name
export ENVOY_SERVICE=$(kubectl get svc -n envoy-gateway-system \
  --selector=gateway.envoyproxy.io/owning-gateway-namespace=default,gateway.envoyproxy.io/owning-gateway-name=semantic-router \
  -o jsonpath='{.items[0].metadata.name}')

kubectl port-forward -n envoy-gateway-system svc/$ENVOY_SERVICE 8080:80
```

### 发送测试请求

网关可访问后，测试推理端点：

```bash
# Test math domain chat completions endpoint
curl -i -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vllm-sr/auto",
    "messages": [
      {"role": "user", "content": "What is the derivative of f(x) = x^3?"}
    ]
  }'
```

## 故障排查

### 常见问题

**网关无法访问：**

```bash
# Check gateway status
kubectl get gateway semantic-router -n default

# Check Envoy service
kubectl get svc -n envoy-gateway-system
```

**AI Gateway 控制器未就绪：**

```bash
# Check AI gateway controller logs
kubectl logs -n envoy-ai-gateway-system deployment/ai-gateway-controller

# Check controller status
kubectl get deployment -n envoy-ai-gateway-system
```

**Semantic Router 无响应：**

```bash
# Check semantic router pod status
kubectl get pods -n vllm-semantic-router-system

# Check semantic router logs
kubectl logs -n vllm-semantic-router-system deployment/semantic-router
```

## 清理

要移除整个部署：

```bash
# Remove Gateway API resources and Demo LLM
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/ai-gateway/aigw-resources/gwapi-resources.yaml
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/ai-gateway/aigw-resources/base-model.yaml

# Remove semantic router
helm uninstall semantic-router -n vllm-semantic-router-system

# Remove AI gateway
helm uninstall aieg -n envoy-ai-gateway-system
helm uninstall aieg-crd -n envoy-ai-gateway-system

# Remove Envoy gateway
helm uninstall eg -n envoy-gateway-system

# Delete kind cluster (optional)
kind delete cluster --name semantic-router-cluster
```

## 后续步骤

- 用网关团队负责的提供商资源和凭证替换演示后端。
- 保持 `AIGatewayRoute` 中的模型别名与 Semantic Router 发出的名称一致。
- 在暴露网关之前，在各自所属层添加认证、速率限制、可观测性和容量策略。

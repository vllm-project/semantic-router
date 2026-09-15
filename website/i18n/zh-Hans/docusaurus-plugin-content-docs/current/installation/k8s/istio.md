---
title: 使用 Istio Gateway 部署
description: 在 Istio Gateway 后面将 Semantic Router 作为 ExtProc 服务运行，并接入两个模型后端。
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/installation/k8s/istio.md"
  outdated: false
---

# 使用 Istio Gateway 部署

本指南展示参考拓扑：在 Istio Gateway 后面将 Semantic Router 作为 ExtProc 服务运行。Istio 负责入口和 `HTTPRoute` 处理；Semantic Router 负责感知提示词的模型选择。所提供的 `EnvoyFilter` 和 `DestinationRule` 仅适用于本示例，因此不要在未比较 ExtProc 模式的情况下复用其他网关集成的清单。

## 职责划分

部署包含：

- **Semantic Router** 评估路由策略并选择模型。
- **Istio Gateway** 接受客户端流量，并通过 ExtProc 调用 Semantic Router。
- **Gateway API 资源** 将所选模型映射到 Kubernetes 后端。
- **两个 vLLM 部署** 是示例推理后端，需要合适的算力和模型访问。

## 前置条件

需要：

- Kubernetes `1.31`–`1.35`，这是所固定的 Istio `1.29` 发行版的支持范围，并且为所提供的模型清单至少准备两个可调度的 NVIDIA GPU，或为替换后端准备等效容量；
- [kubectl](https://kubernetes.io/docs/tasks/tools/)；
- [Helm](https://helm.sh/docs/intro/install/)；以及
- [istioctl](https://istio.io/latest/docs/ops/diagnostic-tools/istioctl/)。

所提供的清单为两个 vLLM Deployment 各请求一个 `nvidia.com/gpu` 设备。它们固定本示例使用的 vLLM 镜像。你可以用其他 OpenAI 兼容后端替换，但 Router 提供商、Service 名称和 `HTTPRoute` 匹配必须一起更改。

## 步骤 1：验证集群

```bash
kubectl wait --for=condition=Ready nodes --all --timeout=300s
```

## 步骤 2：部署 LLM 模型

示例分别用独立的 vLLM 服务器部署 `meta-llama/Llama-3.1-8B-Instruct` 和 `microsoft/Phi-4-mini-instruct`。创建 Kubernetes Secret 前先导出 Hugging Face token。若使用不同的 OpenAI 兼容后端，请同时更新 Router values 和路由清单中的模型名称与端点引用。

```bash
kubectl create secret generic hf-token-secret --from-literal=token=$HF_TOKEN
```

```bash
# Create vLLM service running llama3-8b
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/istio/vLlama3.yaml
```

首次启动会下载模型权重，可能需要几分钟。部署第二个后端，然后等待两个 Deployment。

```bash
# Create vLLM service running phi4-mini
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/istio/vPhi4.yaml
```

```bash
kubectl wait --for=condition=Available deployment/llama-8b --timeout=900s
kubectl wait --for=condition=Available deployment/phi4-mini --timeout=900s
kubectl get pods,services
```

## 步骤 3：安装 Gateway API 和 Istio

此直接 Service 拓扑需要 Kubernetes Gateway API 和 Istio；不需要 Gateway API Inference Extension CRD。安装兼容的一对，并在部署自动化中固定版本：

```bash
export GATEWAY_API_VERSION=v1.5.1
export ISTIO_VERSION=1.29.6

kubectl apply --server-side \
  -f "https://github.com/kubernetes-sigs/gateway-api/releases/download/${GATEWAY_API_VERSION}/standard-install.yaml"

curl -L https://istio.io/downloadIstio | ISTIO_VERSION="${ISTIO_VERSION}" sh -
export PATH="$PWD/istio-${ISTIO_VERSION}/bin:$PATH"
istioctl install -y --set profile=minimal

kubectl wait --for=condition=Available deployment/istiod \
  -n istio-system --timeout=300s
```

## 步骤 4：更新 vsr 配置（可选）

Semantic Router 配置通过 Helm values 文件提供。若需要自定义配置（例如匹配不同的模型名称或端点），下载 values 文件并修改：

```bash
# Download the values file for customization
curl -O https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/istio/semantic-router-values/values.yaml
```

确保配置文件中的模型与你使用的模型匹配。通常应先从 vsr 的基础功能开始，例如提示词分类和模型路由，再试验 PromptGuard 或 ToolCalling 等其他功能。

## 步骤 5：部署 vLLM Semantic Router

使用集成 values 部署 Semantic Router：

```bash
# Install semantic router using Helm from GHCR OCI registry
helm install semantic-router oci://ghcr.io/vllm-project/charts/semantic-router \
  --version 0.0.0-latest \
  --namespace vllm-semantic-router-system \
  --create-namespace \
  -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/istio/semantic-router-values/values.yaml

# Wait for deployment to be ready (this may take several minutes for model downloads)
kubectl wait --for=condition=Available deployment/semantic-router -n vllm-semantic-router-system --timeout=600s

# Verify deployment status
kubectl get pods -n vllm-semantic-router-system
```

**说明**：values 文件包含提供商 binding、信号、决策和路由规则。在适配真实提供商池之前，请下载并审阅 [values.yaml](https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/istio/semantic-router-values/values.yaml)。保持 `global.router.clear_route_cache: true`：ExtProc 写入 `x-selected-model` 后，Envoy 必须丢弃先前路由，并再次评估基于头的 `HTTPRoute`。

## 步骤 6：安装额外的 Istio 配置

安装将 Istio 网关通过 ExtProc 连接到 Semantic Router 的 `DestinationRule` 和网关范围的 `EnvoyFilter`：

```bash
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/istio/destinationrule.yaml
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/istio/envoyfilter.yaml
```

示例过滤器设置 `failure_mode_allow: true`，在 ExtProc 不可用时让 Envoy 继续其 HTTP 过滤器链。这并不保证请求能到达后端：所提供的路由要求 `x-selected-model`，因此当 Router 未添加该头时，路由选择仍可能失败。真正的旁路需要刻意的兜底路由，这也会改变安全边界。当语义路由是授权或数据边界控制时，优先使用 fail-close 行为，并在生产使用前测试确切的中断路径。

## 步骤 7：安装网关路由

创建由 Istio 管理的 Gateway，然后安装两条 `HTTPRoute` 资源。

```bash
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/istio/gateway.yaml
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/istio/httproute-llama3-8b.yaml
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/istio/httproute-phi4-mini.yaml
```

## 步骤 8：测试部署

按[测试 Kubernetes Gateway 部署](gateway-testing) 解析实际网关 URL，比较直接请求和已路由模型请求，检查路由头，并验证所选后端。不要复制示例 IP 或端口；它们由集群分配。

## 故障排查

### 常见问题

**Gateway / 前端不工作：**

```bash
# Check istio gateway status
kubectl get gateway

# Check istio gw service status
kubectl get svc inference-gateway-istio

# Check Istio's Envoy logs
kubectl logs deploy/inference-gateway-istio -c istio-proxy
```

**Semantic Router 无响应：**

```bash
# Check semantic router pod
kubectl get pods -n vllm-semantic-router-system

# Check semantic router service
kubectl get svc -n vllm-semantic-router-system

# Check semantic router logs
kubectl logs -n vllm-semantic-router-system deployment/semantic-router
```

## 清理

要移除整个部署：

```bash
# Remove gateway routes
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/istio/httproute-llama3-8b.yaml
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/istio/httproute-phi4-mini.yaml
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/istio/gateway.yaml

# Remove Istio configuration
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/istio/envoyfilter.yaml
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/istio/destinationrule.yaml

# Remove semantic router
helm uninstall semantic-router -n vllm-semantic-router-system

# Remove Istio
istioctl uninstall --purge

# Remove LLMs
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/istio/vLlama3.yaml
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/istio/vPhi4.yaml

```

## 后续步骤

- 用已固定、由生产运维的后端替换示例模型。
- 添加认证、网络策略、可观测性和容量控制。
- 当每个所选模型需要 endpoint picker 而不是直接 Service 后端时，使用 [Gateway API Inference Extension 指南](gateway-api-inference-extension)。

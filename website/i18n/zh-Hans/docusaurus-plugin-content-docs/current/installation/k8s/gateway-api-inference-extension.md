---
translation:
  source_commit: "74867e20"
  source_file: "docs/installation/k8s/gateway-api-inference-extension.md"
  outdated: false
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 使用 Gateway API Inference Extension 安装

本指南分步介绍如何将 vLLM Semantic Router (vSR) 与符合 Gateway API Inference Extension (GIE) 规范的推理网关集成。GIE 让您可以通过 `InferencePool` 等 Kubernetes 原生 API 管理自托管的 OpenAI 兼容模型，而 vSR 则通过网关的 ExtProc 集成提供感知提示词的模型路由能力。

## 架构概览

该部署由三个主要组件构成：

- **vLLM Semantic Router**：对传入请求进行分类并选择目标模型。
- **符合 GIE 规范的推理网关**：提供 Kubernetes Gateway API 数据平面。本指南通过标签页分别介绍 Istio 和 agentgateway。
- **Gateway API Inference Extension (GIE)**：提供 `InferencePool` 等 Kubernetes 原生推理 API，用于感知负载的后端选择。

## 集成优势

将 vSR 与 GIE 集成，可为 LLM 服务提供一个稳健的 Kubernetes 原生解决方案，并带来以下几项主要优势：

### 1. **Kubernetes 原生 LLM 管理**

使用熟悉的自定义资源定义 (CRD)，直接通过 `kubectl` 管理模型、路由和扩缩容策略。

### 2. **智能模型与副本路由**

将 vSR 基于提示词的模型路由与 GIE 智能且感知负载的副本选择相结合。这样可以确保请求不仅会发送到正确的模型，还会发送到运行状况最佳的副本，并且整个过程只需一次高效转发。

### 3. **保护模型免受过载影响**

内置调度器会跟踪后端负载和请求队列，并自动卸载流量，防止模型服务器在高负载下崩溃。

### 4. **深度可观测性**

结合高层 Gateway 指标与详细的 vSR 性能数据（例如 token 用量和分类准确率），洞察、监控并排查整个 AI 技术栈的问题。

### 5. **安全多租户**

使用标准 Kubernetes 命名空间和 `HTTPRoute` 隔离租户工作负载。在共享安全的通用网关基础设施时，还可应用速率限制和其他策略。

## 支持的后端模型和 API

本指南中的演示模型使用 llm-d 推理模拟器，通过 **OpenAI 兼容 API** 模拟 Llama3 和 Phi-4。借助该模拟器，您可以在本地 kind 集群中运行本指南的操作流程，而无需 GPU 或下载模型。您可以将其替换为自己的模型服务器，前提是 Semantic Router 配置、网关路由资源和 GIE 后端配置在请求格式与后端目标上保持一致。

OpenAI 兼容 API 并非唯一支持的选项。agentgateway 自定义提供商可以声明提供商原生格式，例如 OpenAI 聊天补全 (chat completions)、Anthropic 消息 (messages)、响应 (responses)、嵌入 (embeddings)、token 计数 (token counting) 和实时 (realtime) API，并可将这些提供商路由到主机、Kubernetes `Service` 或 `InferencePool`。GIE 端点选择器 (endpoint picker) 实现还可以通过解析器配置支持其他请求格式；例如，llm-d EPP 解析器框架包含 OpenAI、Anthropic、vLLM HTTP、vLLM gRPC、Vertex AI 和透传 (passthrough) 解析器。

有关详细信息，请参阅 agentgateway 的[自定义提供商](https://agentgateway.dev/docs/kubernetes/main/llm/providers/custom/)指南和 llm-d EPP 的[请求解析器文档](https://github.com/llm-d/llm-d-router/blob/main/pkg/epp/framework/plugins/requesthandling/parsers/README.md)。

## 前置条件

开始之前，请确保已安装以下工具：

- [Docker](https://docs.docker.com/get-docker/) 或其他容器运行时。
- [kind](https://kind.sigs.k8s.io/) v0.22+ 或任意 Kubernetes 1.29+ 集群。
- [kubectl](https://kubernetes.io/docs/tasks/tools/) v1.30+。
- [Helm](https://helm.sh/) v3.14+。
- 如果选择 Istio 标签页，则需要 [istioctl](https://istio.io/latest/docs/ops/diagnostic-tools/istioctl/) v1.28+。

您可以使用以下命令验证工具链版本：

```bash
kind version
kubectl version --client --short
helm version --short
istioctl version --remote=false
```

## 步骤 1：创建 Kind 集群（可选）

如果您还没有 Kubernetes 集群，可以创建一个本地集群用于测试：

```bash
kind create cluster --name vsr-gie

# 验证集群已就绪
kubectl wait --for=condition=Ready nodes --all --timeout=300s
```

## 步骤 2：安装 Gateway API 和 GIE CRD

在安装网关实现之前，先安装 Gateway API 和 GIE 的共享 CRD：

```bash
export GATEWAY_API_VERSION=v1.5.0
export GIE_VERSION=v1.5.0

# 安装 Gateway API CRD
kubectl apply --server-side --force-conflicts \
  -f "https://github.com/kubernetes-sigs/gateway-api/releases/download/${GATEWAY_API_VERSION}/standard-install.yaml"

# 安装 Gateway API Inference Extension CRD
kubectl apply -f "https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/${GIE_VERSION}/manifests.yaml"

# 验证 CRD 已安装
kubectl get crd | grep 'gateway.networking.k8s.io'
kubectl get crd | grep 'inference.networking.k8s.io'
```

## 步骤 3：安装推理网关

选择您要使用的推理网关。每个标签页仅包含相应网关特有的安装步骤。

<Tabs groupId="gie-gateway" defaultValue="istio" values={[
  {label: 'Istio', value: 'istio'},
  {label: 'agentgateway', value: 'agentgateway'},
]}>
<TabItem value="istio">

安装支持 Gateway API 和外部处理的 Istio：

```bash
# 下载并安装 Istio
export ISTIO_VERSION=1.29.0
curl -L https://istio.io/downloadIstio | ISTIO_VERSION=$ISTIO_VERSION sh -
export PATH="$PWD/istio-$ISTIO_VERSION/bin:$PATH"
istioctl install -y --set profile=minimal --set values.pilot.env.ENABLE_GATEWAY_API=true

# 验证 Istio 已就绪
kubectl wait --for=condition=Available deployment/istiod \
  -n istio-system \
  --timeout=300s
```

</TabItem>
<TabItem value="agentgateway">

安装已启用推理扩展支持的 agentgateway：

```bash
export AGENTGATEWAY_VERSION=v1.3.0-alpha.1

# 安装 agentgateway CRD
helm upgrade -i --create-namespace \
  --namespace agentgateway-system \
  --version "${AGENTGATEWAY_VERSION}" \
  agentgateway-crds oci://cr.agentgateway.dev/charts/agentgateway-crds

# 安装已启用推理扩展支持的 agentgateway
helm upgrade -i -n agentgateway-system \
  agentgateway oci://cr.agentgateway.dev/charts/agentgateway \
  --version "${AGENTGATEWAY_VERSION}" \
  --set inferenceExtension.enabled=true

# 验证 agentgateway 已就绪
kubectl get pods -n agentgateway-system
```

本指南使用 agentgateway `v1.3.0-alpha.1` 或更高版本，以便 ExtProc 策略可以设置 `processingOptions` 和 `allowModeOverride`。

</TabItem>
</Tabs>

## 步骤 4：部署演示 LLM 服务器

部署两个轻量级推理模拟器实例 Llama3 和 Phi-4 作为后端。这些模拟器部署不需要 GPU 或 Hugging Face 令牌，其标签与本指南后续使用的 `InferencePool` 选择器相匹配。

```bash
# 部署模拟器模型服务器
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/inference-sim.yaml

# 等待模拟器模型就绪
kubectl wait --for=condition=Available deployment/vllm-llama3-8b-instruct --timeout=300s
kubectl wait --for=condition=Available deployment/phi4-mini --timeout=300s
```

该演示清单在 `default` 命名空间中运行，并公开名为 `vllm-llama3-8b-instruct` 和 `phi4-mini` 的服务。

## 步骤 5：部署 vLLM Semantic Router

使用官方 Helm chart 部署 vLLM Semantic Router。该组件作为 ExtProc 服务器运行，由所选网关调用以执行路由决策。

```bash
helm upgrade -i semantic-router oci://ghcr.io/vllm-project/charts/semantic-router \
  --version v0.0.0-latest \
  --namespace vllm-semantic-router-system \
  --create-namespace \
  -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/istio/semantic-router-values/values.yaml

# 等待路由器就绪
kubectl -n vllm-semantic-router-system wait \
  --for=condition=Available deploy/semantic-router \
  --timeout=10m
```

该 values 文件将 vSR 配置为：一般提示词选择 `llama3-8b`，数学提示词选择 `phi4-mini`。

## 步骤 6：部署 Gateway 和路由逻辑

应用网关特定资源，以创建面向外部的 `Gateway`、将 vSR 作为 ExtProc 服务附加到网关，并将选定的模型路由到 GIE `InferencePool`。

<Tabs groupId="gie-gateway" defaultValue="istio" values={[
  {label: 'Istio', value: 'istio'},
  {label: 'agentgateway', value: 'agentgateway'},
]}>
<TabItem value="istio">

应用现有的 Istio 网关、`InferencePool`、`HTTPRoute`、`DestinationRule` 和 `EnvoyFilter` 资源：

```bash
# 应用所有路由和网关资源
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/istio/gateway.yaml
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/inferencepool-llama.yaml
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/inferencepool-phi4.yaml
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/httproute-llama-pool.yaml
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/httproute-phi4-pool.yaml
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/istio/destinationrule.yaml
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/istio/envoyfilter.yaml

# 验证 Istio 已配置 Gateway
kubectl wait --for=condition=Programmed gateway/inference-gateway --timeout=120s
kubectl wait --for=condition=Available deployment/llm-d-inference-scheduler-llama3-8b --timeout=300s
kubectl wait --for=condition=Available deployment/llm-d-inference-scheduler-phi4-mini --timeout=300s
```

Istio 特有的 `EnvoyFilter` 会插入调用 `semantic-router` 服务的 ExtProc 过滤器。

</TabItem>
<TabItem value="agentgateway">

创建由 agentgateway 支持的 `Gateway`，应用共享的 GIE `InferencePool` 和 `HTTPRoute` 资源，并通过 `AgentgatewayPolicy` 附加 vSR：

```bash
# 创建 agentgateway 推理 Gateway
kubectl apply -f- <<'EOF'
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: inference-gateway
  namespace: default
spec:
  gatewayClassName: agentgateway
  listeners:
  - name: http
    protocol: HTTP
    port: 80
    allowedRoutes:
      namespaces:
        from: All
EOF

# 应用共享的 GIE 路由资源
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/inferencepool-llama.yaml
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/inferencepool-phi4.yaml
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/httproute-llama-pool.yaml
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/httproute-phi4-pool.yaml

# 将 Semantic Router 作为网关 ExtProc 服务附加到网关
kubectl apply -f- <<'EOF'
apiVersion: agentgateway.dev/v1alpha1
kind: AgentgatewayPolicy
metadata:
  name: semantic-router-extproc
  namespace: default
spec:
  targetRefs:
  - group: gateway.networking.k8s.io
    kind: Gateway
    name: inference-gateway
  traffic:
    phase: PreRouting
    extProc:
      backendRef:
        name: semantic-router
        namespace: vllm-semantic-router-system
        port: 50051
      processingOptions:
        requestHeaderMode: Send
        requestBodyMode: Buffered
        responseHeaderMode: Send
        responseBodyMode: Buffered
        allowModeOverride: true
EOF

# 验证 agentgateway 已配置 Gateway
kubectl wait --for=condition=Programmed gateway/inference-gateway --timeout=120s
kubectl wait --for=condition=Available deployment/llm-d-inference-scheduler-llama3-8b --timeout=300s
kubectl wait --for=condition=Available deployment/llm-d-inference-scheduler-phi4-mini --timeout=300s
```

agentgateway 使用 `AgentgatewayPolicy` 配置 ExtProc，因此不需要 Istio `DestinationRule` 或 `EnvoyFilter` 资源。该策略使用 `phase: PreRouting`，以便 vSR 在 agentgateway 计算 `HTTPRoute` 请求头匹配条件之前添加 `x-selected-model`。

</TabItem>
</Tabs>

## 测试部署

### 端口转发

设置端口转发，以便从本地计算机访问所选网关。

<Tabs groupId="gie-gateway" defaultValue="istio" values={[
  {label: 'Istio', value: 'istio'},
  {label: 'agentgateway', value: 'agentgateway'},
]}>
<TabItem value="istio">

```bash
# Gateway 服务名为 inference-gateway-istio
kubectl port-forward svc/inference-gateway-istio 8080:80
```

</TabItem>
<TabItem value="agentgateway">

```bash
# agentgateway 为 Gateway 创建服务
kubectl port-forward svc/inference-gateway 8080:80
```

</TabItem>
</Tabs>

### 发送测试请求

端口转发启动后，向 `localhost:8080` 发送 OpenAI 兼容请求。

**测试 1：显式请求模型**

此请求应由 Llama 模拟器处理。如果要检查 `x-inference-pod` 和 `x-vsr-selected-model` 等响应头，请在命令中添加 `-i`。

```bash
curl -sS http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "llama3-8b",
    "messages": [{"role": "user", "content": "Summarize the Kubernetes Gateway API in three sentences."}]
  }'
```

**测试 2：让 Semantic Router 选择模型**

通过设置 `"model": "auto"`，您可以让 vSR 对提示词进行分类。它会将此请求识别为数学查询并添加 `x-selected-model: phi4-mini` 请求头，随后 `HTTPRoute` 使用该请求头将请求路由到 Phi-4 `InferencePool`。在 agentgateway 标签页中，`AgentgatewayPolicy` 在 `PreRouting` 阶段运行，因此在路由匹配之前该请求头就已存在。

```bash
curl -sS http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "What is 2+2 * (5-1)?"}],
    "max_tokens": 64
  }'
```

## 故障排除

**问题：缺少 CRD**

如果看到类似 `no matches for kind "InferencePool"` 的错误，请检查是否已安装 CRD。

```bash
# 检查 GIE CRD
kubectl get crd | grep inference.networking.k8s.io
```

**问题：Gateway 未就绪**

如果 `kubectl port-forward` 失败或请求超时，请检查 Gateway 状态。

```bash
# Programmed 条件应为 True
kubectl get gateway inference-gateway -o yaml
```

<Tabs groupId="gie-gateway" defaultValue="istio" values={[
  {label: 'Istio', value: 'istio'},
  {label: 'agentgateway', value: 'agentgateway'},
]}>
<TabItem value="istio">

**问题：未调用 vSR**

如果请求可以正常处理，但路由似乎不正确，请检查 Istio 代理日志中是否存在 `ext_proc` 错误。

```bash
# 获取 Istio 网关 Pod 名称
export ISTIO_GW_POD=$(kubectl get pod -l istio=ingressgateway -o jsonpath='{.items[0].metadata.name}')

# 检查其日志
kubectl logs $ISTIO_GW_POD -c istio-proxy | grep ext_proc
```

</TabItem>
<TabItem value="agentgateway">

**问题：agentgateway 或 ExtProc 策略未就绪**

检查 agentgateway 控制器、生成的 Gateway 工作负载和 `AgentgatewayPolicy`。

```bash
kubectl get pods -n agentgateway-system
kubectl get deployment inference-gateway
kubectl logs -n agentgateway-system deployment/agentgateway
kubectl describe agentgatewaypolicy semantic-router-extproc
```

</TabItem>
</Tabs>

**问题：请求失败**

检查 vSR 和后端模型的日志。

```bash
# 检查 vSR 日志
kubectl logs deploy/semantic-router -n vllm-semantic-router-system

# 检查后端日志
kubectl logs deployment/vllm-llama3-8b-instruct
kubectl logs deployment/phi4-mini
```

## 清理

要删除本指南中创建的所有资源，请运行与您所用网关标签页对应的清理命令。

<Tabs groupId="gie-gateway" defaultValue="istio" values={[
  {label: 'Istio', value: 'istio'},
  {label: 'agentgateway', value: 'agentgateway'},
]}>
<TabItem value="istio">

```bash
# 删除路由和网关资源
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/istio/envoyfilter.yaml
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/istio/destinationrule.yaml
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/httproute-phi4-pool.yaml
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/httproute-llama-pool.yaml
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/inferencepool-phi4.yaml
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/inferencepool-llama.yaml
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/istio/gateway.yaml

# 删除演示模型
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/inference-sim.yaml

# 卸载 Helm release 和 Istio
helm uninstall semantic-router -n vllm-semantic-router-system
istioctl uninstall -y --purge

# 删除 kind 集群（如果已创建）
kind delete cluster --name vsr-gie
```

</TabItem>
<TabItem value="agentgateway">

```bash
# 删除网关特有资源
kubectl delete agentgatewaypolicy semantic-router-extproc --ignore-not-found
kubectl delete gateway inference-gateway --ignore-not-found

# 删除共享的 GIE 路由资源
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/httproute-phi4-pool.yaml
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/httproute-llama-pool.yaml
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/inferencepool-phi4.yaml
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/inferencepool-llama.yaml

# 删除演示模型
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/inference-sim.yaml

# 卸载 Helm release
helm uninstall semantic-router -n vllm-semantic-router-system
helm uninstall agentgateway -n agentgateway-system
helm uninstall agentgateway-crds -n agentgateway-system

# 删除 kind 集群（如果已创建）
kind delete cluster --name vsr-gie
```

</TabItem>
</Tabs>

## 后续步骤

- **自定义路由**：修改 `semantic-router` Helm chart 的 `values.yaml` 文件，定义您自己的路由类别和规则。
- **添加您自己的模型**：将演示用的 Llama3 和 Phi-4 部署替换为您自己的 OpenAI 兼容模型服务器。
- **探索 GIE 高级功能**：研究如何使用 `InferenceObjective` 实现更高级的自动扩缩容和调度策略。
- **监控性能**：将 Gateway 和 vSR 与 Prometheus 和 Grafana 集成，以构建监控仪表板。

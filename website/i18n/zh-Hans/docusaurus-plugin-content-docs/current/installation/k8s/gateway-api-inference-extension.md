# 使用 Gateway API Inference Extension 安装

本指南提供了将 vLLM Semantic Router (vSR) 与 Istio 和 Kubernetes Gateway API Inference Extension (GIE) 集成的分步说明。这种强大的组合允许您使用 Kubernetes 原生 API 管理自托管的 OpenAI 兼容模型，实现高级的 load-aware routing。

## 架构概览

部署包含三个主要组件：

- **vLLM Semantic Router**：基于请求内容对传入请求进行分类的智能核心。
- **Istio & Gateway API**：网络网格和所有进入集群流量的前门。
- **Gateway API Inference Extension (GIE)**：用于管理和扩展自托管模型后端的 Kubernetes 原生 API 集（`InferencePool` 等）。

## 集成优势

将 vSR 与 Istio 和 GIE 集成，为服务 LLM 提供了一个强大的 Kubernetes 原生解决方案，具有以下关键优势：

### 1. **Kubernetes 原生 LLM 管理**

使用熟悉的自定义资源定义 (CRD) 通过 `kubectl` 直接管理您的模型、路由和扩展策略。

### 2. **智能模型和副本路由**

结合 vSR 基于提示词的模型路由与 GIE 的智能负载感知副本选择。这确保请求不仅发送到正确的模型，还发送到最健康的副本，一次高效跳转完成所有操作。

### 3. **保护模型免受过载**

内置调度器跟踪 GPU 负载和请求队列，在高需求时自动卸载流量，防止模型服务器崩溃。

### 4. **深度可观测性**

从高级别 Gateway 指标和详细的 vSR 性能数据（如 token 使用和分类准确性）获取洞察，以监控和排查整个 AI 堆栈。

### 5. **安全的多租户**

使用标准 Kubernetes 命名空间和 `HTTPRoute` 隔离租户工作负载。在共享公共安全网关基础设施的同时应用速率限制和其他策略。

## 支持的后端模型

此架构旨在与任何暴露 **OpenAI 兼容 API** 的自托管模型配合使用。本指南中的演示模型使用 `vLLM` 服务 Llama3 和 Phi-3，但您可以轻松地用自己的模型服务器替换它们。

## 前置条件

开始之前，请确保已安装以下工具：

- [Docker](https://docs.docker.com/get-docker/) 或其他容器运行时。
- [kind](https://kind.sigs.k8s.io/) v0.22+ 或任何 Kubernetes 1.29+ 集群。
- [kubectl](https://kubernetes.io/docs/tasks/tools/) v1.30+。
- [Helm](https://helm.sh/) v3.14+。
- [istioctl](https://istio.io/latest/docs/ops/diagnostic-tools/istioctl/) v1.28+。
- 存储在 `HF_TOKEN` 环境变量中的 Hugging Face 令牌，示例 vLLM 部署需要它来下载模型。

您可以使用以下命令验证工具链版本：

```bash
kind version
kubectl version --client --short
helm version --short
istioctl version --remote=false
```

## 步骤 1：创建 Kind 集群（可选）

如果您没有 Kubernetes 集群，可以创建一个本地集群进行测试：

```bash
kind create cluster --name vsr-gie

# 验证集群就绪
kubectl wait --for=condition=Ready nodes --all --timeout=300s
```

## 步骤 2：安装 Istio

安装支持 Gateway API 和外部处理的 Istio：

```bash
# 下载并安装 Istio
export ISTIO_VERSION=1.29.0
curl -L https://istio.io/downloadIstio | ISTIO_VERSION=$ISTIO_VERSION sh -
export PATH="$PWD/istio-$ISTIO_VERSION/bin:$PATH"
istioctl install -y --set profile=minimal --set values.pilot.env.ENABLE_GATEWAY_API=true

# 验证 Istio 就绪
kubectl wait --for=condition=Available deployment/istiod -n istio-system --timeout=300s
```

## 步骤 3：安装 Gateway API & GIE CRDs

安装标准 Gateway API 和 Inference Extension 的自定义资源定义 (CRD)：

```bash
# 安装 Gateway API CRDs
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.2.0/standard-install.yaml

# 安装 Gateway API Inference Extension CRDs
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/v1.1.0/manifests.yaml

# 验证 CRDs 已安装
kubectl get crd | grep 'gateway.networking.k8s.io'
kubectl get crd | grep 'inference.networking.k8s.io'
```

## 步骤 4：部署演示 LLM 服务器

部署两个 `vLLM` 实例（Llama3 和 Phi-3）作为后端。它们将从 Hugging Face 自动下载。

```bash
# 为模型创建命名空间和密钥
kubectl create namespace llm-backends --dry-run=client -o yaml | kubectl apply -f -
kubectl -n llm-backends create secret generic hf-token --from-literal=token=$HF_TOKEN

# 部署模型服务器
kubectl -n llm-backends apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/istio/vLlama3.yaml
kubectl -n llm-backends apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/istio/vPhi4.yaml

# 等待模型就绪（可能需要几分钟）
kubectl -n llm-backends wait --for=condition=Ready pods --all --timeout=10m
```

## 步骤 5：部署 vLLM Semantic Router

使用官方 Helm chart 部署 vLLM Semantic Router。此组件将作为 `ext_proc` 服务器运行，Istio 调用它进行路由决策。

```bash
helm upgrade -i semantic-router oci://ghcr.io/vllm-project/charts/semantic-router \
  --version v0.0.0-latest \
  --namespace vllm-semantic-router-system \
  --create-namespace \
  -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/ai-gateway/semantic-router-values/values.yaml

# 等待路由就绪
kubectl -n vllm-semantic-router-system wait --for=condition=Available deploy/semantic-router --timeout=10m
```

## 步骤 6：部署 Gateway 和路由逻辑

应用最后一组资源来创建面向公众的 Gateway 并将所有组件连接在一起。这包括 `Gateway`、GIE 的 `InferencePools`、流量匹配的 `HTTPRoutes` 和 Istio 的 `EnvoyFilter`。

```bash
# 应用所有路由和网关资源
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/istio/gateway.yaml
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/inferencepool-llama.yaml
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/inferencepool-phi4.yaml
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/httproute-llama-pool.yaml
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/httproute-phi4-pool.yaml
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/istio/destinationrule.yaml
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/istio/envoyfilter.yaml

# 验证 Gateway 已被 Istio 编程
kubectl wait --for=condition=Programmed gateway/inference-gateway --timeout=120s
```

## 测试部署

### 方式 1：端口转发

设置端口转发以从本地机器访问网关。

```bash
# Gateway 服务名为 'inference-gateway-istio'，位于 default 命名空间
kubectl port-forward svc/inference-gateway-istio 8080:80
```

### 发送测试请求

端口转发激活后，您可以向 `localhost:8080` 发送 OpenAI 兼容请求。

**测试 1：显式请求模型**
此请求绕过 Semantic Router 的逻辑，直接发送到指定的模型池。

```bash
curl -sS http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "llama3-8b",
    "messages": [{"role": "user", "content": "Summarize the Kubernetes Gateway API in three sentences."}]
  }'
```

**测试 2：让 Semantic Router 选择模型**
通过设置 `"model": "auto"`，您让 vSR 对 prompt 进行分类。它将识别这是一个"math"查询并添加 `x-selected-model: phi4-mini` header，`HTTPRoute` 使用它来路由请求。

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

**问题：CRDs 缺失**
如果看到类似 `no matches for kind "InferencePool"` 的错误，检查 CRDs 是否已安装。

```bash
# 检查 GIE CRDs
kubectl get crd | grep inference.networking.k8s.io
```

**问题：Gateway 未就绪**
如果 `kubectl port-forward` 失败或请求超时，检查 Gateway 状态。

```bash
# "Programmed" 条件应为 "True"
kubectl get gateway inference-gateway -o yaml
```

**问题：vSR 未被调用**
如果请求正常但路由似乎不正确，检查 Istio 代理日志中的 `ext_proc` 错误。

```bash
# 获取 Istio gateway pod 名称
export ISTIO_GW_POD=$(kubectl get pod -l istio=ingressgateway -o jsonpath='{.items[0].metadata.name}')

# 检查其日志
kubectl logs $ISTIO_GW_POD -c istio-proxy | grep ext_proc
```

**问题：请求失败**
检查 vLLM Semantic Router 和后端模型的日志。

```bash
# 检查 vSR 日志
kubectl logs deploy/semantic-router -n vllm-semantic-router-system

# 检查 Llama3 后端日志
kubectl logs -n llm-backends -l app=vllm-llama3-8b-instruct
```

## 清理

要删除本指南中创建的所有资源，运行以下命令。

```bash
# 1. 删除所有已应用的 Kubernetes 资源
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/istio/gateway.yaml
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/inferencepool-llama.yaml
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/inferencepool-phi4.yaml
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/httproute-llama-pool.yaml
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/httproute-phi4-pool.yaml
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/istio/destinationrule.yaml
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/istio/envoyfilter.yaml
kubectl delete ns llm-backends

# 2. 卸载 Helm releases
helm uninstall semantic-router -n vllm-semantic-router-system

# 3. 卸载 Istio
istioctl uninstall -y --purge

# 4. 删除 kind 集群（可选）
kind delete cluster --name vsr-gie
```

## 后续步骤

- **自定义路由**：修改 `semantic-router` Helm chart 的 `values.yaml` 文件以定义您自己的路由类别和规则。
- **添加您自己的模型**：用您自己的 OpenAI 兼容模型服务器替换演示 Llama3 和 Phi-3 部署。
- **探索高级 GIE 功能**：查看使用 `InferenceObjective` 实现更高级的自动扩缩和调度策略。
- **监控性能**：将您的 Gateway 和 vSR 与 Prometheus 和 Grafana 集成以构建监控仪表板。

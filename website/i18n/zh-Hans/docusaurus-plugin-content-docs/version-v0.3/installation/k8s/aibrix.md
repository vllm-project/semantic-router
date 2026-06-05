# 使用 vLLM AIBrix 安装

本指南提供了集成 vLLM AIBrix 的分步说明。

## 关于 vLLM AIBrix

[vLLM AIBrix](https://github.com/vllm-project/aibrix) 是一个开源项目，旨在提供构建可扩展 GenAI 推理基础设施的基本构建块。AIBrix 提供了一个云原生解决方案，专为部署、管理和扩展大语言模型（LLM）推理而优化，专门针对企业需求量身定制。

### 主要功能

- **高密度 LoRA 管理**：简化对模型轻量级低秩适配的支持
- **LLM 网关和路由**：高效管理和引导跨多个模型和副本的流量
- **LLM 应用定制自动扩缩器**：根据实时需求动态扩展推理资源
- **统一 AI 运行时**：多功能边车，支持指标标准化、模型下载和管理
- **分布式推理**：可扩展架构，处理跨多节点的大型工作负载
- **分布式 KV 缓存**：支持高容量、跨引擎 KV 重用
- **经济高效的异构服务**：支持混合 GPU 推理以降低成本，同时保证 SLO
- **GPU 硬件故障检测**：主动检测 GPU 硬件问题

### 集成优势

将 vLLM Semantic Router 与 AIBrix 集成提供了多项优势：

1. **智能请求路由**：Semantic Router 基于内容理解分析传入请求并将其路由到最合适的模型，而 AIBrix 的网关高效管理跨模型副本的流量分配

2. **增强的可扩展性**：AIBrix 的自动扩缩器与 Semantic Router 无缝协作，根据路由模式和实时需求动态调整资源

3. **成本优化**：通过结合 Semantic Router 的智能路由与 AIBrix 的异构服务能力，您可以优化 GPU 利用率并降低基础设施成本，同时保持 SLO 保证

4. **生产就绪的基础设施**：AIBrix 提供企业级功能，如分布式 KV 缓存、GPU 故障检测和统一运行时管理，使在生产环境中部署 Semantic Router 更加容易

5. **简化运维**：集成利用 Kubernetes 原生模式和 Gateway API 资源，为 DevOps 团队提供熟悉的运维模式

## 前置条件

开始之前，请确保已安装以下工具：

- [kind](https://kind.sigs.k8s.io/docs/user/quick-start/#installation) - Kubernetes in Docker（可选）
- [kubectl](https://kubernetes.io/docs/tasks/tools/) - Kubernetes CLI
- [Helm](https://helm.sh/docs/intro/install/) - Kubernetes 包管理器

## 步骤 1：创建 Kind 集群（可选）

创建针对 Semantic Router 工作负载优化的本地 Kubernetes 集群：

```bash
kind create cluster --name semantic-router-cluster

# 验证集群就绪
kubectl wait --for=condition=Ready nodes --all --timeout=300s
```

## 步骤 2：部署 vLLM Semantic Router 

使用 Helm 部署包含所有必需组件的 Semantic Router 服务：

```bash
# 从 GHCR OCI 仓库安装自定义配置
# （可选）如果使用镜像代理，请添加：--set global.imageRegistry=<your-registry>
helm install semantic-router oci://ghcr.io/vllm-project/charts/semantic-router \
  --version v0.0.0-latest \
  --namespace vllm-semantic-router-system \
  --create-namespace \
  -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/aibrix/semantic-router-values/values.yaml

# 等待部署就绪（模型下载可能需要几分钟）
kubectl wait --for=condition=Available deployment/semantic-router -n vllm-semantic-router-system --timeout=600s

# 验证部署状态
kubectl get pods -n vllm-semantic-router-system
```

**注意**：values 文件包含 Semantic Router 的配置，包括模型设置、类别和路由规则。您可以从 [values.yaml](https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/aibrix/semantic-router-values/values.yaml) 下载并自定义。

## 步骤 3：安装 vLLM AIBrix

安装核心 vLLM AIBrix 组件：

```bash
# 安装 vLLM AIBrix
kubectl create -f https://github.com/vllm-project/aibrix/releases/download/v0.4.1/aibrix-dependency-v0.4.1.yaml

kubectl create -f https://github.com/vllm-project/aibrix/releases/download/v0.4.1/aibrix-core-v0.4.1.yaml

# 等待部署就绪
kubectl wait --timeout=2m -n aibrix-system deployment/aibrix-gateway-plugins --for=condition=Available
kubectl wait --timeout=2m -n aibrix-system deployment/aibrix-metadata-service --for=condition=Available
kubectl wait --timeout=2m -n aibrix-system deployment/aibrix-controller-manager --for=condition=Available
```

## 步骤 4：部署演示 LLM

创建一个演示 LLM 作为 Semantic Router 的后端：

```bash
# 部署演示 LLM
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/aibrix/aigw-resources/base-model.yaml

kubectl wait --timeout=2m -n default deployment/vllm-llama3-8b-instruct --for=condition=Available
```

## 步骤 5：创建 Gateway API 资源

为 envoy gateway 创建必要的 Gateway API 资源：

```bash
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/aibrix/aigw-resources/gwapi-resources.yaml
```

## 测试部署

### 方式 1：端口转发（推荐用于本地测试）

设置端口转发以在本地访问网关：

```bash
# 获取 Envoy 服务名称
export ENVOY_SERVICE=$(kubectl get svc -n envoy-gateway-system \
  --selector=gateway.envoyproxy.io/owning-gateway-namespace=aibrix-system,gateway.envoyproxy.io/owning-gateway-name=aibrix-eg \
  -o jsonpath='{.items[0].metadata.name}')

kubectl port-forward -n envoy-gateway-system svc/$ENVOY_SERVICE 8080:80
```

### 发送测试请求

网关可访问后，测试推理端点：

```bash
# 测试数学领域的对话补全端点
curl -i -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MoM",
    "messages": [
      {"role": "user", "content": "What is the derivative of f(x) = x^3?"}
    ]
  }'
```

您将看到演示 LLM 的响应，以及 Semantic Router 注入的额外响应头。

```bash
HTTP/1.1 200 OK
server: fasthttp
date: Thu, 06 Nov 2025 06:38:08 GMT
content-type: application/json
x-inference-pod: vllm-llama3-8b-instruct-984659dbb-gp5l9
x-went-into-req-headers: true
request-id: b46b6f7b-5645-470f-9868-0dd8b99a7163
x-vsr-selected-category: math
x-vsr-selected-reasoning: on
x-vsr-selected-model: vllm-llama3-8b-instruct
x-vsr-injected-system-prompt: true
transfer-encoding: chunked

{"id":"chatcmpl-f390a0c6-b38f-4a73-b019-9374a3c5d69b","created":1762411088,"model":"vllm-llama3-8b-instruct","usage":{"prompt_tokens":42,"completion_tokens":48,"total_tokens":90},"object":"chat.completion","do_remote_decode":false,"do_remote_prefill":false,"remote_block_ids":null,"remote_engine_id":"","remote_host":"","remote_port":0,"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"I am your AI assistant, how can I help you today? To be or not to be that is the question. Alas, poor Yorick! I knew him, Horatio: A fellow of infinite jest Testing, testing 1,2,3"}}]}
```

## 清理

删除整个部署：

```bash
# 删除 Gateway API 资源和演示 LLM
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/aibrix/aigw-resources/gwapi-resources.yaml
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/aibrix/aigw-resources/base-model.yaml

# 删除 Semantic Router 
helm uninstall semantic-router -n vllm-semantic-router-system

# 删除 kind 集群（可选）
kind delete cluster --name semantic-router-cluster
```

## 后续步骤

- 设置监控和可观测性
- 实施身份验证和授权
- 为生产工作负载扩展Semantic Router 部署

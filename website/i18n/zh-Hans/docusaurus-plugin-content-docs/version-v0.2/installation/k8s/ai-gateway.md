# 使用 Envoy AI Gateway 安装

本指南提供了在 Kubernetes 上将 vLLM Semantic Router 与 Envoy AI Gateway 集成的分步说明，以实现高级流量管理和 AI 特定功能。

## 架构概览

部署包含以下组件：

- **vLLM Semantic Router**：提供智能请求路由和语义理解
- **Envoy Gateway**：核心网关功能和流量管理
- **Envoy AI Gateway**：基于 Envoy Gateway 构建的 LLM Provider AI Gateway

## 集成优势

将 vLLM Semantic Router 与 Envoy AI Gateway 集成，为生产级 LLM 部署提供企业级能力：

### 1. **混合模型选择**

在云端 LLM 提供商（OpenAI、Anthropic 等）和自托管模型之间无缝路由请求。

### 2. **Token 速率限制**

通过细粒度速率限制保护您的基础设施并控制成本：

- **输入 token 限制**：控制请求大小以防止滥用
- **输出 token 限制**：管理响应生成成本
- **总 token 限制**：为每个用户/租户设置总体使用配额
- **基于时间窗口**：配置每秒、每分钟或每小时的限制

### 3. **模型/提供商故障转移**

通过自动故障转移机制确保高可用性：

- 检测不健康的后端并将流量路由到健康实例
- 支持主动-被动和主动-主动故障转移策略
- 当主要模型不可用时优雅降级

### 4. **流量分割和金丝雀测试**

通过渐进式发布能力安全部署新模型：

- **A/B 测试**：在模型版本之间分割流量以比较性能
- **金丝雀部署**：逐步将流量转移到新模型（例如 5% → 25% → 50% → 100%）
- **影子流量**：向新模型发送重复请求而不影响生产
- **基于权重的路由**：微调跨模型变体的流量分配

### 5. **LLM 可观测性与监控**

深入了解您的 LLM 基础设施：

- **请求/响应指标**：跟踪延迟、吞吐量、token 使用和错误率
- **模型性能**：监控准确性、质量评分和用户满意度
- **成本分析**：分析跨模型和提供商的支出模式
- **分布式追踪**：通过 OpenTelemetry 集成实现端到端可见性
- **自定义仪表板**：在 Prometheus、Grafana 或您首选的监控堆栈中可视化指标

## 支持的 LLM 提供商

| 提供商名称                                                | [AIServiceBackend](https://aigateway.envoyproxy.io/docs/api/#aiservicebackendspec) 上的 API Schema 配置 | [BackendSecurityPolicy](https://aigateway.envoyproxy.io/docs/api/#backendsecuritypolicyspec) 上的上游认证配置 | 状态 |
| ------------------------------------------------------------ | :----------------------------------------------------------: | :----------------------------------------------------------: | :----: |
| [OpenAI](https://platform.openai.com/docs/api-reference)     |              `{"name":"OpenAI","version":"v1"}`              | [API Key](https://aigateway.envoyproxy.io/docs/api/#backendsecuritypolicyapikey) |   ✅    |
| [AWS Bedrock](https://docs.aws.amazon.com/bedrock/latest/APIReference/) |                   `{"name":"AWSBedrock"}`                    | [AWS Bedrock 凭证](https://aigateway.envoyproxy.io/docs/api/#backendsecuritypolicyawscredentials) |   ✅    |
| [Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-services/openai/reference) | `{"name":"AzureOpenAI","version":"2025-01-01-preview"}` 或 `{"name":"OpenAI", "version": "openai/v1"}` | [Azure 凭证](https://aigateway.envoyproxy.io/docs/api/#backendsecuritypolicyazurecredentials) 或 [Azure API Key](https://aigateway.envoyproxy.io/docs/api/#backendsecuritypolicyazureapikey) |   ✅    |
| [Google Gemini on AI Studio](https://ai.google.dev/gemini-api/docs/openai) |        `{"name":"OpenAI","version":"v1beta/openai"}`         | [API Key](https://aigateway.envoyproxy.io/docs/api/#backendsecuritypolicyapikey) |   ✅    |
| [Google Vertex AI](https://cloud.google.com/vertex-ai/docs/reference/rest) |                   `{"name":"GCPVertexAI"}`                   | [GCP 凭证](https://aigateway.envoyproxy.io/docs/api/#backendsecuritypolicygcpcredentials) |   ✅    |
| [Anthropic on GCP Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude) |   `{"name":"GCPAnthropic", "version":"vertex-2023-10-16"}`   | [GCP 凭证](https://aigateway.envoyproxy.io/docs/api/#backendsecuritypolicygcpcredentials) |   ✅    |
| [Groq](https://console.groq.com/docs/openai)                 |          `{"name":"OpenAI","version":"openai/v1"}`           | [API Key](https://aigateway.envoyproxy.io/docs/api/#backendsecuritypolicyapikey) |   ✅    |
| [Grok](https://docs.x.ai/docs/api-reference?utm_source=chatgpt.com#chat-completions) |              `{"name":"OpenAI","version":"v1"}`              | [API Key](https://aigateway.envoyproxy.io/docs/api/#backendsecuritypolicyapikey) |   ✅    |
| [Together AI](https://docs.together.ai/docs/openai-api-compatibility) |              `{"name":"OpenAI","version":"v1"}`              | [API Key](https://aigateway.envoyproxy.io/docs/api/#backendsecuritypolicyapikey) |   ✅    |
| [Cohere](https://docs.cohere.com/v2/docs/compatibility-api)  | `{"name":"Cohere","version":"v2"}` 或 `{"name":"OpenAI","version":"v1"}` | [API Key](https://aigateway.envoyproxy.io/docs/api/#backendsecuritypolicyapikey) |   ✅    |
| [Mistral](https://docs.mistral.ai/api/#tag/chat/operation/chat_completion_v1_chat_completions_post) |              `{"name":"OpenAI","version":"v1"}`              | [API Key](https://aigateway.envoyproxy.io/docs/api/#backendsecuritypolicyapikey) |   ✅    |
| [DeepInfra](https://deepinfra.com/docs/inference)            |          `{"name":"OpenAI","version":"v1/openai"}`           | [API Key](https://aigateway.envoyproxy.io/docs/api/#backendsecuritypolicyapikey) |   ✅    |
| [DeepSeek](https://api-docs.deepseek.com/)                   |              `{"name":"OpenAI","version":"v1"}`              | [API Key](https://aigateway.envoyproxy.io/docs/api/#backendsecuritypolicyapikey) |   ✅    |
| [Hunyuan](https://cloud.tencent.com/document/product/1729/111007) |              `{"name":"OpenAI","version":"v1"}`              | [API Key](https://aigateway.envoyproxy.io/docs/api/#backendsecuritypolicyapikey) |   ✅    |
| [Tencent LLM Knowledge Engine](https://www.tencentcloud.com/document/product/1255/70381?lang=en) |              `{"name":"OpenAI","version":"v1"}`              | [API Key](https://aigateway.envoyproxy.io/docs/api/#backendsecuritypolicyapikey) |   ✅    |
| [Tetrate Agent Router Service (TARS)](https://router.tetrate.ai/) |              `{"name":"OpenAI","version":"v1"}`              | [API Key](https://aigateway.envoyproxy.io/docs/api/#backendsecuritypolicyapikey) |   ✅    |
| [SambaNova](https://docs.sambanova.ai/sambastudio/latest/open-ai-api.html) |              `{"name":"OpenAI","version":"v1"}`              | [API Key](https://aigateway.envoyproxy.io/docs/api/#backendsecuritypolicyapikey) |   ✅    |
| [Anthropic](https://docs.claude.com/en/home)                 |                    `{"name":"Anthropic"}`                    | [Anthropic API Key](https://aigateway.envoyproxy.io/docs/api/#backendsecuritypolicyanthropicapikey) |   ✅    |
| 自托管模型                                           |              `{"name":"OpenAI","version":"v1"}`              |                             无                              |   ✅    |

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
  -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/ai-gateway/semantic-router-values/values.yaml

# 等待部署就绪（模型下载可能需要几分钟）
kubectl wait --for=condition=Available deployment/semantic-router -n vllm-semantic-router-system --timeout=600s

# 验证部署状态
kubectl get pods -n vllm-semantic-router-system
```

**注意**：values 文件包含 Semantic Router 的配置，包括模型设置、类别和路由规则。您可以从 [values.yaml](https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/ai-gateway/semantic-router-values/values.yaml) 下载并自定义。

## 步骤 3：安装 Envoy Gateway

安装用于流量管理的核心 Envoy Gateway：

```bash
# 使用 Helm 安装 Envoy Gateway
helm upgrade -i eg oci://docker.io/envoyproxy/gateway-helm \
  --version v0.0.0-latest \
  --namespace envoy-gateway-system \
  --create-namespace \
  -f https://raw.githubusercontent.com/envoyproxy/ai-gateway/main/manifests/envoy-gateway-values.yaml

kubectl wait --timeout=2m -n envoy-gateway-system deployment/envoy-gateway --for=condition=Available
```

## 步骤 4：安装 Envoy AI Gateway

安装用于推理工作负载的 AI 特定扩展：

```bash
# 使用 Helm 安装 Envoy AI Gateway
helm upgrade -i aieg oci://docker.io/envoyproxy/ai-gateway-helm \
    --version v0.0.0-latest \
    --namespace envoy-ai-gateway-system \
    --create-namespace

# 安装 Envoy AI Gateway CRDs
helm upgrade -i aieg-crd oci://docker.io/envoyproxy/ai-gateway-crds-helm --version v0.0.0-latest --namespace envoy-ai-gateway-system

# 等待 AI Gateway Controller 就绪
kubectl wait --timeout=300s -n envoy-ai-gateway-system deployment/ai-gateway-controller --for=condition=Available
```

## 步骤 5：部署演示 LLM

创建一个演示 LLM 作为 Semantic Router 的后端：

```bash
# 部署演示 LLM
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/ai-gateway/aigw-resources/base-model.yaml
```

## 步骤 6：创建 Gateway API 资源

为 AI 网关创建必要的 Gateway API 资源：

```bash
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/ai-gateway/aigw-resources/gwapi-resources.yaml
```

## 测试部署

### 方式 1：端口转发（推荐用于本地测试）

设置端口转发以在本地访问网关：

```bash
# 获取 Envoy 服务名称
export ENVOY_SERVICE=$(kubectl get svc -n envoy-gateway-system \
  --selector=gateway.envoyproxy.io/owning-gateway-namespace=default,gateway.envoyproxy.io/owning-gateway-name=semantic-router \
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

## 故障排除

### 常见问题

**网关无法访问：**

```bash
# 检查网关状态
kubectl get gateway semantic-router -n default

# 检查 Envoy 服务
kubectl get svc -n envoy-gateway-system
```

**AI Gateway controller 未就绪：**

```bash
# 检查 AI gateway controller 日志
kubectl logs -n envoy-ai-gateway-system deployment/ai-gateway-controller

# 检查 controller 状态
kubectl get deployment -n envoy-ai-gateway-system
```

** Semantic Router 无响应：**

```bash
# 检查 Semantic Router  pod 状态
kubectl get pods -n vllm-semantic-router-system

# 检查 Semantic Router 日志
kubectl logs -n vllm-semantic-router-system deployment/semantic-router
```

## 清理

删除整个部署：

```bash
# 删除 Gateway API 资源和演示 LLM
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/ai-gateway/aigw-resources/gwapi-resources.yaml
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/ai-gateway/aigw-resources/base-model.yaml

# 删除 Semantic Router 
helm uninstall semantic-router -n vllm-semantic-router-system

# 删除 AI gateway
helm uninstall aieg -n envoy-ai-gateway-system
helm uninstall aieg-crd -n envoy-ai-gateway-system

# 删除 Envoy gateway
helm uninstall eg -n envoy-gateway-system

# 删除 kind 集群（可选）
kind delete cluster --name semantic-router-cluster
```

## 后续步骤

- 在 AI Gateway 中配置自定义路由规则
- 设置监控和可观测性
- 实施身份验证和授权
- 为生产工作负载扩展 Semantic Router 部署

# 使用 vLLM Production Stack 安装

本教程改编自 [vLLM production stack 教程](https://github.com/vllm-project/production-stack/blob/main/tutorials/24-semantic-router-integration.md)

## 什么是 vLLM Semantic Router？

vLLM Semantic Router 是一个智能的 Mixture of Models (MoM) Router，作为 Envoy 外部处理器运行，将 OpenAI API 兼容请求 Semantic Router 到最合适的后端模型。使用基于 BERT 的分类，它通过将请求（例如 math、code、creative、general）匹配到专业模型来提高质量和成本效率。

- **模型自动选择**：将 math、creative writing、code 和 general 查询路由到最适合的模型。
- **安全与隐私**：PII 检测、Prompt Guard 和敏感 prompt 的安全路由。
- **性能优化**：Semantic Cache 和更好的工具选择以减少延迟和 token。
- **架构**：紧密的 Envoy ExtProc 集成；双 Go 和 Python 实现；生产就绪且可扩展。
- **监控**：Grafana 仪表板、Prometheus 指标和 tracing，实现全面可见性。

了解更多：[vLLM Semantic Router](https://vllm-semantic-router.com/docs/intro)

## 集成有什么好处？

vLLM Production Stack 提供了多种部署方式，可以启动 vLLM 服务器，将流量定向到不同模型，通过 Kubernetes API 执行服务发现和容错，并支持轮询、基于会话、前缀感知、KV 感知和分解预填充路由，原生支持 LMCache。 Semantic Router 添加了一个系统智能层，对每个用户请求进行分类，从池中选择最合适的模型，注入领域特定的系统提示词，执行语义缓存并执行企业级安全检查，如 PII 和越狱检测。

通过结合这两个系统，我们获得了一个统一的推理堆栈。Semantic Router 确保每个请求由最佳可能的模型回答。Production-Stack 路由最大化基础设施和推理效率，并暴露丰富的指标。

---

本教程将指导您：

- 部署一个最小的 vLLM Production Stack
- 部署 vLLM Semantic Router 并将其指向您的 vLLM Router 服务
- 通过 Envoy AI Gateway 测试 endpoint

## 前置条件

- kubectl
- Helm
- Kubernetes 集群（kind、minikube、GKE 等）

---

## 步骤 1：使用您的 Helm values 部署 vLLM Production Stack

使用您的 chart 和位于 `tutorials/assets/values-23-SR.yaml` 的 values 文件。

```bash
helm repo add vllm-production-stack https://vllm-project.github.io/production-stack
helm install vllm-stack vllm-production-stack/vllm-stack -f ./tutorials/assets/values-23-SR.yaml
```

作为参考，以下是示例 value 文件：

```yaml
servingEngineSpec:
  runtimeClassName: ""
  strategy:
    type: Recreate
  modelSpec:
  - name: "qwen3"
    repository: "lmcache/vllm-openai"
    tag: "v0.3.7"
    modelURL: "Qwen/Qwen3-8B"
    pvcStorage: "50Gi"
    vllmConfig:
      # maxModelLen: 131072
      extraArgs: ["--served-model-name", "Qwen/Qwen3-8B", "qwen3"]

    replicaCount: 2

    requestCPU: 8
    requestMemory: "16Gi"
    requestGPU: 1

routerSpec:
  repository: lmcache/lmstack-router
  tag: "latest"
  resources:
    requests:
      cpu: "1"
      memory: "2G"
    limits:
      cpu: "1"
      memory: "2G"
  routingLogic: "roundrobin"
  sessionKey: "x-user-id"
```

识别 chart 创建的路由服务的 ClusterIP 和端口（名称可能有所不同）：

```bash
kubectl get svc vllm-router-service
# 记录路由服务的 ClusterIP 和端口（例如 10.97.254.122:80）
```

---

## 步骤 2：部署 vLLM Semantic Router 并将其指向您的 vLLM Router 服务

按照官方网站的官方指南和**以下更新的配置文件**：[在 Kubernetes 中安装](https://vllm-semantic-router.com/docs/installation/k8s/ai-gateway)。

使用自定义 values 通过 Helm 部署：

```bash
   # 从 GHCR OCI 仓库部署 vLLM Semantic Router 并使用自定义 values
   # （可选）如果使用镜像代理，请添加：--set global.imageRegistry=<your-registry>
   helm install semantic-router oci://ghcr.io/vllm-project/charts/semantic-router \
     --version v0.0.0-latest \
     --namespace vllm-semantic-router-system \
     --create-namespace \
     -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/ai-gateway/semantic-router-values/values.yaml

   kubectl wait --for=condition=Available deployment/semantic-router \
     -n vllm-semantic-router-system --timeout=600s

   # 安装 Envoy Gateway
   helm upgrade -i eg oci://docker.io/envoyproxy/gateway-helm \
     --version v0.0.0-latest \
     --namespace envoy-gateway-system \
     --create-namespace \
     -f https://raw.githubusercontent.com/envoyproxy/ai-gateway/main/manifests/envoy-gateway-values.yaml

   # 安装 Envoy AI Gateway
   helm upgrade -i aieg oci://docker.io/envoyproxy/ai-gateway-helm \
     --version v0.0.0-latest \
     --namespace envoy-ai-gateway-system \
     --create-namespace

   # 安装 Envoy AI Gateway CRDs
   helm upgrade -i aieg-crd oci://docker.io/envoyproxy/ai-gateway-crds-helm \
     --version v0.0.0-latest \
     --namespace envoy-ai-gateway-system

   kubectl wait --timeout=300s -n envoy-ai-gateway-system \
     deployment/ai-gateway-controller --for=condition=Available
```

**注意**：values 文件包含 Semantic Router 的配置。您可以从 [values.yaml](https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/ai-gateway/semantic-router-values/values.yaml) 下载并自定义以匹配您的 vLLM Production Stack 设置。

创建 LLM 演示后端和 AI Gateway 路由：

```bash
   # 应用 LLM 演示后端
   kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/ai-gateway/aigw-resources/base-model.yaml
   # 应用 AI Gateway 路由
   kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/ai-gateway/aigw-resources/gwapi-resources.yaml
```

---

## 步骤 3：测试部署

端口转发到 Envoy 服务并发送测试请求，按照指南：

```bash
  export ENVOY_SERVICE=$(kubectl get svc -n envoy-gateway-system \
    --selector=gateway.envoyproxy.io/owning-gateway-namespace=default,gateway.envoyproxy.io/owning-gateway-name=semantic-router \
    -o jsonpath='{.items[0].metadata.name}')

  kubectl port-forward -n envoy-gateway-system svc/$ENVOY_SERVICE 8080:80
```

发送对话补全请求：

```bash
  curl -i -X POST http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "MoM",
      "messages": [
        {"role": "user", "content": "What is the derivative of f(x) = x^3?"}
      ]
    }'
```

---

## 清理

删除整个部署：

```bash
# 删除 Gateway API 资源和演示 LLM
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/ai-gateway/aigw-resources/gwapi-resources.yaml
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/ai-gateway/aigw-resources/base-model.yaml

# 删除Semantic Router 
helm uninstall semantic-router -n vllm-semantic-router-system

# 删除 AI gateway
helm uninstall aieg -n envoy-ai-gateway-system
helm uninstall aieg-crd -n envoy-ai-gateway-system

# 删除 Envoy gateway
helm uninstall eg -n envoy-gateway-system

# 删除 vLLM Production Stack
helm uninstall vllm-stack

# 删除 kind 集群（可选）
kind delete cluster --name semantic-router-cluster
```

---

## 故障排除

- 如果网关无法访问，请按照指南检查 Gateway 和 Envoy 服务。
- 如果推理池未就绪，请 `kubectl describe` InferencePool 并检查 controller 日志。
- 如果 Semantic Router 无响应，请检查其 pod 状态和日志。
- 如果返回错误代码，请检查 production stack 路由日志。

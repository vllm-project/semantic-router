---
translation:
  source_commit: "9052d81a"
  source_file: "docs/installation/k8s/agentgateway.md"
  outdated: false
---

# 使用 agentgateway 安装

本指南逐步介绍如何在 Kubernetes 上将 vLLM Semantic Router 与 [agentgateway](https://agentgateway.dev/) 集成。agentgateway 充当 OpenAI 兼容流量的 Gateway API 数据平面，而 vLLM Semantic Router 作为 Envoy ExtProc 服务器运行，对每个请求进行分类并修改请求体，然后 agentgateway 再将其转发给 vLLM。

## 架构概览

该部署由以下组件组成：

- **vLLM Semantic Router**：通过 ExtProc 提供 prompt 分类、模型选择、请求修改和响应处理
- **agentgateway**：提供 Kubernetes Gateway API 代理以及 `AgentgatewayBackend`、`HTTPRoute` 和 `AgentgatewayPolicy` 资源
- **演示用 vLLM 兼容后端**：通过 OpenAI 兼容 API 提供基础模型和 LoRA adapter

## 前提条件

开始之前，请确保已安装以下工具：

- [kind](https://kind.sigs.k8s.io/docs/user/quick-start/#installation) - Docker 中的 Kubernetes（可选）
- [kubectl](https://kubernetes.io/docs/tasks/tools/) - Kubernetes CLI
- [Helm](https://helm.sh/docs/intro/install/) - Kubernetes 包管理器

本指南要求使用 agentgateway `v1.3.0-alpha.1` 或更高版本，因为其中使用了 ExtProc 的 `processingOptions` 和 `allowModeOverride` 字段，这些字段是在 `v1.2.1` 之后添加的。

## 步骤 1：创建 Kind 集群（可选）

创建用于测试的本地 Kubernetes 集群：

```bash
kind create cluster --name semantic-router-agentgateway

# 验证集群已就绪
kubectl wait --for=condition=Ready nodes --all --timeout=300s
```

## 步骤 2：安装 agentgateway

安装 Kubernetes Gateway API CRD 和 agentgateway 控制平面：

```bash
export AGENTGATEWAY_VERSION=v1.3.0-alpha.1

kubectl apply --server-side --force-conflicts \
  -f https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.5.0/standard-install.yaml

helm upgrade -i agentgateway-crds oci://cr.agentgateway.dev/charts/agentgateway-crds \
  --create-namespace \
  --namespace agentgateway-system \
  --version "${AGENTGATEWAY_VERSION}" \
  --set controller.image.pullPolicy=Always

helm upgrade -i agentgateway oci://cr.agentgateway.dev/charts/agentgateway \
  --namespace agentgateway-system \
  --version "${AGENTGATEWAY_VERSION}" \
  --set controller.image.pullPolicy=Always \
  --set controller.extraEnv.KGW_ENABLE_GATEWAY_API_EXPERIMENTAL_FEATURES=true \
  --wait

kubectl get pods -n agentgateway-system
```

## 步骤 3：创建 agentgateway 代理

创建一个使用 agentgateway GatewayClass 的 Gateway：

```bash
kubectl apply -f- <<'EOF'
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: agentgateway-proxy
  namespace: agentgateway-system
spec:
  gatewayClassName: agentgateway
  listeners:
  - protocol: HTTP
    port: 80
    name: http
    allowedRoutes:
      namespaces:
        from: All
EOF

kubectl wait --for=condition=Available deployment/agentgateway-proxy \
  -n agentgateway-system \
  --timeout=300s
```

## 步骤 4：部署演示 LLM

部署一个轻量级 OpenAI 兼容模拟器，提供 `base-model` 以及 Semantic Router 演示配置所选择的 LoRA adapter 名称：

```bash
kubectl apply -f- <<'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-llama3-8b-instruct
  namespace: default
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-llama3-8b-instruct
  template:
    metadata:
      labels:
        app: vllm-llama3-8b-instruct
    spec:
      containers:
      - name: vllm-sim
        image: ghcr.io/llm-d/llm-d-inference-sim:v0.5.0
        imagePullPolicy: IfNotPresent
        args:
        - --model
        - base-model
        - --port
        - "8000"
        - --max-loras
        - "6"
        - --lora-modules
        - '{"name": "math-expert"}'
        - '{"name": "science-expert"}'
        - '{"name": "social-expert"}'
        - '{"name": "humanities-expert"}'
        - '{"name": "law-expert"}'
        - '{"name": "general-expert"}'
        ports:
        - containerPort: 8000
          name: http
          protocol: TCP
        readinessProbe:
          httpGet:
            path: /health
            port: http
          periodSeconds: 5
          timeoutSeconds: 5
          failureThreshold: 3
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-llama3-8b-instruct
  namespace: default
  labels:
    app: vllm-llama3-8b-instruct
spec:
  type: ClusterIP
  ports:
  - port: 8000
    targetPort: 8000
    protocol: TCP
  selector:
    app: vllm-llama3-8b-instruct
EOF

kubectl wait --for=condition=Available deployment/vllm-llama3-8b-instruct \
  -n default \
  --timeout=300s
```

## 步骤 5：部署 vLLM Semantic Router

在 `agentgateway-system` namespace 中安装 Semantic Router，以便 agentgateway ExtProc policy 可以直接引用 `semantic-router` service：

```bash
helm install semantic-router oci://ghcr.io/vllm-project/charts/semantic-router \
  --version v0.0.0-latest \
  --namespace agentgateway-system \
  -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/agentgateway/semantic-router-values/values.yaml \
  --set config.global.router.streamed_body.enabled=true \
  --set config.global.router.streamed_body.max_bytes=10485760 \
  --set config.global.router.streamed_body.timeout_sec=30

kubectl wait --for=condition=Available deployment/semantic-router \
  -n agentgateway-system \
  --timeout=600s
```

该 values 文件将 Semantic Router 配置为向 `vllm-llama3-8b-instruct.default.svc.cluster.local:8000` 发送流量，并选择 `math-expert`、`science-expert` 和 `general-expert` 等 adapter 名称。

## 步骤 6：创建 agentgateway 路由资源

为 vLLM 兼容后端创建 `AgentgatewayBackend`，并将 OpenAI 兼容请求路由到该后端：

```bash
kubectl apply -f- <<'EOF'
apiVersion: agentgateway.dev/v1alpha1
kind: AgentgatewayBackend
metadata:
  name: semantic-router-vllm
  namespace: agentgateway-system
spec:
  ai:
    provider:
      openai: {}
      host: vllm-llama3-8b-instruct.default.svc.cluster.local
      port: 8000
---
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: semantic-router-vllm
  namespace: agentgateway-system
spec:
  parentRefs:
  - name: agentgateway-proxy
    namespace: agentgateway-system
  rules:
  - backendRefs:
    - name: semantic-router-vllm
      namespace: agentgateway-system
      group: agentgateway.dev
      kind: AgentgatewayBackend
EOF
```

这里有意省略了 `openai.model` 字段，使 agentgateway 可以在 Semantic Router 选择目标模型或 LoRA adapter 后，使用请求体中的模型名称。

## 步骤 7：将 Semantic Router 作为 ExtProc 挂载

创建一个 `AgentgatewayPolicy`，将请求和响应处理阶段发送到 Semantic Router ExtProc service：

```bash
kubectl apply -f- <<'EOF'
apiVersion: agentgateway.dev/v1alpha1
kind: AgentgatewayPolicy
metadata:
  name: semantic-router-extproc
  namespace: agentgateway-system
spec:
  targetRefs:
  - group: gateway.networking.k8s.io
    kind: Gateway
    name: agentgateway-proxy
  traffic:
    extProc:
      backendRef:
        name: semantic-router
        namespace: agentgateway-system
        port: 50051
      processingOptions:
        requestHeaderMode: Send
        requestBodyMode: FullDuplexStreamed
        responseHeaderMode: Send
        responseBodyMode: Buffered
        requestTrailerMode: Send
        responseTrailerMode: Send
        allowModeOverride: true
EOF
```

项目自带的 agentgateway 示例显式启用全双工流式请求体。这是该示例特有的选择；其他代理默认配置和示例可以继续使用缓冲请求体。上面的 Semantic Router Helm 命令显式启用了 `global.router.streamed_body`，使 router 能够累积请求分块，并在流结束时处理完整请求体。

agentgateway 不支持 `Streamed` 模式；其流式处理选项为 `FullDuplexStreamed`。可部署的 policy 位于 `deploy/kubernetes/agentgateway/extproc-policy.yaml`，与之匹配的 router 配置通过步骤 5 中的 Helm 命令传入。有关协议行为和验证清单，请参阅[流式 ExtProc 与即时响应](./streamed-extproc.md)。

## 测试部署

启动到 agentgateway 代理的端口转发：

```bash
kubectl port-forward -n agentgateway-system svc/agentgateway-proxy 8080:80
```

在另一个终端中，发送一个包含 `"model": "auto"` 的 OpenAI 兼容请求：

```bash
curl -i -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [
      {"role": "user", "content": "What is the derivative of f(x) = x^3?"}
    ],
    "max_tokens": 64,
    "temperature": 0
  }'
```

Semantic Router 应对数学 prompt 进行分类，选择配置的数学路由，并在 agentgateway 将请求转发给 vLLM 兼容后端之前修改请求中的模型。使用 `-i` 可检查 Semantic Router 的响应 header，例如所选模型的元数据。

## 故障排除

**agentgateway 代理未就绪：**

```bash
kubectl get gateway agentgateway-proxy -n agentgateway-system
kubectl get deployment agentgateway-proxy -n agentgateway-system
kubectl logs -n agentgateway-system deployment/agentgateway
```

**HTTPRoute 或 agentgateway 后端未被接受：**

```bash
kubectl describe httproute semantic-router-vllm -n agentgateway-system
kubectl describe agentgatewaybackend semantic-router-vllm -n agentgateway-system
```

**Semantic Router 未响应 ExtProc：**

```bash
kubectl get pods -n agentgateway-system
kubectl get svc semantic-router -n agentgateway-system
kubectl logs -n agentgateway-system deployment/semantic-router
kubectl describe agentgatewaypolicy semantic-router-extproc -n agentgateway-system
```

**演示 LLM 未响应：**

```bash
kubectl get pods -n default -l app=vllm-llama3-8b-instruct
kubectl logs -n default deployment/vllm-llama3-8b-instruct
```

## 清理

要删除整个部署，请运行：

```bash
kubectl delete agentgatewaypolicy semantic-router-extproc -n agentgateway-system
kubectl delete httproute semantic-router-vllm -n agentgateway-system
kubectl delete agentgatewaybackend semantic-router-vllm -n agentgateway-system
kubectl delete gateway agentgateway-proxy -n agentgateway-system
kubectl delete deployment vllm-llama3-8b-instruct -n default
kubectl delete service vllm-llama3-8b-instruct -n default

helm uninstall semantic-router -n agentgateway-system
helm uninstall agentgateway -n agentgateway-system
helm uninstall agentgateway-crds -n agentgateway-system

kind delete cluster --name semantic-router-agentgateway
```

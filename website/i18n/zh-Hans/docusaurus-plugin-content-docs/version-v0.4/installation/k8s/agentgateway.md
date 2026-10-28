---
title: 使用 agentgateway 部署
description: 将 Semantic Router 作为 ExtProc 服务接入 agentgateway 的 Kubernetes 数据面。
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/installation/k8s/agentgateway.md"
  outdated: false
---

# 使用 agentgateway 部署

当 [agentgateway](https://agentgateway.dev/) 负责 Kubernetes Gateway API 数据面时，使用此拓扑。Semantic Router 作为 Envoy ExtProc 服务运行：它评估所选配方，将选定的模型写入请求，再由 agentgateway 转发到 OpenAI 兼容后端。

## 职责划分

部署包含：

- **Semantic Router** 负责语义策略、模型选择，以及按配方处理请求或响应。
- **agentgateway** 负责 Gateway、`HTTPRoute`、后端和 ExtProc 策略。
- **模型服务器** 负责推理容量。下面的模拟器仅用于验证集成，不能用于生产推理。

## 前置条件

需要：

- Kubernetes `1.31`–`1.36`；演示可用 [kind](https://kind.sigs.k8s.io/docs/user/quick-start/#installation)；
- Gateway API `1.4`–`1.6` CRD（本指南安装 `1.6.0`）；
- [kubectl](https://kubernetes.io/docs/tasks/tools/) 与集群支持的版本偏差范围内；以及
- [Helm](https://helm.sh/docs/intro/install/) `3.12` 或更高。

本指南固定 agentgateway 1.4 发布线，包含 ExtProc `processingOptions` 和 `allowModeOverride`。升级集成任一侧之前，请先阅读上游 [ExtProc 参考](https://agentgateway.dev/docs/kubernetes/latest/traffic-management/extproc/)。

## 步骤 1：创建 Kind 集群（可选）

创建用于测试的本地 Kubernetes 集群：

```bash
kind create cluster --name semantic-router-agentgateway

# Verify cluster is ready
kubectl wait --for=condition=Ready nodes --all --timeout=300s
```

## 步骤 2：安装 agentgateway

安装 Kubernetes Gateway API CRD 和 agentgateway 控制面：

```bash
export AGENTGATEWAY_VERSION=v1.4.1

kubectl apply --server-side --force-conflicts \
  -f https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.6.0/standard-install.yaml

helm upgrade -i agentgateway-crds oci://cr.agentgateway.dev/charts/agentgateway-crds \
  --create-namespace \
  --namespace agentgateway-system \
  --version "${AGENTGATEWAY_VERSION}"

helm upgrade -i agentgateway oci://cr.agentgateway.dev/charts/agentgateway \
  --namespace agentgateway-system \
  --version "${AGENTGATEWAY_VERSION}" \
  --wait

kubectl get pods -n agentgateway-system
```

## 步骤 3：创建 agentgateway 代理

创建使用 agentgateway GatewayClass 的 Gateway：

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

部署轻量级 OpenAI 兼容模拟器，提供 `base-model` 以及 Semantic Router 演示配置所选的 LoRA adapter 名称：

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
        image: ghcr.io/llm-d/llm-d-inference-sim:v0.6.1
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

将 Semantic Router 安装到 `agentgateway-system` 命名空间，以便 agentgateway ExtProc 策略可以直接引用 `semantic-router` 服务：

```bash
helm install semantic-router oci://ghcr.io/vllm-project/charts/semantic-router \
  --version 0.0.0-latest \
  --namespace agentgateway-system \
  -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/agentgateway/semantic-router-values/values.yaml \
  --set config.global.router.streamed_body.enabled=true \
  --set config.global.router.streamed_body.max_bytes=10485760 \
  --set config.global.router.streamed_body.timeout_sec=30

kubectl wait --for=condition=Available deployment/semantic-router \
  -n agentgateway-system \
  --timeout=600s
```

values 文件将 Semantic Router 配置为把流量发到 `vllm-llama3-8b-instruct.default.svc.cluster.local:8000`，并选择 `math-expert`、`science-expert` 和 `general-expert` 等 adapter 名称。

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

有意省略 `openai.model` 字段，以便 agentgateway 使用请求体中的模型名。该名称由 Semantic Router 在选定目标模型或 LoRA adapter 后写入。

## 步骤 7：将 Semantic Router 接入为 ExtProc

创建 `AgentgatewayPolicy`，将请求和响应处理阶段发送到 Semantic Router ExtProc 服务：

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

附带的 agentgateway 示例显式启用全双工流式请求体。这是该示例的选择；其他代理默认值和示例可能仍使用缓冲请求体。上面的 Semantic Router Helm 命令显式启用 `global.router.streamed_body`，让 Router 累积请求分片，并在流结束时处理完整请求体。

agentgateway 不支持 `Streamed` 模式；`FullDuplexStreamed` 是其流式选项。可部署的策略在 `deploy/kubernetes/agentgateway/extproc-policy.yaml`，匹配的 Router 配置通过步骤 5 的 Helm 命令传入。协议行为和验证清单见 [Streamed ExtProc 与立即响应](./streamed-extproc)。

## 测试部署

对 agentgateway 代理启动 port-forward：

```bash
kubectl port-forward -n agentgateway-system svc/agentgateway-proxy 8080:80
```

在另一个终端，使用稳定的自动路由别名发送 OpenAI 兼容请求：

```bash
curl -i -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vllm-sr/auto",
    "messages": [
      {"role": "user", "content": "What is the derivative of f(x) = x^3?"}
    ],
    "max_tokens": 64,
    "temperature": 0
  }'
```

Semantic Router 应分类该数学提示词，选择配置的数学路由，并在 agentgateway 将请求转发到 vLLM 兼容后端之前改写请求模型。使用 `-i` 检查 Semantic Router 响应头，例如所选模型元数据。

## 故障排查

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

**演示 LLM 无响应：**

```bash
kubectl get pods -n default -l app=vllm-llama3-8b-instruct
kubectl logs -n default deployment/vllm-llama3-8b-instruct
```

## 清理

要移除整个部署：

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

---
sidebar_position: 3
sidebar_label: 使用 Operator 安装
translation:
  source_commit: "5ff00978"
  source_file: "docs/installation/k8s/operator.md"
  outdated: false
---

# 使用 Operator 安装

Semantic Router Operator 提供了一种 Kubernetes 原生的方式，通过自定义资源定义（CRD）来部署与管理 vLLM Semantic Router 实例。它可以在 Kubernetes 与 OpenShift 平台上简化部署、配置与生命周期管理。

## 特性

- **声明式部署**：使用 Kubernetes CRD 定义语义路由实例
- **自动配置**：生成并管理用于语义路由配置的 ConfigMap
- **持久化存储**：管理用于 ML 模型存储的 PVC，并自动处理生命周期
- **平台探测**：自动识别 OpenShift 或标准 Kubernetes，并做相应配置
- **内置可观测性**：默认支持指标、链路追踪与监控
- **生产能力**：HPA、Ingress、Service Mesh 集成、Pod Disruption Budget
- **默认安全**：移除全部 capability，禁止特权提升

## 前置条件

- Kubernetes 1.24+ 或 OpenShift 4.12+
- 已配置好的 `kubectl` 或 `oc` 命令行
- 集群管理员权限（用于安装 CRD）

## 安装

### 选项 1：使用 Kustomize（标准 Kubernetes）

```bash
# Clone the repository
git clone https://github.com/vllm-project/semantic-router
cd semantic-router/deploy/operator

# Install CRDs
make install

# Deploy the operator
make deploy IMG=ghcr.io/vllm-project/semantic-router/operator:latest
```

验证 operator 正在运行：

```bash
kubectl get pods -n semantic-router-operator-system
```

### 选项 2：使用 OLM（OpenShift）

适用于通过 Operator Lifecycle Manager 部署到 OpenShift 的场景：

```bash
cd semantic-router/deploy/operator

# Build and push to your registry (Quay, internal registry, etc.)
podman login quay.io
make podman-build IMG=quay.io/<your-org>/semantic-router-operator:latest
make podman-push IMG=quay.io/<your-org>/semantic-router-operator:latest

# Deploy using OLM
make openshift-deploy
```

## 部署你的第一个 Router

### 使用示例配置快速开始

根据你的基础设施选择一个预配置示例：

```bash
# Simple standalone deployment with KServe backend
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/operator/config/samples/vllm.ai_v1alpha1_semanticrouter_simple.yaml

# Full-featured OpenShift deployment with Routes
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/operator/config/samples/vllm.ai_v1alpha1_semanticrouter_openshift.yaml

# Gateway integration mode (Istio/Envoy Gateway)
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/operator/config/samples/vllm.ai_v1alpha1_semanticrouter_gateway.yaml

# Llama Stack backend discovery
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/operator/config/samples/vllm.ai_v1alpha1_semanticrouter_llamastack.yaml

# Redis cache backend for production caching
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/operator/config/samples/vllm.ai_v1alpha1_semanticrouter_redis_cache.yaml

# Milvus cache backend for large-scale deployments
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/operator/config/samples/vllm.ai_v1alpha1_semanticrouter_milvus_cache.yaml

# Hybrid cache backend for optimal performance
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/operator/config/samples/vllm.ai_v1alpha1_semanticrouter_hybrid_cache.yaml

# mmBERT 2D Matryoshka embeddings with layer early exit
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/operator/config/samples/vllm.ai_v1alpha1_semanticrouter_mmbert.yaml

# Complexity-aware routing for intelligent model selection
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/operator/config/samples/vllm.ai_v1alpha1_semanticrouter_complexity.yaml
```

### 自定义配置

创建一个 `my-router.yaml` 文件：

```yaml
apiVersion: vllm.ai/v1alpha1
kind: SemanticRouter
metadata:
  name: my-router
  namespace: default
spec:
  replicas: 2

  image:
    repository: ghcr.io/vllm-project/semantic-router/extproc
    tag: latest

  # Configure vLLM backend endpoints
  vllmEndpoints:
    # KServe InferenceService (RHOAI 3.x)
    - name: llama3-8b-endpoint
      model: llama3-8b
      reasoningFamily: qwen3
      loras:
        - name: computer-science-expert
          description: Adapter for advanced computer science prompts
      backend:
        type: kserve
        inferenceServiceName: llama-3-8b
      weight: 1

  resources:
    limits:
      memory: "7Gi"
      cpu: "2"
    requests:
      memory: "3Gi"
      cpu: "1"

  persistence:
    enabled: true
    size: 10Gi
    storageClassName: "standard"

  config:
    providers:
      defaults:
        default_model: llama3-8b
        default_reasoning_effort: medium
        reasoning_families:
          qwen3:
            type: chat_template_kwargs
            parameter: enable_thinking
      models:
        - name: llama3-8b
          provider_model_id: llama3-8b
          backend_refs:
            - name: llama3-8b-endpoint
              endpoint: llama-3-8b-predictor.default.svc.cluster.local:80
              protocol: http

    routing:
      modelCards:
        - name: llama3-8b
          modality: text
          capabilities: ["chat", "reasoning"]
      decisions:
        - name: default-route
          description: Catch-all route
          priority: 100
          rules:
            operator: AND
            conditions: []
          modelRefs:
            - model: llama3-8b
              use_reasoning: false

    global:
      stores:
        semantic_cache:
          enabled: true
          backend_type: memory
          max_entries: 1000
          ttl_seconds: 3600
      integrations:
        tools:
          enabled: true
          top_k: 3
          similarity_threshold: 0.2
      model_catalog:
        system:
          prompt_guard: models/mmbert32k-jailbreak-detector-merged
        modules:
          prompt_guard:
            enabled: true
            model_ref: prompt_guard
            threshold: 0.7

  toolsDb:
    - tool:
        type: "function"
        function:
          name: "get_weather"
          description: "Get weather information for a location"
          parameters:
            type: "object"
            properties:
              location:
                type: "string"
                description: "City and state, e.g. San Francisco, CA"
            required: ["location"]
      description: "Weather information tool"
      category: "weather"
      tags: ["weather", "temperature"]
```

应用配置：

```bash
kubectl apply -f my-router.yaml
```

`spec.config` 应使用与本地 `config.yaml` 相同的规范化 `providers/routing/global` 布局。`spec.vllmEndpoints` 仍是 Kubernetes 适配层，用于发现后端与 served-model alias；operator 在渲染 runtime config 时，会将其转换为规范化的 `providers.models[].backend_refs[]` 与 `routing.modelCards` 条目（包含可选的 `loras`）。

## 高级特性

### Embedding 模型配置

operator 支持三种高性能 embedding 模型，用于语义理解与缓存。你可以根据场景配置这些模型以优化效果。

#### 可用的 embedding 模型

1. **Qwen3-Embedding**（1024 维，32K 上下文）
   - 适合：高质量语义理解与长上下文
   - 场景：复杂查询、研究文档、细致分析

2. **EmbeddingGemma**（768 维，8K 上下文）
   - 适合：更快性能与较好精度
   - 场景：实时应用、高吞吐

3. **mmBERT 2D Matryoshka**（64-768 维，多语言）
   - 适合：可通过 layer early exit 自适应权衡速度与质量
   - 场景：多语言部署、需要灵活的质量/速度权衡

#### 示例：mmBERT + Layer Early Exit

```yaml
spec:
  config:
    global:
      model_catalog:
        embeddings:
          semantic:
            mmbert_model_path: "models/mom-embedding-ultra"
            use_cpu: true
            embedding_config:
              model_type: "mmbert"
              # Layer early exit: balance speed vs accuracy
              # Layer 3: ~7x speedup (fast, good for high-volume queries)
              # Layer 6: ~3.6x speedup (balanced - recommended)
              # Layer 11: ~2x speedup (higher accuracy)
              # Layer 22: full model (maximum accuracy)
              target_layer: 6
              # Dimension reduction for faster similarity search
              # Options: 64, 128, 256, 512, 768
              target_dimension: 256
              preload_embeddings: true
              enable_soft_matching: true
              top_k: 1
              min_score_threshold: "0.5"
      stores:
        semantic_cache:
          enabled: true
          backend_type: "memory"
          embedding_model: "mmbert"
          similarity_threshold: "0.85"
          max_entries: 5000
          ttl_seconds: 7200
```

完整示例可参考 [mmbert sample configuration](https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/operator/config/samples/vllm.ai_v1alpha1_semanticrouter_mmbert.yaml)。

#### 示例：Qwen3 + Redis Cache

```yaml
spec:
  config:
    global:
      model_catalog:
        embeddings:
          semantic:
            qwen3_model_path: "models/qwen3-embedding"
            use_cpu: true
      stores:
        semantic_cache:
          enabled: true
          backend_type: "redis"
          embedding_model: "qwen3"
          redis:
            connection:
              host: redis.cache-backends.svc.cluster.local
              port: 6379
        index:
          vector_field:
            dimension: 1024  # Qwen3 dimension
```

完整示例可参考 [redis cache sample configuration](https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/operator/config/samples/vllm.ai_v1alpha1_semanticrouter_redis_cache.yaml)。

### 基于复杂度的路由（Complexity-Aware Routing）

根据复杂度分类将请求路由到不同模型：简单请求走更快的模型，复杂请求走更强的模型。

#### 示例配置

```yaml
spec:
  # Configure multiple backends with different capabilities
  vllmEndpoints:
    - name: llama-8b-fast
      model: llama3-8b
      reasoningFamily: qwen3
      backend:
        type: kserve
        inferenceServiceName: llama-3-8b
      weight: 2  # Prefer for simple queries

    - name: llama-70b-reasoning
      model: llama3-70b
      reasoningFamily: deepseek
      backend:
        type: kserve
        inferenceServiceName: llama-3-70b
      weight: 1  # Use for complex queries

  config:
    # Define complexity rules
    complexity_rules:
      # Rule 1: Code complexity
      - name: "code-complexity"
        description: "Classify coding tasks by complexity"
        threshold: "0.3"  # Lower threshold works better for embedding-based similarity

        # Examples of complex coding tasks
        hard:
          candidates:
            - "Implement a distributed lock manager with leader election"
            - "Design a database migration system with rollback support"
            - "Create a compiler optimization pass for loop unrolling"

        # Examples of simple coding tasks
        easy:
          candidates:
            - "Write a function to reverse a string"
            - "Create a class to represent a rectangle"
            - "Implement a simple counter with increment/decrement"

      # Rule 2: Reasoning complexity
      - name: "reasoning-complexity"
        description: "Classify reasoning and problem-solving tasks"
        threshold: "0.3"  # Lower threshold works better for embedding-based similarity

        hard:
          candidates:
            - "Analyze the geopolitical implications of renewable energy adoption"
            - "Evaluate the ethical considerations of AI in healthcare"
            - "Design a multi-stage marketing strategy for a new product launch"

        easy:
          candidates:
            - "What is the capital of France?"
            - "How many days are in a week?"
            - "Name three common pets"

      # Rule 3: Domain-specific complexity with conditional application
      - name: "medical-complexity"
        description: "Classify medical queries (only for medical domain)"
        threshold: "0.3"  # Lower threshold works better for embedding-based similarity

        hard:
          candidates:
            - "Differential diagnosis for chest pain with dyspnea"
            - "Treatment protocol for multi-drug resistant tuberculosis"

        easy:
          candidates:
            - "What is the normal body temperature?"
            - "What are common symptoms of a cold?"

        # Only apply this rule if domain signal indicates medical domain
        composer:
          operator: "AND"
          conditions:
            - type: "domain"
              name: "medical"
```

**工作原理：**

1. 输入查询会与 `hard` 与 `easy` 的候选示例做相似度比较
2. 相似度分数用于判定复杂度
3. 输出 signals：`{rule-name}:hard`、`{rule-name}:easy` 或 `{rule-name}:medium`
4. Router 根据 signals 选择后端模型
5. `composer` 支持根据其他 signals 做条件性规则应用

完整示例可参考 [complexity routing sample configuration](https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/operator/config/samples/vllm.ai_v1alpha1_semanticrouter_complexity.yaml)。

## 验证部署

```bash
# Check the SemanticRouter resource
kubectl get semanticrouter my-router

# Check created resources
kubectl get deployment,service,configmap -l app.kubernetes.io/instance=my-router

# View status
kubectl describe semanticrouter my-router

# View logs
kubectl logs -f deployment/my-router
```

期望输出示例：

```
NAME                        PHASE     REPLICAS   READY   AGE
semanticrouter.vllm.ai/my-router   Running   2          2       5m
```

## 后端发现类型（Backend Discovery Types）

operator 支持三种后端发现方式，用于连接 semantic router 与 vLLM 模型服务。选择与你的基础设施匹配的类型即可。

### KServe InferenceService 发现

适用于 RHOAI 3.x 或独立 KServe 部署。operator 会自动发现 KServe 创建的 predictor service。

```yaml
spec:
  vllmEndpoints:
    - name: llama3-8b-endpoint
      model: llama3-8b
      reasoningFamily: qwen3
      backend:
        type: kserve
        inferenceServiceName: llama-3-8b  # InferenceService in same namespace
      weight: 1
```

**适用场景：**

- 运行在 Red Hat OpenShift AI（RHOAI）3.x
- 使用 KServe 做模型服务
- 需要自动服务发现

**工作原理：**

- 发现 predictor service：`{inferenceServiceName}-predictor`
- 使用 8443 端口（KServe 默认 HTTPS 端口）
- 与 `SemanticRouter` 在同一命名空间内工作

### Llama Stack Service 发现

通过 Kubernetes label selector 发现 Llama Stack 部署。

```yaml
spec:
  vllmEndpoints:
    - name: llama-405b-endpoint
      model: llama-3.3-70b-instruct
      reasoningFamily: gpt
      backend:
        type: llamastack
        discoveryLabels:
          app: llama-stack
          model: llama-3.3-70b
      weight: 1
```

**适用场景：**

- 使用 Meta 的 Llama Stack 做模型服务
- 同时有多个 Llama Stack 服务、不同模型
- 需要基于 label 做服务发现

**工作原理：**

- 列出匹配 label selector 的 services
- 若匹配多个，默认使用第一个
- 从 service 定义中提取端口

### 直连 Kubernetes Service

可直连任意 Kubernetes service（vLLM、TGI 等）。

```yaml
spec:
  vllmEndpoints:
    - name: custom-vllm-endpoint
      model: deepseek-r1-distill-qwen-7b
      reasoningFamily: deepseek
      backend:
        type: service
        service:
          name: vllm-deepseek
          namespace: vllm-serving  # Can reference service in another namespace
          port: 8000
      weight: 1
```

**适用场景：**

- 直接部署的 vLLM 服务
- 自定义模型服务器（OpenAI 兼容 API）
- 跨命名空间引用 service
- 希望完全控制 service endpoint

**工作原理：**

- 直接连接指定 service
- 不做 discovery，完全按显式配置
- 支持跨命名空间引用

### 多后端（Multiple Backends）

你可以配置多个后端，并用权重做负载均衡：

```yaml
spec:
  vllmEndpoints:
    # KServe backend
    - name: llama3-8b
      model: llama3-8b
      reasoningFamily: qwen3
      backend:
        type: kserve
        inferenceServiceName: llama-3-8b
      weight: 2  # Higher weight = more traffic

    # Direct service backend
    - name: qwen-7b
      model: qwen2.5-7b
      reasoningFamily: qwen3
      backend:
        type: service
        service:
          name: vllm-qwen
          port: 8000
      weight: 1
```

## 部署模式（Deployment Modes）

operator 支持两种部署模式，对应不同架构。

### Standalone 模式（默认）

部署 semantic router，并带一个 **Envoy sidecar 容器**作为入口网关。

**架构：**

```text
Client → Service (8080) → Envoy Sidecar → ExtProc gRPC → Semantic Router → vLLM
```

**适用场景：**

- 没有现成 service mesh 的简单部署
- 测试与开发
- 自包含部署、依赖最少

**配置：**

```yaml
spec:
  # No gateway configuration - defaults to standalone mode
  service:
    type: ClusterIP
    api:
      port: 8080  # Client traffic enters here
      targetPort: 8080  # Envoy ingress port
    grpc:
      port: 50051  # ExtProc communication
      targetPort: 50051
```

**operator 行为：**

- 在 pod spec 中部署两个容器：semantic router + Envoy sidecar
- Envoy 负责 ingress，并通过 ExtProc gRPC 转发到 semantic router
- status 显示 `gatewayMode: "standalone"`

### Gateway Integration 模式

复用 **已有 Gateway**（Istio、Envoy Gateway 等），并要求你自行管理匹配的 HTTPRoute。

当前状态：controller 能解析所引用的 Gateway 并切换到 gateway mode，但自动创建 HTTPRoute 仍是占位实现。

**架构：**

```text
Client → Gateway (Istio/Envoy) → user-managed HTTPRoute → Service (8080) → Semantic Router API → vLLM
```

**适用场景：**

- 已有 Istio 或 Envoy Gateway 部署
- 统一的 ingress 管理
- 共享网关的多租户场景
- 高级流量治理（熔断、重试、限流等）

**配置：**

```yaml
spec:
  gateway:
    existingRef:
      name: istio-ingressgateway  # Or your Envoy Gateway name
      namespace: istio-system

  # Service only needs API port in gateway mode
  service:
    type: ClusterIP
    api:
      port: 8080
      targetPort: 8080
```

**operator 行为：**

1. 解析 referenced Gateway 并进入 gateway integration 模式
2. 目前不创建 HTTPRoute，你需要自行 apply 与管理该资源
3. pod spec 中不再部署 Envoy sidecar
4. 设置 `status.gatewayMode: "gateway-integration"`
5. semantic router 以纯 API 模式运行（不启用 ExtProc）

**示例：** 参考 [`vllm.ai_v1alpha1_semanticrouter_gateway.yaml`](https://github.com/vllm-project/semantic-router/blob/main/deploy/operator/config/samples/vllm.ai_v1alpha1_semanticrouter_gateway.yaml)。

该示例仅为 gateway mode 配置 `SemanticRouter` 资源，并不会替你安装 Gateway 或 HTTPRoute。

## OpenShift Routes

在 OpenShift 上，operator 可创建 Route 用于外部访问，并支持 TLS 终止。

### 基础 OpenShift Route

```yaml
spec:
  openshift:
    routes:
      enabled: true
      hostname: semantic-router.apps.openshift.example.com  # Optional - auto-generated if omitted
      tls:
        termination: edge  # TLS terminates at Route, plain HTTP to backend
        insecureEdgeTerminationPolicy: Redirect  # Redirect HTTP to HTTPS
```

### TLS 终止方式

- **edge**（推荐）：TLS 终止在 Route，后端走明文 HTTP
- **passthrough**：TLS 透传到后端（要求后端支持 TLS）
- **reencrypt**：TLS 终止在 Route，并对后端重新加密

### 何时使用 OpenShift Routes

- 运行在 OpenShift 4.x
- 希望无需配置 Ingress 就提供外部访问
- 希望自动生成 hostname
- 需要 OpenShift 原生的 TLS 管理能力

### 状态信息

创建 Route 后可查看状态：

```bash
kubectl get semanticrouter my-router -o jsonpath='{.status.openshiftFeatures}'
```

输出示例：

```json
{
  "routesEnabled": true,
  "routeHostname": "semantic-router-default.apps.openshift.example.com"
}
```

**示例：** 参考 [`vllm.ai_v1alpha1_semanticrouter_route.yaml`](https://github.com/vllm-project/semantic-router/blob/main/deploy/operator/config/samples/vllm.ai_v1alpha1_semanticrouter_route.yaml)。

## 如何选择配置

使用下面的决策树选择合适的配置：

```
┌─ Need to run on OpenShift?
│  ├─ YES → Use openshift sample (Routes + KServe/service backends)
│  └─ NO ↓
│
├─ Have existing Gateway (Istio/Envoy)?
│  ├─ YES → Use gateway sample (Gateway integration mode)
│  └─ NO ↓
│
├─ Using Meta Llama Stack?
│  ├─ YES → Use llamastack sample
│  └─ NO ↓
│
└─ Simple deployment → Use simple sample (standalone mode)
```

**后端选择：**

```
┌─ Running RHOAI 3.x or KServe?
│  ├─ YES → Use KServe backend type
│  └─ NO ↓
│
├─ Using Meta Llama Stack?
│  ├─ YES → Use llamastack backend type
│  └─ NO ↓
│
└─ Have direct vLLM service? → Use service backend type
```

## 架构

operator 会为每个 `SemanticRouter` 管理一整套资源：

```
┌─────────────────────────────────────────────────────┐
│              SemanticRouter CR                       │
│  apiVersion: vllm.ai/v1alpha1                       │
│  kind: SemanticRouter                               │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  Operator Controller │
        │  - Watches CR        │
        │  - Reconciles state  │
        │  - Platform detection│
        └─────────┬────────────┘
                  │
     ┌────────────┼────────────┬──────────────┐
     ▼            ▼            ▼              ▼
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│Deployment│  │ Service │  │ConfigMap│  │   PVC   │
│         │  │ - gRPC  │  │ - config│  │ - models│
│         │  │ - API   │  │ - tools │  │         │
│         │  │ - metrics│  │         │  │         │
└─────────┘  └─────────┘  └─────────┘  └─────────┘
```

**管理的资源：**

- **Deployment**：运行 semantic router pods，可配置副本数
- **Service**：暴露 gRPC（50051）、HTTP API（8080）与 metrics（9190）
- **ConfigMap**：包含 semantic router 配置与 tools database
- **ServiceAccount**：RBAC（可选，仅当指定时创建）
- **PersistentVolumeClaim**：ML 模型存储（可选，仅当启用 persistence 时创建）
- **HorizontalPodAutoscaler**：自动伸缩（可选，仅当启用 autoscaling 时创建）
- **Ingress**：外部访问（可选，仅当启用 ingress 时创建）

## 平台探测与安全

operator 会自动识别平台，并设置适配的安全上下文。

### OpenShift 平台

当运行在 OpenShift 时，operator 会：

- **探测**：检查是否存在 `route.openshift.io` API 资源
- **安全上下文**：不会设置 `runAsUser`、`runAsGroup`、`fsGroup`
- **原因**：让 OpenShift SCC 从 namespace 允许的 UID/GID 范围里分配
- **兼容性**：`restricted` SCC（默认）与自定义 SCC

### 标准 Kubernetes

当运行在标准 Kubernetes 时，operator 会：

- **安全上下文**：设置 `runAsUser: 1000`、`fsGroup: 1000`、`runAsNonRoot: true`
- **原因**：提供安全的默认值，满足常见的 Pod 安全标准

### 两个平台通用

无论平台如何：

- 移除全部 capability（`drop: [ALL]`）
- 禁止特权提升（`allowPrivilegeEscalation: false`）
- 除默认权限外不需要额外权限或 SCC

### 覆盖安全上下文

你也可以在 CR 中覆盖自动安全上下文：

```yaml
spec:
  # Container security context
  securityContext:
    runAsNonRoot: true
    runAsUser: 2000
    allowPrivilegeEscalation: false
    capabilities:
      drop:
        - ALL

  # Pod security context
  podSecurityContext:
    runAsNonRoot: true
    runAsUser: 2000
    fsGroup: 2000
```

:::caution OpenShift Note
在 OpenShift 上，建议省略 `runAsUser` 与 `fsGroup`，让 SCC 自动分配 UID/GID。
:::

## 配置参考

### 镜像配置

```yaml
spec:
  image:
    repository: ghcr.io/vllm-project/semantic-router/extproc
    tag: latest
    pullPolicy: IfNotPresent
    imageRegistry: ""  # 可选：自定义镜像仓库前缀

  # 可选：镜像拉取 Secret
  imagePullSecrets:
    - name: ghcr-secret
```

### Service 配置

```yaml
spec:
  service:
    type: ClusterIP  # 或 NodePort、LoadBalancer

    grpc:
      port: 50051
      targetPort: 50051

    api:
      port: 8080
      targetPort: 8080

    metrics:
      enabled: true
      port: 9190
      targetPort: 9190
```

### 持久化配置

```yaml
spec:
  persistence:
    enabled: true
    storageClassName: "standard"  # 根据你的集群调整
    accessMode: ReadWriteOnce
    size: 10Gi

    # 可选：使用现有 PVC
    existingClaim: "my-existing-pvc"

    # 可选：PVC annotation
    annotations:
      backup.velero.io/backup-volumes: "models"
```

:::info 存储验证
operator 会在创建 PVC 之前验证指定的 StorageClass 是否存在。如果省略 `storageClassName`，则使用集群的默认 StorageClass。
:::

**StorageClass 示例：**

- **AWS EKS**：`gp3-csi`、`gp2`
- **GKE**：`standard`、`premium-rwo`
- **Azure AKS**：`managed`、`managed-premium`
- **OpenShift**：`gp3-csi`、`thin`、`ocs-storagecluster-ceph-rbd`

### 语义缓存后端

operator 支持多种语义缓存后端。通过缓存相似查询及其响应，可以显著降低延迟和 token 用量。

:::warning 前置条件
operator **不会**部署 Redis 或 Milvus。在将其用作缓存后端之前，你必须在集群中单独部署这些服务。operator 只负责配置 SemanticRouter，使其连接到现有的 Redis/Milvus 部署。

部署示例请参阅下文的 [Redis](#deploying-redis) 与 [Milvus](#deploying-milvus) 小节。

**替代方案：** 如果希望自动部署 Redis/Milvus，可以考虑使用 [Helm chart](https://github.com/vllm-project/semantic-router/tree/main/deploy/helm)，它可以将缓存后端作为 Helm chart 依赖项进行部署。
:::

#### 支持的后端

##### 1. 内存缓存（默认）

简单的内存缓存，适用于开发环境和小规模部署。

**特性：**

- 无外部依赖
- 访问速度快
- 不持久化（重启时清空）
- 受 pod 内存限制

**配置：**

```yaml
spec:
  config:
    global:
      stores:
        semantic_cache:
          enabled: true
          backend_type: memory
          similarity_threshold: "0.8"
          max_entries: 1000
          ttl_seconds: 3600
          eviction_policy: fifo  # fifo、lru 或 lfu
```

**适用场景：**

- 开发与测试
- 小规模部署（缓存查询少于 1000 条）
- 没有持久化要求

##### 2. Redis 缓存

使用具备向量搜索能力的 Redis 构建高性能分布式缓存。

**特性：**

- 分布式且可扩展
- 持久化存储（使用 AOF/RDB）
- HNSW 或 FLAT 索引
- 生态支持广泛

**前置条件：**

- Redis 7.0+，并安装 RediSearch 模块
- 创建用于保存密码的 Kubernetes Secret：

```bash
kubectl create secret generic redis-credentials \
  --from-literal=password='your-redis-password'
```

**配置：**

```yaml
spec:
  config:
    global:
      stores:
        semantic_cache:
          enabled: true
          backend_type: redis
          similarity_threshold: "0.85"
          ttl_seconds: 3600
          embedding_model: mmbert
          redis:
            connection:
              host: redis.default.svc.cluster.local
              port: 6379
              database: 0
              password_secret_ref:
                name: redis-credentials
                key: password
              timeout: 30
              tls:
                enabled: false
            index:
              name: semantic_cache_idx
              prefix: "cache:"
              vector_field:
                name: embedding
                dimension: 768  # 与默认的 mmBERT ultra embedding 模型一致
                metric_type: COSINE
              index_type: HNSW
              params:
                M: 16
                efConstruction: 64
            search:
              topk: 1
            development:
              auto_create_index: true
              verbose_errors: true
```

**适用场景：**

- 中等规模的生产部署
- 需要持久化和高可用性
- 已有 Redis 基础设施
- 需要高速的内存性能

**示例：** [`vllm.ai_v1alpha1_semanticrouter_redis_cache.yaml`](https://github.com/vllm-project/semantic-router/blob/main/deploy/operator/config/samples/vllm.ai_v1alpha1_semanticrouter_redis_cache.yaml)

##### 3. Milvus 缓存

面向大容量缓存生产部署的企业级向量数据库。

**特性：**

- 高度可扩展且支持分布式部署
- 高级索引（HNSW、IVF 等）
- 内置数据生命周期管理
- 支持高可用性

**前置条件：**

- Milvus 2.3+（standalone 或集群模式）
- 创建用于保存凭据的 Kubernetes Secret：

```bash
kubectl create secret generic milvus-credentials \
  --from-literal=password='your-milvus-password'
```

**配置：**

```yaml
spec:
  config:
    global:
      stores:
        semantic_cache:
          enabled: true
          backend_type: milvus
          similarity_threshold: "0.90"
          ttl_seconds: 7200
          embedding_model: mmbert
          milvus:
            connection:
              host: milvus-standalone.default.svc.cluster.local
              port: 19530
              database: semantic_router_cache
              timeout: 30
              auth:
                enabled: true
                username: root
                password_secret_ref:
                  name: milvus-credentials
                  key: password
            collection:
              name: semantic_cache
              description: "Semantic cache for LLM responses"
              vector_field:
                name: embedding
                dimension: 768  # 与默认的 mmBERT ultra embedding 模型一致
                metric_type: IP
              index:
                type: HNSW
                params:
                  M: 16
                  efConstruction: 64
            search:
              params:
                ef: 64
              topk: 10
              consistency_level: Session
            performance:
              connection_pool:
                max_connections: 10
                max_idle_connections: 5
              batch:
                insert_batch_size: 100
            data_management:
              ttl:
                enabled: true
                timestamp_field: created_at
                cleanup_interval: 3600
            development:
              auto_create_collection: true
```

**适用场景：**

- 大规模生产部署
- 需要高级向量搜索能力
- 需要数据生命周期管理（TTL、compaction）
- 对高可用性和可扩展性有要求

**示例：** [`vllm.ai_v1alpha1_semanticrouter_milvus_cache.yaml`](https://github.com/vllm-project/semantic-router/blob/main/deploy/operator/config/samples/vllm.ai_v1alpha1_semanticrouter_milvus_cache.yaml)

##### 4. 混合缓存

将内存 HNSW 索引与 Milvus 持久化存储结合，以兼顾最佳性能和持久性。

**特性：**

- 使用 HNSW 进行高速内存搜索
- 在 Milvus 中持久化存储
- 兼具两者优势
- 自动同步

**配置：**

```yaml
spec:
  config:
    global:
      stores:
        semantic_cache:
          enabled: true
          backend_type: hybrid
          similarity_threshold: "0.85"
          ttl_seconds: 3600
          max_entries: 5000
          eviction_policy: lru
          embedding_model: mmbert
          # HNSW 内存配置
          hnsw:
            use_hnsw: true
            hnsw_m: 32
            hnsw_ef_construction: 128
            max_memory_entries: 5000

      # Milvus 持久化存储（配置与 milvus 后端相同）
      milvus:
        connection:
          host: milvus-standalone.default.svc.cluster.local
          port: 19530
          # ... 其余 milvus 配置
```

**适用场景：**

- 需要尽可能快的缓存查询
- 需要持久化和数据耐久性
- 愿意用内存换取性能
- 高吞吐量生产部署

**示例：** [`vllm.ai_v1alpha1_semanticrouter_hybrid_cache.yaml`](https://github.com/vllm-project/semantic-router/blob/main/deploy/operator/config/samples/vllm.ai_v1alpha1_semanticrouter_hybrid_cache.yaml)

#### 部署 Redis {#deploying-redis}

在使用 Redis 缓存后端之前，请先在集群中部署带有 RediSearch 模块的 Redis：

```yaml
---
apiVersion: v1
kind: Namespace
metadata:
  name: cache-backends
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: cache-backends
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis/redis-stack-server:latest
        ports:
        - containerPort: 6379
        resources:
          requests:
            cpu: 100m
            memory: 256Mi
          limits:
            cpu: 500m
            memory: 512Mi
---
apiVersion: v1
kind: Service
metadata:
  name: redis
  namespace: cache-backends
spec:
  type: ClusterIP
  ports:
  - port: 6379
    targetPort: 6379
  selector:
    app: redis
```

应用配置：

```bash
kubectl apply -f redis-deployment.yaml
```

创建凭据 Secret：

```bash
kubectl create secret generic redis-credentials \
  --from-literal=password=''  # 不使用密码时留空，也可以设置你的密码
```

**对于生产部署**，建议使用：

- [Redis Operator](https://github.com/spotahome/redis-operator)
- [Redis Enterprise Operator](https://docs.redis.com/latest/kubernetes/)
- 托管 Redis 服务（AWS ElastiCache、Azure Cache for Redis、GCP Memorystore）

#### 部署 Milvus {#deploying-milvus}

在使用 Milvus 缓存后端之前，请先在集群中部署 Milvus：

```yaml
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: milvus-standalone
  namespace: cache-backends
spec:
  replicas: 1
  selector:
    matchLabels:
      app: milvus
  template:
    metadata:
      labels:
        app: milvus
    spec:
      containers:
      - name: milvus
        image: milvusdb/milvus:v2.4.0
        command: ["milvus", "run", "standalone"]
        ports:
        - containerPort: 19530
          name: grpc
        - containerPort: 9091
          name: metrics
        env:
        - name: ETCD_USE_EMBED
          value: "true"
        - name: COMMON_STORAGETYPE
          value: "local"
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 2000m
            memory: 4Gi
        volumeMounts:
        - name: milvus-data
          mountPath: /var/lib/milvus
      volumes:
      - name: milvus-data
        emptyDir: {}  # 生产环境请使用 PVC
---
apiVersion: v1
kind: Service
metadata:
  name: milvus-standalone
  namespace: cache-backends
spec:
  type: ClusterIP
  ports:
  - port: 19530
    targetPort: 19530
    name: grpc
  selector:
    app: milvus
```

应用配置：

```bash
kubectl apply -f milvus-deployment.yaml
```

创建凭据 Secret：

```bash
kubectl create secret generic milvus-credentials \
  --from-literal=password='Milvus'  # Milvus 默认密码
```

**对于生产部署**，建议使用：

- [Milvus Operator](https://milvus.io/docs/install_cluster-milvusoperator.md)
- [Milvus Helm Chart](https://milvus.io/docs/install_cluster-helm.md)
- [Zilliz Cloud](https://cloud.zilliz.com/)（托管 Milvus 服务）

:::tip 生产最佳实践
对于生产环境的缓存后端：

1. 使用持久卷（而不是 emptyDir）
2. 启用身份验证和 TLS
3. 合理配置资源限制
4. 设置监控和告警
5. 使用 operator 或 Helm chart 简化管理
6. 考虑使用托管服务以降低运维开销
:::

#### Embedding 模型

语义缓存支持使用不同的 embedding 模型计算相似度：

- **bert**（默认）：轻量级，384 维，适合通用场景
- **qwen3**：质量更高，1024 维，准确性更好
- **gemma**：较为均衡，768 维，性能适中

配置方式如下：

```yaml
spec:
  config:
    global:
      stores:
        semantic_cache:
          embedding_model: mmbert  # 或通过显式覆盖 model_catalog 使用 qwen3、gemma
```

**注意：** 请确保缓存配置中的 `dimension` 与所选 embedding 模型一致。

#### 后端迁移

从内存缓存迁移到 Redis 或 Milvus 的流程很简单：

1. 在集群中部署 Redis 或 Milvus
2. 创建凭据 Secret
3. 使用新的后端配置更新 SemanticRouter CR
4. 应用变更，operator 将执行滚动更新

迁移后缓存为空，但会随着查询处理自然填充。

#### 缓存配置参考

如需详细的配置选项，请使用 `kubectl explain`：

```bash
# Redis 缓存配置
kubectl explain semanticrouter.spec.config.global.stores.semantic_cache.redis

# Milvus 缓存配置
kubectl explain semanticrouter.spec.config.global.stores.semantic_cache.milvus

# HNSW 配置
kubectl explain semanticrouter.spec.config.global.stores.semantic_cache.hnsw
```

### Semantic Router 配置

完整的 semantic router 配置嵌入在 CR 中。请参阅上方的完整示例以及 [`deploy/operator/config/samples/`](https://github.com/vllm-project/semantic-router/tree/main/deploy/operator/config/samples)。

主要配置部分如下：

```yaml
spec:
  config:
    providers:
      defaults:
        reasoning_families:
          deepseek:
            type: "chat_template_kwargs"
            parameter: "thinking"
          qwen3:
            type: "chat_template_kwargs"
            parameter: "enable_thinking"
          gpt:
            type: "reasoning_effort"
            parameter: "reasoning_effort"

    global:
      model_catalog:
        embeddings:
          semantic:
            mmbert_model_path: "models/mom-embedding-ultra"
            use_cpu: true
            embedding_config:
              model_type: "mmbert"
              preload_embeddings: true
              target_dimension: 768
              target_layer: 22
              top_k: 1
              min_score_threshold: 0.5
        system:
          prompt_guard: "models/mmbert32k-jailbreak-detector-merged"
          domain_classifier: "models/mmbert32k-intent-classifier-merged"
          pii_classifier: "models/mmbert32k-pii-detector-merged"
        modules:
          prompt_guard:
            enabled: true
            model_id: "models/mmbert32k-jailbreak-detector-merged"
            threshold: 0.7
            use_cpu: true
            use_mmbert_32k: true
          classifier:
            domain:
              model_id: "models/mmbert32k-intent-classifier-merged"
              threshold: 0.5
              use_cpu: true
              use_mmbert_32k: true
            pii:
              model_id: "models/mmbert32k-pii-detector-merged"
              threshold: 0.9
              use_cpu: true
              use_mmbert_32k: true

      stores:
        semantic_cache:
          enabled: true
          backend_type: "memory"  # 或 redis、milvus、hybrid
          similarity_threshold: 0.8
          max_entries: 1000
          ttl_seconds: 3600
          eviction_policy: "fifo"

      integrations:
        tools:
          enabled: true
          top_k: 3
          similarity_threshold: 0.2
          tools_db_path: "config/tools_db.json"
          fallback_to_empty: true

      services:
        api:
          batch_classification:
            max_batch_size: 100
            concurrency_threshold: 5
            max_concurrency: 8
            metrics:
              enabled: true
              detailed_goroutine_tracking: true
              sample_rate: 1.0
        observability:
          tracing:
            enabled: false
            provider: "opentelemetry"
            exporter:
              type: "otlp"
              endpoint: "jaeger:4317"
```

### 工具数据库

定义可供自动选择的工具：

```yaml
spec:
  toolsDb:
    - tool:
        type: "function"
        function:
          name: "search_web"
          description: "Search the web for information"
          parameters:
            type: "object"
            properties:
              query:
                type: "string"
                description: "Search query"
            required: ["query"]
      description: "Search the internet, web search, find information online"
      category: "search"
      tags: ["search", "web", "internet"]

    - tool:
        type: "function"
        function:
          name: "calculate"
          description: "Perform mathematical calculations"
          parameters:
            type: "object"
            properties:
              expression:
                type: "string"
            required: ["expression"]
      description: "Calculate mathematical expressions"
      category: "math"
      tags: ["math", "calculation"]
```

### 自动扩缩容（HPA）

```yaml
spec:
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 10
    targetCPUUtilizationPercentage: 70
    targetMemoryUtilizationPercentage: 80
```

### Ingress 配置

```yaml
spec:
  ingress:
    enabled: true
    className: "nginx"  # 或 "haproxy"、"traefik" 等
    annotations:
      cert-manager.io/cluster-issuer: "letsencrypt-prod"
    hosts:
      - host: router.example.com
        paths:
          - path: /
            pathType: Prefix
            servicePort: 8080
    tls:
      - secretName: router-tls
        hosts:
          - router.example.com
```

## 生产部署

### 高可用设置

```yaml
apiVersion: vllm.ai/v1alpha1
kind: SemanticRouter
metadata:
  name: prod-router
spec:
  replicas: 3

  # 用反亲和性将 pod 分散到不同节点
  affinity:
    podAntiAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        - labelSelector:
            matchLabels:
              app.kubernetes.io/instance: prod-router
          topologyKey: kubernetes.io/hostname

  # 自动扩缩容
  autoscaling:
    enabled: true
    minReplicas: 3
    maxReplicas: 20
    targetCPUUtilizationPercentage: 70

  # 生产环境资源
  resources:
    limits:
      memory: "10Gi"
      cpu: "4"
    requests:
      memory: "5Gi"
      cpu: "2"

  # 严格的探针配置
  livenessProbe:
    enabled: true
    initialDelaySeconds: 60
    periodSeconds: 30
    failureThreshold: 3

  readinessProbe:
    enabled: true
    initialDelaySeconds: 30
    periodSeconds: 10
    failureThreshold: 3
```

### Pod Disruption Budget

创建 PDB，确保更新期间的可用性：

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: prod-router-pdb
spec:
  maxUnavailable: 1
  selector:
    matchLabels:
      app.kubernetes.io/instance: prod-router
```

### 资源分配指南

| 工作负载类型 | 内存请求 | CPU 请求 | 内存限制 | CPU 限制 |
|-------------|----------|----------|----------|----------|
| 开发        | 1Gi      | 500m     | 2Gi      | 1        |
| 预发布      | 3Gi      | 1        | 7Gi      | 2        |
| 生产        | 5Gi      | 2        | 10Gi     | 4        |

## 监控与可观测性

### 指标

Prometheus 指标通过 9190 端口暴露：

```bash
# 通过端口转发在本地访问指标
kubectl port-forward svc/my-router 9190:9190

# 查看指标
curl http://localhost:9190/metrics
```

**关键指标：**

| 指标族 | 示例指标 |
|--------|----------|
| 请求 | `llm_model_requests_total`, `llm_request_errors_total` |
| 错误 | `llm_request_errors_total{reason="timeout"}` |
| 延迟 | `llm_model_completion_latency_seconds`, `llm_model_ttft_seconds`, `llm_model_tpot_seconds`, `llm_model_routing_latency_seconds` |
| Token 与成本 | `llm_model_tokens_total`, `llm_model_prompt_tokens_total`, `llm_model_completion_tokens_total`, `llm_model_cost_total` |
| 路由 | `llm_model_routing_modifications_total`, `llm_routing_reason_codes_total` |
| 选择 | `llm_model_selection_total`, `llm_model_selection_duration_seconds`, `llm_model_inflight_requests` |
| 缓存 | `llm_cache_plugin_hits_total`, `llm_cache_plugin_misses_total`, `llm_cache_warmth_estimate` |
| RAG | `rag_retrieval_attempts_total`, `rag_retrieval_latency_seconds`, `rag_cache_hits_total`, `rag_cache_misses_total` |
| 会话 | `llm_session_model_transitions_total`, `llm_session_turn_prompt_tokens`, `llm_session_turn_completion_tokens`, `llm_session_turn_cost` |
| 翻译与请求参数策略 | `llm_translation_lossy_total`, `sr_request_params_blocked_total`, `sr_request_params_unknown_field_stripped_total` |

### ServiceMonitor（Prometheus Operator）

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: semantic-router-metrics
spec:
  selector:
    matchLabels:
      app.kubernetes.io/instance: my-router
  endpoints:
    - port: metrics
      interval: 30s
      path: /metrics
```

### 分布式追踪

启用 OpenTelemetry tracing：

```yaml
spec:
  config:
    observability:
      tracing:
        enabled: true
        provider: "opentelemetry"
        exporter:
          type: "otlp"
          endpoint: "jaeger-collector:4317"
          insecure: true
        sampling:
          type: "probabilistic"
          rate: 0.1
```

推荐使用 `probabilistic` 采样类型。为兼容现有配置，也接受 `traceidratio` 和 `trace_id_ratio` 作为别名。

## 故障排除

### 常见问题

#### 后端发现失败

**症状：** 日志中出现 "No backends found" 或 "Failed to discover backend"

**对于 KServe 后端：**

```bash
# 检查 InferenceService 是否存在并已就绪
kubectl get inferenceservice llama-3-8b

# 检查 KServe 是否已创建 predictor service
kubectl get service llama-3-8b-predictor

# 验证 InferenceService 状态
kubectl describe inferenceservice llama-3-8b
```

**对于 Llama Stack 后端：**

```bash
# 验证具有正确 label 的 service 是否存在
kubectl get services -l app=llama-stack,model=llama-3.3-70b

# 检查 service label 是否与 CR 中的 discoveryLabels 一致
kubectl get service <service-name> -o jsonpath='{.metadata.labels}'
```

**对于直连 service 后端：**

```bash
# 验证 service 是否存在于指定 namespace
kubectl get service vllm-deepseek -n vllm-serving

# 检查 service 是否定义了端口
kubectl get service vllm-deepseek -n vllm-serving -o jsonpath='{.spec.ports[0]}'
```

#### Gateway 集成问题

**症状：** 尚不存在 HTTPRoute，或流量未到达 semantic router

当前 operator 实现不会自动创建 HTTPRoute 资源。如果你尚未自行应用 HTTPRoute manifest，这属于预期行为，并非 reconcile 失败。

```bash
# 验证 Gateway 是否存在
kubectl get gateway istio-ingressgateway -n istio-system

# 检查你是否自行创建了 HTTPRoute
kubectl get httproute -l app.kubernetes.io/instance=my-router

# 验证 Gateway 是否支持 HTTPRoute（Gateway API v1）
kubectl get gateway istio-ingressgateway -n istio-system -o yaml | grep -A5 listeners

# 检查 operator 状态
kubectl get semanticrouter my-router -o jsonpath='{.status.gatewayMode}'
# 应显示："gateway-integration"

# 检查 operator 日志中的当前占位消息
kubectl logs deployment/semantic-router-operator-controller-manager -n semantic-router-system | grep 'HTTPRoute creation placeholder'
```

如果 operator 日志显示 `HTTPRoute creation placeholder - requires Gateway API version-specific implementation`，说明 controller 已识别 gateway mode，但没有创建 route。请手动应用 HTTPRoute manifest，并验证它是否以 8080 端口上的 semantic router service 为目标。

#### OpenShift Route 问题

**症状：** OpenShift 上未创建 Route

```bash
# 验证是否运行在 OpenShift 集群上
kubectl api-resources | grep route.openshift.io

# 检查 Route 是否已创建
kubectl get route -l app.kubernetes.io/instance=my-router

# 检查 operator 是否探测到 OpenShift
kubectl logs -n semantic-router-operator-system \
  deployment/semantic-router-operator-controller-manager \
  | grep -i "openshift\|route"

# 验证 Route 状态
kubectl get semanticrouter my-router -o jsonpath='{.status.openshiftFeatures}'
```

#### Pod 卡在 `ImagePullBackOff`

```bash
# 检查镜像拉取 Secret
kubectl describe pod <pod-name>

# 创建镜像拉取 Secret
kubectl create secret docker-registry ghcr-secret \
  --docker-server=ghcr.io \
  --docker-username=<username> \
  --docker-password=<personal-access-token>

# 添加到 SemanticRouter
spec:
  imagePullSecrets:
    - name: ghcr-secret
```

#### PVC 卡在 `Pending`

```bash
# 检查 StorageClass 是否存在
kubectl get storageclass

# 检查 PVC event
kubectl describe pvc my-router-models

# 更新 CR 中的 StorageClass
spec:
  persistence:
    storageClassName: "your-available-storage-class"
```

#### 模型无法下载

```bash
# 检查 HF token Secret 是否存在
kubectl get secret hf-token-secret

# 创建 HF token Secret
kubectl create secret generic hf-token-secret \
  --from-literal=token=hf_xxxxxxxxxxxxx

# 添加到 SemanticRouter CR
spec:
  env:
    - name: HF_TOKEN
      valueFrom:
        secretKeyRef:
          name: hf-token-secret
          key: token
```

#### operator 未正确探测平台

```bash
# 检查 operator 日志中的平台探测信息
kubectl logs -n semantic-router-operator-system \
  deployment/semantic-router-operator-controller-manager \
  | grep -i "platform\|openshift"

# 应看到以下消息之一：
# "Detected OpenShift platform - will use OpenShift-compatible security contexts"
# "Detected standard Kubernetes platform - will use standard security contexts"
```

## 开发

### 本地开发

```bash
cd deploy/operator

# 运行测试
make test

# 生成 CRD 和代码
make generate
make manifests

# 构建 operator 二进制文件
make build

# 使用你的 kubeconfig 在本地运行
make run
```

### 使用 kind 测试

```bash
# 创建 kind 集群
kind create cluster --name operator-test

# 构建并加载镜像
make docker-build IMG=semantic-router-operator:dev
kind load docker-image semantic-router-operator:dev --name operator-test

# 部署
make install
make deploy IMG=semantic-router-operator:dev

# 创建测试实例
kubectl apply -f config/samples/vllm_v1alpha1_semanticrouter.yaml
```

## 下一步

- [配置 semantic router 功能](../configuration)
- [配置监控与可观测性](../../tutorials/global/api-and-observability)
- [探索其他部署方式](/docs/installation)
- [加入社区](../../community/overview)

---
sidebar_position: 3
sidebar_label: 使用 Operator 安装
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

1. 在 pod spec 中部署两个容器：semantic router + Envoy sidecar
2. Envoy 负责 ingress，并通过 ExtProc gRPC 转发到 semantic router
3. status 显示 `gatewayMode: "standalone"`

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

::::caution OpenShift Note
在 OpenShift 上，建议省略 `runAsUser` 与 `fsGroup`，让 SCC 自动分配 UID/GID。
::::

## 下一步

- [配置 semantic router 功能](../configuration)
- [配置监控与可观测性](../../tutorials/global/api-and-observability)
- [探索其他部署方式](/docs/installation)
- [加入社区](../../community/overview)

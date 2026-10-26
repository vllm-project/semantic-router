---
is_mtpe: false
sidebar_position: 3
sidebar_label: Kubernetes 操作符
title: 使用 Kubernetes 操作符部署
description: 安装 Semantic Router Operator，连接模型服务，并创建第一个 Router 部署。
translation:
  source_commit: "f8c1197a9ed47f7a265cbab83bf2d84eb5fa505e"
  source_file: "docs/installation/k8s/operator.md"
  outdated: false
---

# 使用 Kubernetes 操作符部署

Semantic Router Operator 将 `SemanticRouter` 自定义资源协调为 Router 工作负载、Service、配置、存储和可选平台集成。当 Kubernetes 应负责 Router 生命周期，且模型后端已作为 Kubernetes 服务、KServe `InferenceService` 或 Llama Stack 服务暴露时，使用它。

Operator 不部署模型服务器。它发现或引用它们，并生成 Router 使用的提供商 binding。

## Operator 管理什么

- Router Deployment 和 Service
- 从自定义资源生成的规范 Router 配置
- 可选的持久模型存储
- probe、资源、调度、自动扩缩容和 ingress 设置
- OpenShift 安全默认值和可选的 Route 创建
- standalone 模式（Router 自己提供 listener），或与现有 Gateway 集成

顶层字段族以及已安装 schema 的链接，见 [SemanticRouter CRD 参考](../../api/semantic-router-crd)。

## 前置条件

- 受支持的 Kubernetes 或 OpenShift 集群
- 已针对目标集群配置的 `kubectl` 或 `oc`
- 下面基于源码的安装需要 Git、GNU Make 和 Go 1.25 或更高
- 安装 CRD 和集群范围 RBAC 的权限
- 至少一个可达的 OpenAI 兼容模型服务

## 安装 Operator

### 在 Kubernetes 上使用 Kustomize

```bash
git clone https://github.com/vllm-project/semantic-router.git
cd semantic-router/deploy/operator

make install
make deploy IMG=ghcr.io/vllm-project/semantic-router/operator:latest
```

验证控制器：

```bash
kubectl get pods -n semantic-router-operator-system
kubectl logs -n semantic-router-operator-system \
  deployment/semantic-router-operator-controller-manager
```

受控环境请固定已发布的镜像标签或 digest，而不是使用 `latest`。

### 在 OpenShift 上使用 OLM

在 OpenShift 上直接部署控制器时，使用上面的 Kustomize 流程。协调器会检测 OpenShift 并应用其平台特定的工作负载默认值。

对于 Operator Lifecycle Manager 安装，先发布或选择包含 Semantic Router bundle 的 OLM catalog，然后创建集群所需的 `CatalogSource`、`OperatorGroup` 和 `Subscription`。仓库的 `make openshift-deploy` 目标只创建命名空间、OperatorGroup 和 Subscription；它假定 `openshift-marketplace` 中已存在 `semantic-router-catalog` CatalogSource。因此它是维护者便利目标，不是独立安装命令。

## 创建第一个 Router

此示例绑定同一命名空间中名为 `model-server` 的现有 Service。Operator 创建提供商模型和模型卡，并将第一个发现的模型作为默认值。

```yaml
apiVersion: vllm.ai/v1alpha1
kind: SemanticRouter
metadata:
  name: my-router
  namespace: default
spec:
  replicas: 1
  vllmEndpoints:
    - name: local-backend
      model: local/model
      backend:
        type: service
        service:
          name: model-server
          port: 8000
  resources:
    requests:
      cpu: 500m
      memory: 1Gi
    limits:
      cpu: "2"
      memory: 4Gi
```

应用并等待就绪：

```bash
kubectl apply -f my-router.yaml
kubectl get semanticrouter my-router -w
kubectl get deployment,service \
  -l app.kubernetes.io/instance=my-router
```

用具体模型名称发送直接请求，或在 `spec.config.routing` 下添加规范路由，以暴露自动或策略驱动行为。

## 添加路由策略

`spec.config.routing` 接受规范路由对象。下面的示例为已发现的模型添加兜底决策：

```yaml
spec:
  config:
    routing:
      strategy: priority
      modelCards:
        - name: local/model
          modality: text
          capabilities: [chat]
      decisions:
        - name: default-route
          description: Route unmatched requests to the discovered local model.
          priority: 1
          rules:
            operator: AND
            conditions: []
          modelRefs:
            - model: local/model
```

其余 `spec.config` 字段是 Operator 对共享 Router 设置的 adapter，例如响应缓存、分类器、工具、可观测性和推理家族。它们会被转换为规范 `global` 和提供商段落。查阅 CRD 参考，不要把本地 `config.yaml` 中的字段复制到任意 CR 路径。

## 后端发现

每个 `spec.vllmEndpoints[]` 条目声明一个逻辑模型和一种解析其后端的方式：

| `backend.type` | 何时使用 | 必填字段 |
|----------------|----------|-----------------|
| `service` | 已存在 OpenAI 兼容的 Kubernetes Service | `service.name`、`service.port`；可选 `service.namespace` |
| `kserve` | KServe 负责模型部署 | `inferenceServiceName` |
| `llamastack` | 应按标签选择服务 | `discoveryLabels` |

另一个命名空间中的服务示例：

```yaml
spec:
  vllmEndpoints:
    - name: qwen-backend
      model: qwen/assistant
      reasoning:
        family: qwen3
      backend:
        type: service
        service:
          name: qwen-vllm
          namespace: model-serving
          port: 8000
```

`model` 值必须匹配提供商所服务的名称。`name` 值标识生成的后端引用。可选的 LoRA 声明会成为生成的路由模型卡下的条目。

## 部署模式

### 独立模式

没有 `spec.gateway` 时，Router 以 standalone 模式运行（`-gateway=standalone`）：它在自己的 listener `http-8801`（端口 **8801**）上直接提供 OpenAI 兼容 API，为每个请求做路由，并转发到所选后端。Pod 中不运行 Envoy。Pod 的探针检查该 listener 上的 `/ready` 和 `/health`，Service 将其暴露为 8801 端口（`http-8801`）。

当 Router 应自包含，且集群尚未提供兼容网关时，此模式适用。

standalone 模式之前的 Operator 版本在此模式下运行 Envoy sidecar，并提供同一个 Service 端口 8801。升级 Operator 后，由 Router 自己提供该端口；滚动更新完成后，Operator 会删除 sidecar 使用的 `<name>-envoy-config` ConfigMap。

### 现有 Gateway {#existing-gateway}

接入现有 Gateway 有两种不同方式。

#### 将 HTTP 转发到 standalone Router

对于转发 HTTP 的 Gateway API 控制器，保持 Router 处于 standalone 模式：**省略 `spec.gateway.existingRef`**。Router 的推理 listener 是 Service 端口 **8801**（`http-8801`）。创建指向该端口的 `HTTPRoute`：

```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: my-router
  namespace: default
spec:
  parentRefs:
    - name: shared-gateway
      namespace: gateway-system
  rules:
    - matches:
        - path:
            type: PathPrefix
            value: /v1
      backendRefs:
        - name: my-router
          port: 8801
      timeouts:
        request: 300s
        backendRequest: 300s
```

被引用的 Gateway 监听器必须通过 `allowedRoutes.namespaces` 允许来自 Router 命名空间的路由。上面的路由和 Router Service 共享命名空间；将后端 Service 移到另一个命名空间还需要 `ReferenceGrant`。按部署配置 Gateway 的主机名、TLS 和认证。Operator 不会创建 `HTTPRoute`；你负责此路由的生命周期。

流量路径为 Gateway → Router Service `http-8801` → Router → 所选模型后端。`api` 端口（默认 8080）服务 Router 管理请求，不得作为推理路由的后端。

完整的 Router 和路由示例见 [`vllm.ai_v1alpha1_semanticrouter_gateway.yaml`](https://github.com/vllm-project/semantic-router/blob/main/deploy/operator/config/samples/vllm.ai_v1alpha1_semanticrouter_gateway.yaml)。适配其现有 Gateway 和模型 Service 名称后，应用这两个资源，并在发送真实 completion 前检查路由是否被接受。这些命令使用该示例的资源名称和命名空间：

```bash
kubectl -n vllm-serving get httproute semantic-router-gateway -o yaml
kubectl -n vllm-serving get service semantic-router-gateway -o jsonpath='{.spec.ports}'
kubectl -n vllm-serving port-forward service/semantic-router-gateway 8801:8801
# 在另一个终端中，通过这个本地数据面验证同一请求。
curl --fail-with-body http://localhost:8801/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local/model","messages":[{"role":"user","content":"Hello"}]}'
```

然后通过你的 Gateway URL 发送请求。其 `HTTPRoute` 父状态必须报告 `Accepted=True` 和 `ResolvedRefs=True`；仅路由被接受并不能证明后端推理成功。

#### 让现有 Gateway 调用 ExtProc

仅当现有 Gateway 将负责推理数据面，并且已配置该网关特定集成时，才使用 `spec.gateway.existingRef`：

```yaml
spec:
  gateway:
    existingRef:
      name: shared-gateway
      namespace: gateway-system
```

此模式验证 Gateway 存在，并以 extproc 模式运行 Router（`-gateway=extproc`）。你必须配置 Gateway 的 ExtProc 策略，使其调用 Router Service 的 **gRPC 端口 50051**（或你配置的 `service.grpc.port`），以及指向实际模型后端、并尊重 Router 选择的路由。仅引用 Gateway 并不会安装这些策略或路由。按匹配的 [Kubernetes 网关集成指南](gateways) 了解其处理模式和后端选择约定。不要将推理 `HTTPRoute` 指向 Router 管理 API；它不实现 `/v1/chat/completions`。

当该 ExtProc 策略以 `STREAMED` 或 `FullDuplexStreamed` 模式发送请求体时，还要按 [Streamed ExtProc](streamed-extproc) 所述设置 `spec.config.streamed_body.enabled: true`。

### OpenShift Route

在 OpenShift 上，Operator 可以创建 Route：

```yaml
spec:
  openshift:
    routes:
      enabled: true
      tls:
        termination: edge
        insecureEdgeTerminationPolicy: Redirect
```

省略主机名可让 OpenShift 分配一个，或提供由你的 DNS 和证书配置覆盖的主机名。

## 密钥

将提供商、仓库和模型下载凭证放在 Kubernetes Secrets 中。从 `spec.env` 引用它们，不要在自定义资源中放入字面值：

```yaml
spec:
  env:
    - name: HF_TOKEN
      valueFrom:
        secretKeyRef:
          name: model-download-credentials
          key: token
```

应用最小权限 RBAC，并限制谁可以读取生成的 ConfigMaps、Secrets、日志和自定义资源。见[安全加固](../security-hardening)。

## 验证部署

```bash
kubectl get semanticrouter my-router -o wide
kubectl describe semanticrouter my-router
kubectl get pods,service,pvc \
  -l app.kubernetes.io/instance=my-router
kubectl logs deployment/my-router -c semantic-router
```

同时检查协调和数据面行为：

1. 自定义资源的 observed generation 与其 generation 匹配；
2. 预期副本已就绪；
3. 每个已发现的后端存在，并服务所配置的模型名称；以及
4. 通过 Service 或 Gateway 的真实 completion 成功。

## 下一步

- [运维 Operator 部署](operator-operations)
- [SemanticRouter CRD 参考](../../api/semantic-router-crd)
- [配置工作流](../configuration-workflows)
- [Kubernetes Gateway API](gateway-api-inference-extension)
- [存储选项](../storage-overview)

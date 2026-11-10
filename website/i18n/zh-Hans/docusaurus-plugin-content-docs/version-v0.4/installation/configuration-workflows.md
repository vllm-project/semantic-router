---
title: 配置工作流
description: 选择 CLI、控制面板、Helm、Operator 和 DSL 如何编写并应用同一份 canonical Router 配置。
translation:
  source_commit: "8ded1a3c28a4af8358c8d638955b0318caeb8ed4"
  source_file: "docs/installation/configuration-workflows.md"
  outdated: false
---

# 配置工作流

所有受支持的工作流都生成或消费同一份 canonical 配置。为部署选择一个主要事实来源，并用其他界面检查或校验它，而不是独立覆盖它。

## CLI 和 YAML

当配置属于源代码控制或现有部署流水线时，使用 YAML：

```bash
vllm-sr config init --output config.yaml
vllm-sr config validate --config config.yaml
vllm-sr serve --config config.yaml
```

`config init` 写入打包的最小 canonical 模板，并且除非显式指定 `--force`，否则拒绝替换现有文件。当 Router 已经在运行时，从 `vllm-sr config get` 开始，以便保留不相关的活动设置。在 `providers.models` 中声明物理模型，并在适用决策的 `modelRefs` 中引用其名称。命名配方的决策位于 `recipes[].routing.decisions`；可复用的内置配方在发布入口点时接收模型分配。可选模型元数据统一保留在顶层 `routing.modelCards`，并根据所选算法或能力补充上下文限制、LoRA 适配器等必需信息。

本地运行时在运行时拥有的状态中派生栈专用服务地址，而不会重写源文件。同一运行时和栈的并发 `serve` 和 `stop` 操作会串行化；在活动生命周期操作完成后重试。

## 控制面板

空的本地工作区会以设置模式启动控制面板。用它绑定模型端点、选择基线策略、预览结果，并激活完整配置。

激活后，**Mixture-of-Models** 工作区将三项任务分开：

- **Built-in Models** 发现已安装的虚拟模型及其 Model Card；
- **Models & Routing** 编辑物理模型、入口点、配方和路由；
- **Probes** 检查配方场景，并支持生成或仅路由校验。

将 provider 生成与路由评估分开验证。即使所选后端无法生成，探针也可以选择预期路由。

可视化 DSL 编辑器拥有路由语义。它在替换其路由表面时保留 listeners、providers、全局设置和设置状态。多配方生命周期变更从 Models & Routing 或管理 API 管理，以免可视化编辑悄悄丢弃另一个配方。

## Helm {#helm}

直接 Helm 部署将完整的 canonical 文档放在 `configOverride` 下。这会在 chart 应用显式 Kubernetes 集成重写之前，将 chart 的示例配置作为一份文档替换；它不会将示例模型或决策合并到你的策略中。将 canonical 文档编写并校验为 `config.yaml`，然后将该文档放在 Helm values 文件的 `configOverride` 下。

```yaml
configOverride:
  version: v0.3
  listeners:
    - name: grpc-50051
      address: 0.0.0.0
      port: 50051
      timeout: 300s
    - name: http-8080
      address: 0.0.0.0
      port: 8080
      timeout: 300s
  providers:
    defaults:
      model: local/general
    models:
      - name: local/general
        provider_model_id: my-served-model
        backend_refs:
          - name: primary
            endpoint: model-server.default.svc.cluster.local:8000
            protocol: http
            provider: vllm
            weight: 100
  routing:
    strategy: priority
    modelCards:
      - name: local/general
        modality: text
        capabilities: [chat]
    decisions:
      - name: default-route
        description: Route requests to the configured model.
        priority: 1
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: local/general
            use_reasoning: false
  global:
    services:
      response_api:
        enabled: false
        store_backend: memory
```

```bash
vllm-sr config validate --config config.yaml

helm upgrade --install semantic-router \
  oci://ghcr.io/vllm-project/charts/semantic-router \
  -f values.yaml
```

`vllm-sr serve --target k8s --config config.yaml` 将所选文档作为原子覆盖传递，因此 chart 示例路由不能合并到其中。该命令拒绝空文档或仅包含 setup 的文档，并且不会注入本地 Docker 服务地址或知识库路径。先运行 `vllm-sr config validate`，以便 schema 和引用错误在部署前失败。

通过 Helm 或 Operator 选择 Kubernetes GPU 镜像、资源和设备插件。本地 `--platform amd` 和 `--platform nvidia` 快捷方式不会配置 Kubernetes 调度。

chart 将控制面板作为自己的 Deployment 和 Service 运行，并且该 Deployment 默认禁用。Router Service 仅承载 gRPC 和 HTTP API 端口，因此仅在启用控制面板后，端口 8700 才会出现在集群中。

```bash
helm upgrade --install semantic-router \
  oci://ghcr.io/vllm-project/charts/semantic-router \
  -f values.yaml --set dashboard.enabled=true

kubectl --namespace vllm-semantic-router-system port-forward \
  svc/semantic-router-dashboard 8700:8700
```

## Operator

Operator 从两个 Kubernetes 原生输入渲染 canonical 配置：

- `spec.vllmEndpoints` 发现模型服务，并创建 provider 绑定和 model card；以及
- `spec.config.routing` 接受 canonical 路由对象。

其他 `spec.config` 字段是用于响应缓存、分类器、工具、可观测性、推理家族和相关共享设置的类型化 Operator 适配器。Operator 将它们翻译为 canonical provider 和 `global` 节。

```yaml
spec:
  vllmEndpoints:
    - name: local-backend
      model: local/model
      backend:
        type: service
        service:
          name: model-server
          port: 8000
  config:
    routing:
      strategy: priority
```

不要将任意 `providers` 或 `global` 键直接复制到 `spec.config` 下；它们不是 CRD 字段。参见 [Kubernetes Operator](k8s/operator) 和 [SemanticRouter CRD 参考](../api/semantic-router-crd)。

## 路由 DSL

DSL 是用于 model card、信号、投影、决策、算法、插件、入口点和配方的聚焦编写表面。Providers、listeners、凭据和全局服务仍由 YAML 拥有。

当路由策略受益于紧凑、可复核的表示时，使用 DSL。将 canonical YAML 作为完整的部署产物。

## 避免分割所有权

- 不要将生成的运行时配置当作源文档来编辑。
- 不要让 GitOps 和交互式控制面板会话在没有显式交接的情况下写入同一部署。
- 不要将密钥放入 DSL、ConfigMap 或已提交的 YAML。
- 不要假设已评估的路由能证明后端就绪；请验证生成。
- 在将完整变更应用到实时栈之前，先预览并校验。

管理端点和并发契约见[管理 API 参考](../api/apiserver)。

---
is_mtpe: false
title: 配置工作流
description: 选择 CLI、控制面板、Helm、Operator 和 DSL 如何编写并应用同一份 canonical Router 配置。
translation:
  source_commit: "bc5c6bfb747ffe9c5f35d98b203427dada8c4c56"
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

- **从模板开始。** `config init` 写入打包的最小 canonical 模板；除非传入 `--force`，否则拒绝替换现有文件。
- **或从运行中的 Router 开始。** Router 已经在运行时，从 `vllm-sr config get` 开始，以便保留不相关的活动设置。
- **模型与决策。** 在 `providers.models` 中声明物理模型，并在 `recipes[].decisions` 下各决策的 `modelRefs` 中引用其名称。内置配方在发布入口时获得模型分配。
- **模型事实**保留在共享的 `providers.models` 条目上。按所选算法或能力补充所需信息，例如上下文上限或 LoRA 适配器。

本地运行时把栈专用的服务地址保存在自己的状态中，而不会重写源文件。同一栈上并发的 `serve` 和 `stop` 操作会依次执行；请在当前操作完成后重试。

## 控制面板

空的本地工作区会以设置模式启动控制面板。用它绑定模型端点、选择基线策略、预览结果，并激活完整配置；仍在等待的 `vllm-sr serve` 随后用这份配置启动 Router。

激活后，从 **Build → Routing → Models** 连接物理 Chat 后端。**Mixture-of-Models** 有三个 tab：

- **Models** 发布和管理公开模型入口；
- **Recipes** 管理入口选择的策略；
- **Probes** 检查配方场景，并支持生成或仅路由校验。

**Build → System One** 单独提供 **Decision Models**、**Decision Playground** 和 **Decision Monitoring**，用于管理判断模型部署、测试类型化问题及查看运行时观测。

将 provider 生成与路由评估分开验证。即使所选后端无法生成，探针也可以选择预期路由。

可视化 DSL 编辑器拥有路由语义。它在替换其路由表面时保留 listeners、providers、全局设置和设置状态。多配方生命周期变更从 Mixture-of-Models 或管理 API 管理，以免可视化编辑悄悄丢弃另一个配方。

## Helm {#helm}

直接 Helm 部署将完整的 canonical 文档放在 `configOverride` 下。这会在 chart 应用显式 Kubernetes 集成重写之前，将 chart 的示例配置作为一份文档替换；它不会将示例模型或决策合并到你的策略中。将 canonical 文档编写并校验为 `config.yaml`，然后将该文档放在 Helm values 文件的 `configOverride` 下。

```yaml title="values.yaml"
configOverride:
  version: v0.5
  listeners:
    - name: http-8899
      address: 0.0.0.0
      port: 8899
      timeout: 300s
  providers:
    defaults:
      model: local/general
    models:
      - name: local/general
        provider_model_id: my-served-model
        capabilities: [chat]
        backend_refs:
          - name: primary
            endpoint: model-server.default.svc.cluster.local:8000
            protocol: http
            provider: vllm
            weight: 100
  recipes:
    - name: default
      strategy: priority
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

Helm values 在安装时为 Router ConfigMap 提供初始内容：

- **在线编辑在升级后保留。** 之后由 Dashboard 或 Router API 保存的配置编辑，会在普通的 Helm 升级中保留。
- **有意应用修订后的 `configOverride`**，需要在那次升级中修改 `configMap.applyValuesRevision`。之后的升级沿用同一修订值时，会保留后续的在线编辑。
- **滚动发布。** 保存的 ConfigMap 编辑会在 Router Deployment 滚动发布后生效。

`vllm-sr serve --target kubernetes --config config.yaml` 将所选文档作为原子覆盖传递，因此 chart 示例路由不能合并到其中。该命令拒绝空文档或仅包含 setup 的文档，并且不会注入本地 Docker 服务地址或知识库路径。先运行 `vllm-sr config validate`，以便 schema 和引用错误在部署前失败。

在 Kubernetes 上，`--platform rocm` 和 `--platform cuda` 会选择 ROCm 或 CUDA 版 Router 镜像（除非显式指定了镜像）。它们按 canonical 设备序号申请 `amd.com/gpu` 或 `nvidia.com/gpu` 资源：共享同一设备的副本需要一块 GPU，不同的序号各需一块。

Pod 会被调度到其设备插件提供这些 GPU 的节点上。节点选择器、容忍度、其他资源以及设备插件本身仍通过 Helm values 或 Operator 配置。

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

- `spec.vllmEndpoints` 发现模型服务，并创建 provider 绑定；以及
- `spec.config.recipes`、`spec.config.entrypoints`、`spec.config.plugins` 和 `spec.config.models` 接受 canonical 配方、入口点、默认插件和模型事实。

- **模型目录。** 由 `spec.config.model_deployments`、`model_bindings` 和 `decision_model` 承载；见[模型部署与决策模型](../api/semantic-router-crd#model-deployments-and-the-decision-model)。
- **类型化适配器。** 其他 `spec.config` 字段（例如响应缓存、推理强度、API 和可观测性设置）是类型化的 Operator 适配器，Operator 会把它们翻译为 canonical 的 `providers` 和 `global` 节。
- **被拒绝的字段。** `spec.config.routing` 以及已移除功能的字段（例如 `classifier` 和 `prompt_guard`）会被拒绝。

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
    recipes:
      - name: default
        strategy: priority
```

不要将任意 `providers` 或 `global` 键直接复制到 `spec.config` 下；它们不是 CRD 字段。参见 [Kubernetes Operator](k8s/operator) 和 [SemanticRouter CRD 参考](../api/semantic-router-crd)。

## 路由 DSL

DSL 是用于模型事实、信号、投影、决策、算法、插件、入口和配方的聚焦编写表面。Providers、listeners、凭据和全局服务仍由 YAML 拥有。

当路由策略受益于紧凑、可复核的表示时，使用 DSL。将 canonical YAML 作为完整的部署产物。

## 避免分割所有权

- 不要将生成的运行时配置当作源文档来编辑。
- 不要让 GitOps 和交互式控制面板会话在没有显式交接的情况下写入同一部署。
- 不要将密钥放入 DSL、ConfigMap 或已提交的 YAML。
- 不要假设已评估的路由能证明后端就绪；请验证生成。
- 在将完整变更应用到实时栈之前，先预览并校验。

管理端点和并发契约见[管理 API 参考](../api/apiserver)。

---
title: SemanticRouter 自定义资源参考
sidebar_label: SemanticRouter 自定义资源
description: vllm.ai/v1alpha1 SemanticRouter 自定义资源的顶层字段指南。
translation:
  source_commit: "867155c924b6527d6a412e1412ce712a9e5cc9b8"
  source_file: "docs/api/semantic-router-crd.md"
  outdated: false
---

# SemanticRouter 自定义资源参考 {#semanticrouter-crd-reference}

`SemanticRouter` 是 Operator 拥有的资源，用于部署 Router 并将其绑定到 Kubernetes 模型服务。

```yaml
apiVersion: vllm.ai/v1alpha1
kind: SemanticRouter
metadata:
  name: my-router
spec: {}
```

本页概括顶层契约。已安装的 CRD 对嵌套 OpenAPI 校验和默认值具有权威性：

```bash
kubectl explain semanticrouter.spec --recursive
```

源 schema 见
[`semanticrouter_types.go`](https://github.com/vllm-project/semantic-router/blob/main/deploy/operator/api/v1alpha1/semanticrouter_types.go)，
生成的 CRD 见
[`vllm.ai_semanticrouters.yaml`](https://github.com/vllm-project/semantic-router/blob/main/deploy/operator/config/crd/bases/vllm.ai_semanticrouters.yaml)。

## 顶层 `spec` 字段 {#top-level-spec-fields}

| 字段 | 用途 |
|-------|---------|
| `image` | Router 镜像仓库、标签、注册表前缀和拉取策略。 |
| `replicas` | 未由自动扩缩控制副本时的固定副本数。 |
| `imagePullSecrets` | 按名称引用的注册表凭据。 |
| `serviceAccount` | 创建或选择工作负载 ServiceAccount。 |
| `service` | Service 类型以及 API、gRPC 和指标端口。 |
| `resources` | 容器 CPU、内存及其他资源请求/限制。 |
| `persistence` | 模型存储 PVC 设置或已有 claim。 |
| `config` | Operator 配置适配器以及规范 `routing` 对象。 |
| `toolsDb` | 为 Router 工具选择物化的函数工具记录。 |
| `vllmEndpoints` | 作为规范 provider 和 Model Card 发现的模型服务。 |
| `autoscaling` | HPA 启用以及 CPU/内存目标。 |
| `startupProbe`、`livenessProbe`、`readinessProbe` | 工作负载探针调优。 |
| `securityContext`、`podSecurityContext` | 容器和 Pod 安全设置。 |
| `podAnnotations` | 附加注解，包括抓取器集成。 |
| `nodeSelector`、`tolerations`、`affinity` | Pod 调度约束。 |
| `env`、`args` | 额外的 Router 环境变量和参数。 |
| `gateway` | 对已有 Kubernetes Gateway 的引用。 |
| `openshift` | OpenShift Route 行为。 |
| `ingress` | Kubernetes Ingress 配置。 |

## `vllmEndpoints` {#vllmendpoints}

每个条目创建一个逻辑 provider 模型和一个 backend 引用：

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
      weight: 1
```

支持的 backend 类型为 `service`、`kserve` 和 `llamastack`。可选的 `loras` 声明由生成的路由 Model Card 暴露的 adapter。第一个解析到的模型成为默认模型，除非配置覆盖它。

## `config` {#config}

Operator 始终渲染规范的 v0.3 Router 文档。其配置表面分为两部分：

- `config.routing` 透传规范 routing 对象，包括 Model Card、信号、投影、决策、算法和路由插件；
- 类型化适配器字段如 `response_cache`、`tools`、`prompt_guard`、`classifier`、`complexity_rules`、`reasoning_effort`、`api` 和 `observability` 会被转换到规范的 provider 或 `global` 位置。

不要假设本地 `config.yaml` 中的任意键可以直接放在 `spec.config` 下。在本地 YAML 与 Operator 之间迁移时，使用 CRD schema 和[配置工作流](../installation/configuration-workflows)。

已弃用的 `semantic_cache` 适配器为兼容性保留；`response_cache` 是规范字段。不要同时设置两者。

## 状态 {#status}

`status` 报告观察到的 generation、副本数、conditions、phase、网关模式以及检测到的 OpenShift 特性。控制器和自动化应使用 conditions 和 `status.observedGeneration`，而不仅仅是 phase。

```bash
kubectl get semanticrouter <name> -o jsonpath='{.status.conditions}'
kubectl get semanticrouter <name> -o jsonpath='{.status.observedGeneration}'
```

## 相关指南 {#related-guides}

- [使用 Kubernetes Operator 部署](../installation/k8s/operator)
- [运维 Operator 部署](../installation/k8s/operator-operations)
- [配置](../installation/configuration)

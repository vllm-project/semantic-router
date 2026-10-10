---
title: 后端目标兼容性
description: 查看哪些后端目标形式能在 Docker、Helm、Operator、控制面板和配方工作流中保留。
translation:
  source_commit: "867155c924b6527d6a412e1412ce712a9e5cc9b8"
  source_file: "docs/installation/backend-target-compatibility.md"
  outdated: false
---

# 后端目标兼容性

Semantic Router 使用 `providers.models[].backend_refs[]` 作为逻辑模型名称与一个或多个物理推理目标之间的 canonical 契约。

此矩阵覆盖**配置的生成与保留**。它并不声称端点可到达、健康，或与特定模型协议兼容。

## 选择正确的矩阵

| 问题 | 事实来源 |
| --- | --- |
| 哪些客户端端点和后端线协议可以一起工作？ | [协议兼容性](protocol-compatibility) |
| 哪些目标字段能在 CLI、Helm、Operator、控制面板和配方工作流中保留？ | 本页 |
| 项目维护哪些部署栈和集成？ | [部署支持](support-matrix) |

同一份 canonical 文档可以在 CLI、Helm 和控制面板之间流转。Operator 则接受 Kubernetes 发现输入，并将其支持的子集翻译为 canonical 后端引用。

## 状态含义

- **Supported**：该表面接受并保留 canonical 目标形式。
- **Adapter**：该表面接受更窄的原生输入，并生成 canonical 后端引用。
- **Partial**：该形式被接受，但存在下文所述限制。
- **Not expressible**：生成方无法表示该目标形式。

## 兼容性矩阵

<!-- BEGIN BACKEND TARGET COMPATIBILITY MATRIX -->

| 目标形式 | Canonical YAML | Docker / CLI | Helm | Operator | 控制面板 | 维护中的配方 |
| --- | --- | --- | --- | --- | --- | --- |
| 作为 `host[:port]` 的直接 `endpoint` | Supported | Supported | Supported | Adapter | Supported | Supported |
| 包含路径的 HTTP(S) `base_url` | Supported | Supported | Supported | Not expressible | Supported | Supported |
| 多个带权重的 ref，并共享路由元数据 | Supported | Supported | Supported | Adapter | Supported | Supported |
| Provider、API 版本、路径和标头元数据 | Supported | Supported | Supported | Not expressible | Supported | Supported |
| Kubernetes Service DNS 目标 | Supported | Supported | Supported | Adapter | Supported | Supported |
| KServe 发现 | Not expressible | Not expressible | Not expressible | Partial | Not expressible | Not expressible |
| 按标签选择的 Service 发现 | Not expressible | Not expressible | Not expressible | Adapter | Not expressible | Not expressible |
| 每个带权重 ref 使用不同路径或请求标头 | Supported | Partial | Supported | Not expressible | Supported | Partial |

<!-- END BACKEND TARGET COMPATIBILITY MATRIX -->

Docker / CLI 路径为逻辑模型生成一条路由。当模型有多个带权重的 ref 时，请求标头、Host 重写和 TLS SNI 来自第一个 ref。仅当每个 ref 使用相同路径时才应用路径前缀；否则本地生成器会省略路径重写。请保持这些路由级属性在模型的 ref 之间兼容。使用实时推理遥测进行端点或副本选择，是 [#2332](https://github.com/vllm-project/semantic-router/issues/2332) 跟踪的单独数据平面契约。

## 可移植的目标形式

对直接主机和可选端口使用 `endpoint`：

```yaml
providers:
  models:
    - name: local/general
      provider_model_id: Qwen/Qwen3-8B
      api_format: openai
      backend_refs:
        - name: primary
          endpoint: model-server.default.svc.cluster.local:8000
          protocol: http
          provider: vllm
          weight: 100
```

当上游身份包含 scheme 或路径时，使用 `base_url`。将凭据保留在环境引用中，而不是提交到 YAML：

```yaml
providers:
  models:
    - name: hosted/reasoning
      provider_model_id: provider/model-id
      api_format: openai
      backend_refs:
        - name: hosted-primary
          base_url: https://provider.example/v1
          provider: openai
          auth_header: Authorization
          auth_prefix: Bearer
          api_key_env: PROVIDER_API_KEY
          extra_headers:
            X-Tenant: production
          weight: 100
```

对于可移植配置，不要在 `endpoint` 中放入 URL scheme 或路径。某些本地生成路径会接受这些形式，但并非每个维护中的生成方都对其含义达成一致。`base_url` 是 canonical URL 形式。

`api_format` 属于模型，而不属于单个后端 ref。它选择后端请求和响应编解码器；后端 ref 上的 `protocol` 选择 HTTP 传输。在选择 `api_format` 之前，参见[协议兼容性](protocol-compatibility)。

## 生成方行为

| 生成方 | 行为和边界 |
| --- | --- |
| Docker / 本地 CLI | 将 ref 翻译为 Envoy cluster 和路由。它保留主机、端口、HTTP 或 HTTPS、权重、共享路径前缀、由环境解析的授权，以及共享的额外标头。被引用的模型服务器必须已经可到达。 |
| Helm | `configOverride` 渲染一份完整的 canonical 映射，而不合并示例 provider 默认值。可达性和模型兼容性仍是运行时检查。 |
| Operator | `spec.vllmEndpoints[]` 是 Kubernetes 发现适配器，而不是完整 provider schema 的副本。它发出下文所述的受支持 canonical 子集。 |
| 控制面板 | 读写受支持的 canonical 后端清单，包括 provider 身份、URL、认证元数据、API 版本、chat 路径、额外标头和环境键引用。 |

Operator 当前发现：

- 命名的 Kubernetes Service；
- KServe `InferenceService`；以及
- 按标签选择的 Llama Stack Service。

这些适配器生成后端名称、端点、协议和权重。将完整的外部 provider 目标放入通过 Helm 或其他 canonical 配置工作流提供的 canonical 配置中。更广泛的 CRD 与 Helm 对等属于 [#2355](https://github.com/vllm-project/semantic-router/issues/2355)。

KServe 发现当前假设常规的 `<InferenceService>-predictor.<namespace>.svc.cluster.local:8443` HTTPS 目标。它不会解析 `status.url` 或检查生成的 Service，因此自定义或命名 predictor 的服务布局需要显式的 Service 后端。

跨生成方的未知字段和受支持版本对等由 [#2469](https://github.com/vllm-project/semantic-router/issues/2469) 跟踪。在该工作落地之前，不要依赖未知扩展在生成方之间迁移后仍然保留。

## 校验边界

仓库在五个层面演练此矩阵：

- CLI 测试校验生成的 Envoy 配置中的直接目标、URL 路径、TLS、权重和标头。
- Helm 校验渲染完整的 canonical 覆盖，并检查其后端字段能保留且不会泄漏示例默认值。
- Operator 测试校验 Service 发现和生成的 canonical ref。
- 控制面板前端和后端测试校验受支持的字段清单以及持久化的 canonical 输出。
- 维护中配置契约解析其枚举的资产清单和每个配方。单独的参考配置契约校验随附的参考配置，其中演练带权重的直接目标和丰富 URL 目标。

这些检查证明配置翻译和保留。它们不能替代直接的后端协议检查，或通过 Router 的实时请求。

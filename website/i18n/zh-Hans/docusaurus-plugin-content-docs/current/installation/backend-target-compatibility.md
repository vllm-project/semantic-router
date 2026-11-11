---
is_mtpe: false
title: 后端目标兼容性
description: 查看哪些后端目标形式能在 Docker、Helm、Operator、控制面板和配方工作流中保留。
translation:
  source_commit: "9156d5bc1ed9edff626b95a2b8260a77cb1712c5"
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

URL 那一行是完整 URL 的目标形式：在 `endpoint` 中写完整 URL（[迁移模型资源](model-resource-migration.md#request-backends)展示了早期写法）。

默认 standalone 前端直接使用后端池；`--gateway extproc` 则为每个逻辑模型生成一条 Envoy 路由。在 Envoy 路径中，请求标头、Host 重写和 TLS SNI 来自第一个 ref；只有所有 ref 的路径相同时才应用路径前缀。为保持部署方式之间的可移植性，请为同一逻辑模型使用兼容的传输元数据。这里的 Chat 后端池与[模型运行时副本池](../overview/component-architecture#scale-a-deployment-through-replicas)不同；基于 Chat 后端实时推理遥测的选择由 [#2332](https://github.com/vllm-project/semantic-router/issues/2332) 跟踪。

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

当上游身份包含 scheme 或路径时，在 `endpoint` 中写完整 URL。将凭据保留在环境引用中，而不是提交到 YAML：

```yaml
providers:
  models:
    - name: hosted/reasoning
      provider_model_id: provider/model-id
      api_format: openai
      backend_refs:
        - name: hosted-primary
          endpoint: https://provider.example.com/v1
          provider: openai
          auth: {header: Authorization, prefix: Bearer, api_key_env: PROVIDER_API_KEY}
          headers:
            X-Tenant: production
          weight: 100
```

`host[:port]` 形式的 `endpoint` 从 `protocol` 取得 scheme；完整 URL 自带 scheme 和路径，其路径就是完整的 API 根。`auth`、`headers` 和 `tls` 与部署的[连接块](model-catalog.md#connection-block)形式相同；后端从 `auth.api_key_env` 读取密钥，并且总是被验证（见[信任私有 CA](#trust-a-private-ca)）。

`api_format` 属于模型，而不属于单个后端 ref。它选择后端请求和响应编解码器；后端 ref 上的 `protocol` 选择 HTTP 传输。在选择 `api_format` 之前，参见[协议兼容性](protocol-compatibility)。

## 信任私有 CA {#trust-a-private-ca}

https 后端按系统根证书、以其端点的主机名验证。证书来自私有 CA 的后端，需要在它的 `tls` 块中指定该 CA，或在 `providers.defaults.tls` 中为所有 https 后端统一指定：

```yaml
providers:
  defaults:
    tls:
      ca_file: /etc/vllm-sr/certs/example-corp-ca.pem
  models:
    - name: private-llm
      provider_model_id: private-llm
      backend_refs:
        - name: internal
          endpoint: https://llm.internal.example/v1
          provider: openai-compatible
          tls:
            ca_file: /etc/vllm-sr/certs/example-corp-ca.pem
            server_name: llm.internal.example
```

| 字段 | 含义 |
| --- | --- |
| `ca_file` | 该后端证书必须链接到的 PEM 证书包。它替换该后端的系统根证书，与 Envoy 的 `trusted_ca` 相同。 |
| `server_name` | 作为 SNI 发送、证书必须包含的名称。默认为端点的主机名。 |
| `cert_file`、`key_file` | 后端要求客户端证书时出示的证书，二者须同时设置。 |
| `providers.defaults.tls.ca_file` | 自身 `tls` 未指定 `ca_file` 的每个 https 后端所用的 `ca_file`。 |

生成的 Envoy cluster 与 Router 自己对该后端的调用使用相同设置，因此二者信任相同的 CA。Router 自己的调用包括回退尝试、影子副本、上下文窗口探测，以及 standalone 模式下的每个请求。

- **证书包中要包含中间证书。**服务器只发送叶证书时，只有证书包同时包含签发它的中间证书和根证书，验证才能通过。只有根证书时握手失败：OpenSSL 报 `unable to verify the first certificate`（代码 21），Go 报 `certificate signed by unknown authority`。检查证书包：

  ```bash
  openssl s_client -connect llm.internal.example:443 \
    -servername llm.internal.example \
    -CAfile /etc/vllm-sr/certs/example-corp-ca.pem </dev/null | grep 'Verify return code'
  ```

  `Verify return code: 0 (ok)` 表示证书包完整。
- **CA 文件替换系统根证书。**指定了 `ca_file` 的后端不再信任系统根证书。默认值会替换所有没有自己 `ca_file` 的 https 后端的系统根证书，公共 provider 也包括在内。要继续信任它们，把系统证书包和你的 CA 放进同一个文件：`cat /etc/ssl/certs/ca-certificates.crt example-corp-ca.pem > /etc/vllm-sr/certs/ca-bundle.pem`。
- **校验。**路径须为绝对路径。纯 HTTP 后端上的 `tls`、缺少私钥的证书以及 `insecure_skip_verify` 都会被拒绝：chat 后端总是被验证。存在的文件必须是可读的 PEM。在校验配置的位置不存在的文件（例如控制面板的容器中）会在读取它的地方失败：Envoy 启动时、standalone Router 加载配置时，或 Router 发起调用时。
- **每个池一个 `tls`。**同一模型别名的后端 ref 共享一个 Envoy cluster，因此必须共享 `tls` 设置。
- **替换文件。**Envoy 在启动时读取这些文件，Router 在加载配置时读取。原地替换文件后请重启整个栈。

### 挂载文件 {#mount-the-files}

- **`vllm-sr serve`** 将每个文件以只读方式、按相同路径挂载到 Router 容器中，使用 `--gateway extproc` 时也挂载到 Envoy 容器中。文件缺失时拒绝启动。请把证书包放在单独的文件中（例如 `/etc/vllm-sr/certs/` 下），不要直接指向主机的系统证书包：在 SELinux 主机上，挂载会重新标记该文件。添加文件后请重新运行 `vllm-sr serve`。
- **Helm。**把证书包放进 Secret 或 ConfigMap，并用 `extraVolumes` 和 `extraVolumeMounts` 挂载到 `ca_file` 指定的路径：

  ```yaml
  extraVolumes:
    - name: backend-ca
      secret:
        secretName: example-corp-ca
  extraVolumeMounts:
    - name: backend-ca
      mountPath: /etc/vllm-sr/certs
      readOnly: true
  ```

  使用 `gateway.mode: extproc` 时，由你运行的网关通过它自己的后端 TLS 设置验证后端；Router 的 `tls` 覆盖 Router 自己的调用。
- **Operator。**`SemanticRouter` 资源目前没有额外挂载的字段。要访问私有 CA 之后的后端，请使用 Helm 部署。

## 生成方行为

| 生成方 | 行为和边界 |
| --- | --- |
| Docker / 本地 CLI | standalone 前端直接使用后端池；`--gateway extproc` 将 ref 翻译为 Envoy cluster 和路由。配置保留主机、端口、HTTP 或 HTTPS、权重、路径和请求标头；上文说明 Envoy 的共享元数据限制。后端模型服务器必须已经可达。 |
| Helm | `configOverride` 渲染一份完整的 canonical 映射，而不合并示例 provider 默认值。可达性和模型兼容性仍是运行时检查。 |
| Operator | `spec.vllmEndpoints[]` 是 Kubernetes 发现适配器，而不是完整 provider schema 的副本。它发出下文所述的受支持 canonical 子集。 |
| 控制面板 | 读写受支持的 canonical 后端清单，包括 provider 身份、URL、认证元数据、API 版本、chat 路径、请求头和环境键引用。 |

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

---
title: Gateway 模式
description: 选择客户端流量从哪里进入 Router：standalone 直接服务，或放在基于 Envoy 的网关之后；适用于 docker 和 kubernetes 两种目标。
translation:
  source_commit: "9156d5bc1ed9edff626b95a2b8260a77cb1712c5"
  source_file: "docs/installation/gateway-modes.md"
  outdated: false
---

# Gateway 模式

**gateway 模式**决定客户端流量从哪里进入，**目标**决定整套服务在哪里运行。它们与 `--engine` 是不同的选择：`--engine` 提供原生判断请求，不启用 Chat 路由；省略时启动 Router 模式。共用的前端与模型运行时见[组件架构](../overview/component-architecture)。

| 模式 | 客户端流量进入 | 适用场景 |
| --- | --- | --- |
| `standalone`（默认） | Router 本身，它在配置的 `listeners` 上提供 OpenAI 兼容接口 | 单机、开发、边缘、大多数自托管 |
| `extproc` | Router 前面的基于 Envoy 的网关，Router 提供 ext_proc | 需要限流、mTLS、高级路由匹配等 Envoy 能力，或接入你已有的网关 |

| 目标 | `standalone` | `extproc` |
| --- | --- | --- |
| `docker`（默认） | Router 容器服务 listener，没有 Envoy 容器 | CLI 启动的 Envoy 容器位于 Router 之前，与以前的版本一致 |
| `kubernetes` | Router Pod 服务 listener，由 Service 暴露 | Router 为你的 Envoy Gateway、Envoy AI Gateway、Istio 或 KServe 提供 ext_proc |

```bash
vllm-sr serve                                      # standalone，docker
vllm-sr serve --gateway extproc                    # Envoy 在前，docker
vllm-sr serve --target kubernetes --config config.yaml
```

除非你需要上面列出的 Envoy 功能，否则先用 `standalone`。在 docker 目标上，切换模式就是用当前生效的配置重启：运行 `vllm-sr serve --gateway extproc`，再运行普通的 `vllm-sr serve` 即可切回。standalone 模式下没有 Envoy 容器，所以 `vllm-sr logs envoy` 和 `x-envoy-*` 响应头只属于 `extproc`。

两种模式运行同一个路由核心，因此同一个请求在两种模式下得到相同的决策、相同的上游请求和相同的响应变换。
[设计文档](../proposals/standalone-mode#与-envoy-模式的差异)列出了少数有意为之的差异，例如只有 Envoy 才会添加的
`x-envoy-*` header。

:::note 升级
以前的版本总是在 Router 前面放 Envoy。现在默认是 standalone；`vllm-sr serve --gateway extproc` 会恢复 Envoy 入口。
参见[发布说明](../release-notes/standalone-mode)。
:::

## standalone 模式下 Router 做什么

- **listener：** `listeners` 中的每一项都以 HTTP/1.1 和 HTTP/2（明文下为 h2c）提供服务，`timeout` 作为空闲超时，
  请求体上限 500 MiB，最多 50,000 个连接，与 Envoy 模板的限制一致。
- **API key：** 设置 `api_keys` 后，客户端需以 `Authorization: Bearer <key>` 或 `api-key: <key>` 发送其中之一；
  其他请求得到 OpenAI 风格的 401。key 在请求到达 provider 之前被移除。
- **模型白名单：** 设置 `models` 后，该 listener 只接受这些请求模型（见[模型白名单](#model-allow-list)）。
- **TLS：** `tls` 让该 listener 以 TLS 1.2 及以上提供服务，通过 ALPN 协商 HTTP/2 或 HTTP/1.1。相对路径相对于配置
  文件所在目录。证书文件变化时 Router 会重新加载密钥对，因此续期后的证书（轮换的 Kubernetes Secret、cert-manager）
  无需重启即可用于新连接；加载失败时继续使用之前的密钥对。`--gateway extproc` 不提供该能力。

  ```yaml
  listeners:
    - name: https-8443
      address: 0.0.0.0
      port: 8443
      tls:
        cert_file: certs/tls.crt
        key_file: certs/tls.key
  ```

- **边缘的信任边界：** 客户端发送的身份 header（`x-authz-*`，以及 `global.services.authz.identity` 指定的名称，
  除非 listener 信任它们，见[身份 header](#identity-headers)）和只有可信代理才能设置的代理控制 header 都不会进入路由
  或到达后端。后者包括 `x-envoy-internal` 以及 Envoy 的重试、超时
  和追踪控制，例如 `x-envoy-max-retries`、`x-envoy-retry-on` 和 `x-envoy-upstream-rq-timeout-ms`。Router 之后的基于
  Envoy 的组件（sidecar、模型服务前的网关）会听从可信调用方发来的这些 header，而 Router 正是这样的调用方，否则客户端就能
  借道设置那里的重试和超时；剥离它们能让 Router 的可靠性策略始终是唯一的重试和超时权威。这份清单就是 Envoy 从外部请求中
  剥离的那一份，并保持这么窄。
- **探针：** 进程运行时 `GET /health` 就会应答；路由核心可以接收流量后 `GET /ready` 才应答。Prometheus 指标仍在
  Router 的指标端口（9190）。
- **重新加载：** Router 原地重新加载配置。修改 listener 的地址、端口、超时或 `tls` 路径，或者新增、删除 listener，
  都会以 `restart_required` 被拒绝，直到 Router 重启；API key 和模型白名单可原地重新加载。

### 模型白名单 {#model-allow-list}

listener 的 `models` 列出它接受的全部请求 `model` 值。可以用它让公开 key 只能访问 Router 的自动模型，而内部
listener 仍可使用所有模型：

```yaml
listeners:
  - name: dashboard-internal   # 第一个 listener：Dashboard Playground 使用它
    address: 127.0.0.1
    port: 8898
  - name: public
    address: 0.0.0.0
    port: 8899
    api_keys: ["${WORKSHOP_KEY}"]
    models: [vllm-sr/auto]
```

- 名称精确匹配（区分大小写，请求值去除首尾空白后比较），不展开别名：请列出客户端可能发送的每个名称。为空或不设置时，
  listener 接受所有模型。
- 检查在 API key 检查之后、任何 signal、缓存或 decision 之前进行，使用的是 Router 为路由解析出的同一个模型。其他模型
  （包括原本会直通的 provider 模型）得到 `403`，错误码 `model_not_allowed`，并以客户端协议返回。没有模型的请求得到
  `400 model_required`。
- 该 listener 上的 `GET /v1/models` 只列出目录中存在的允许名称。
- decision 在进程内发起的模型调用（Looper 和 request-graph hop）不是客户端请求，不受限制，因此 `vllm-sr/auto`
  仍可到达其 decision 指定的所有 provider 模型。
- 即使开启了 `global.router.skip_processing.enabled`，该 listener 也会忽略 `x-vsr-skip-processing`，因为跳过的请求
  会绕过检查。
- `--gateway extproc` 会以 unsupported 拒绝带 `models` 的 listener：CLI 生成的 Envoy listener 目前还不执行它。

### 身份 header {#identity-headers}

默认情况下，standalone listener 会丢弃客户端发送的身份 header，因为 Router 前面没有任何组件认证过它：任何人都可以冒充任何
用户。在认证代理之后，或者对自行设置用户的可信应用服务器，可以让 listener 保留这些 header：

```yaml
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
    identity:
      trust_headers: true
      # 可选：只在来自代理所在网络的连接上保留它们。
      trusted_peers: ["10.0.0.0/8"]
```

- `trust_headers` 保留 `x-authz-*` header 以及 `global.services.authz.identity` 指定的名称。
- `trusted_peers`（CIDR）只在连接的对端地址属于其中某个网络时才保留它们；Router 不会为此读取 `X-Forwarded-For`。
  为空时，信任 header 的 listener 信任所有对端。
- 每个 listener 只为自己的请求做决定，重新加载会应用修改。在不信任身份的 listener 上，请求是匿名的。

Memory、router replay 和按用户学习的选择算法（`gmtrouter`、`rl_driven`）会记录每个请求的用户。没有信任身份的 listener
时，它们仍会加载，并把每个请求视为匿名请求；Router 会在启动时记录一条点名这些功能的警告。

### 哪些需要 `extproc`

按客户端身份执行访问控制的策略需要身份来源。当没有任何 listener 设置 `identity.trust_headers` 时，Router 会在启动和重新加载
时拒绝这类配置，错误信息会同时指向该选项和 `--gateway extproc`：

- 基于 `authz`（角色绑定）信号的决策；
- 匹配 `user` 或 `group` 的限流规则；
- 解析按用户 API key 的 `global.services.authz.providers`。

请在认证代理之后的 listener 上打开 `identity.trust_headers`，或者在能认证客户端的网关之后用 `--gateway extproc` 运行它们。令牌桶限流、mTLS、JWT 或 OIDC 以及高级路由匹配已计划在
standalone 模式中支持；在那之前它们同样需要 Envoy。

## 目标与平台

`--platform auto` 自动检测部署目标的执行后端。显式选择 `cpu`、`rocm` 或
`cuda` 时，分别使用 `vllm-sr`、`vllm-sr-rocm` 或 `vllm-sr-cuda` 镜像。

- **docker：** `rocm` 透传 ROCm 设备，`cuda` 透传 NVIDIA GPU；使用 `--device-ids` 为默认模型部署选择宿主机 GPU。
- **kubernetes：** 生成的 Helm values 会设置 `gateway.mode`、镜像仓库，以及模型放置所需的 GPU 资源
  （`amd.com/gpu` 或 `nvidia.com/gpu`）。集群需要相应的设备插件。在 YAML 中使用 `rocm:0` 等分配序号；
  宿主机 `--device-ids` 仅用于 Docker。`--target k8s` 是 `--target kubernetes` 的旧名，仅在本版本中继续可用。

不使用 CLI 时，Helm chart 读取同一个值 `gateway.mode`（默认 `standalone`）；Operator 默认以 standalone 模式运行 Router，只有当 `spec.gateway` 指定一个 Gateway 时才选择 extproc。迁移由基于 Envoy 的网关调用的现有发行版，请参阅[升级与回滚](upgrade-rollback)。

### macOS

在 macOS 上 docker 目标只使用 CPU：内置模型在 arm64 镜像中以 CPU 运行，因为 Apple 的虚拟化不向 Docker 的 Linux
虚拟机提供 Metal 或 GPU 计算。在那里使用 `--platform rocm` 或 `--platform cuda` 会以明确的错误信息失败。通过宿主机使用
GPU 的支持见 [#4636](https://github.com/vllm-project/semantic-router/issues/4636)。

- **在 CPU 上运行轻松：** 307M 的 Vela 任务模型（领域、PII、越狱与安全防护、embedding、重排）、Vela Omni Nano，以及
  Decision 2.0 的 Kai-0.6B 和 Eos-0.8B 决策模型。
- **更大的模型：** 在 CPU 上每个参数约需 4 字节，因此 2B 的决策模型约需 8 GB。请把 Docker 虚拟机的内存（Docker
  Desktop：Settings、Resources）调到高于配置中各模型所需，或选用更小的模型。参见[选择模型](../model-runtime/choose-a-model)。

## `vllm-sr serve` 的选项

`vllm-sr serve [MODEL]` 每次默认启动 Router；`--engine`（`-e`）启动 Engine。
两者使用同一前端和模型池，保留已保存的路由配置。Dashboard 只展示启动模式，
不提供运行时切换入口。MODEL 只覆盖默认判断 deployment 的 artifact；未填写的
profile 和副本位置保留原值，全新配置默认 Vela 2.0 0.3B。

| 范围 | 选项 |
| --- | --- |
| 模型与运行 | `MODEL`、`--engine`、`--revision`、`--runtime-profile`、`--data-parallel-size` / `-dp` |
| 平台 | `--platform auto/cpu/cuda/rocm`，auto 检查实际 Docker 主机或目标 Kubernetes 集群 |
| Docker 设备 | `--device-ids` 使用主机编号，按已有可见掩码映射；一个编号允许同卡多副本，N 个编号对应 N 个副本 |
| 配置 | `--config`、`--target`、`--gateway`、`--minimal`、`--readonly`、`--algorithm` |
| Docker 部署 | `--image-pull-policy`、`--container-runtime`、`--router-image`、`--envoy-image`、`--dashboard-image`、`--startup-timeout`、`--replace-active-config`、`--recipe-env` |
| Kubernetes 部署 | `--namespace`、`--context`、`--profile`、`--chart-dir` |

Kubernetes 使用 pod 分配后的设备序号，通过 canonical replicas 请求 GPU 资源，
不接受物理主机 `--device-ids`。模型 `--runtime-profile` 与部署 `--profile` 独立。
`--revision` 可填分支、tag 或 commit，启动前只解析一次并记录不可变 SHA；内置模型
省略时使用发布 pin。监听地址、端口和公开模型授权始终通过 config 控制。
完整参数以 [CLI 参考](/docs/api/cli) 为准。

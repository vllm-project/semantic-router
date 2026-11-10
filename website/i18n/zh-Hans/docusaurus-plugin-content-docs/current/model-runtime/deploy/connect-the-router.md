---
title: 把 Router 连接到模型运行时
sidebar_label: 连接 Router
sidebar_position: 1
description: 声明 model_runtime 部署，让 Router 启动并监管它们的 worker，或附加到你自己运行的运行时，把信号绑定到它们，并发布原生 API。
is_mtpe: false
translation:
  source_commit: "bc5c6bfb747ffe9c5f35d98b203427dada8c4c56"
  source_file: "docs/model-runtime/deploy/connect-the-router.md"
  outdated: false
---

import Tabs from '@theme/Tabs'
import TabItem from '@theme/TabItem'

# 把 Router 连接到模型运行时 {#connect-the-router-to-the-model-runtime}

每个需要模型的功能都从一个**部署**获得模型：`global.model_catalog.deployments` 下的一个命名模型。`model_runtime` 部署要么运行在由 Router 启动和监管的 worker 中，要么附加到你自己运行的运行时。无论 worker 运行在哪里，绑定和公共模型名称都通过部署来标识。

<Tabs groupId="runtime-owner">
<TabItem value="managed" label="由 Router 管理" default>

```yaml title="config.yaml"
global:
  model_catalog:
    deployments:
      decision-edge:
        provider: model_runtime
        artifact: vllm-sr/d3-edge
        device: auto
```

没有 `endpoint` 时，Router 为该部署启动一个 worker，并在它退出时重启它。

</TabItem>
<TabItem value="attached" label="附加到你运行的运行时">

```yaml title="config.yaml"
global:
  model_catalog:
    deployments:
      decision-shared:
        provider: model_runtime
        artifact: vllm-sr/d3-edge
        endpoint: http://decision-runtime.example.com:8100
        model: d3-edge
```

有 `endpoint` 时，Router 附加到一个它不负责启动、重启或停止的运行时；参见[附加你自己运行的运行时](#attach-to-a-runtime-you-run)。

</TabItem>
</Tabs>

:::tip 你可能完全不需要这些
内置功能已经知道自己需要哪个模型。只有当你想把模型放到 GPU 上、固定另一个模型、放置或扩展其副本，或附加一个共享 worker 时，才需要自己配置部署。
:::

## Engine 模式与 Router 模式 {#engine-and-router-modes}

前端拥有 API 监听器、认证、配置和模型资源。Router 和 Engine 是同一个前端的两种能力：

- **Engine** 在没有 Chat 后端、也没有用户编写的路由配置的情况下，提供原生 System One 问答。
- **Router** 还会运行信号、决策、算法和 Chat 路由。

```bash
vllm-sr serve vllm-sr/d3-edge -e
vllm-sr serve --config config.yaml
```

- 启动时会设置 `global.router.enabled`：每次不带 `-e` 的 `serve` 调用都启动 Router 模式；`-e` 启动 Engine 模式，并保留已保存的路由配置，但不初始化其信号或 Chat 后端。
- Dashboard 显示这个启动模式，并独立管理模型部署。它不会切换模式。
- 原生 API 在两种模式下都可用。监听器密钥和原生模型授权仍然生效。
- 在 docker 目标上，除非 `--dashboard`、`--observability` 或 `--no-minimal` 要求更多，Engine 模式只启动 Router 容器。

## 内置功能无需配置 {#built-in-features-need-no-configuration}

当你开启一个内置功能，例如 `domain` 信号或语义缓存时，Router 已经知道它需要哪个模型。判断任务共享决策模型；嵌入使用自己的部署（[为信号选择模型](../../installation/choose-a-model-for-a-signal.md)）。

哪些 worker 会启动：

- 已启用的信号或功能所解析到的部署，或已发布的原生模型授权所指定的部署，会被加载。
- Engine 模式还会加载决策模型。在 Router 模式下，只有当有内容解析到决策模型时才加载它。
- 其他声明不会启动 worker。

要选择 CPU 或 GPU 放置，请按下文以及[放置与扩展 worker](./placement-and-scaling.md)中的说明配置部署的 `device` 或 `replicas`。

## 描述一个部署 {#describe-a-deployment}

编写部署来选择模型、其引擎参数和 worker 放置：

```yaml title="config.yaml"
global:
  model_catalog:
    deployments:
      vela-domain:
        provider: model_runtime
        artifact: vllm-sr/Vela-1.0-Encoder-307M-Domain
        device: cpu
        input:
          max_tokens: 512
          overflow: reject
      decision-edge:
        provider: model_runtime
        artifact: vllm-sr/d3-edge
        device: auto
```

| 字段 | 含义 |
| --- | --- |
| `provider` | 对运行时提供的模型，始终为 `model_runtime`。 |
| `artifact` | Hugging Face 仓库，或本地副本的绝对路径。 |
| `revision` | 要加载的确切 40 字符提交。内置模型已经固定；其他仓库需要提供。 |
| `device` | `auto`（默认）、`cpu`、`cuda:N`、`rocm:N`、`xpu:N`、`npu:N`、`mps`，或插件添加的加速器。 |
| `max_num_seqs`、`max_num_batched_tokens`、`batch_window_ms`、`dtype`、`enable_shared_context`、`max_model_len` | 引擎参数：worker 如何合并请求、计算使用的精度，以及 Decision 模型的输入上限。省略它们则使用运行时的默认值。`max_model_len` 适用于没有副本的托管 worker。参见[引擎参数](./engine-args.md)。 |
| `input` | 用于任务模型：输入的最大 token 数（`max_tokens`），以及更长输入的处理方式（`overflow`：`reject`、`truncate` 或 `window`）。 |
| `replicas` | 该模型的显式托管 `device` 放置，或附加的 `endpoint`/`model` 放置。只有一个 worker 时省略。 |
| `endpoint` | 附加到你运行的运行时，而不是启动一个。 |
| `model` | 附加运行时上模型的 ID，即其 `GET /v1/models` 列出的 ID（默认：部署名称）。 |

附加的运行时还接受 `auth`、`headers`、`timeout_ms` 和 `tls`；[模型目录参考](../../installation/model-catalog.md#connection-block)列出了每个字段。

### 把信号绑定到部署 {#bind-a-signal-to-a-deployment}

用一个按信号类型作键的**绑定**把信号连接到部署：

```yaml title="config.yaml"
global:
  model_catalog:
    bindings:
      domain: vela-domain
```

- `global.model_catalog.bindings` 下的绑定在所有地方生效。recipe 可以在自己的 `recipes[].model_bindings` 中覆盖它。
- 自己提问的规则（`decision`、`classifiers`、`safety`、`complexity`）改为在规则上指定其部署；参见[为信号选择模型](../../installation/choose-a-model-for-a-signal.md)。
- 信号决定它读取哪种答案：语义判断、标签概率、独立的标签分数、文本 span、向量或相关性分数。绑定只指定部署。

Router 会依据模型的描述检查绑定：head、标签、维度、问题类型和输入上限。已知的不匹配会被拒绝。

:::note
对于发往离线附加运行时的自定义问题，元数据校验会推迟到观察到就绪的模型卡之后；请求不能运行未经检查的任务。参见[就绪](./readiness-and-monitoring.md#when-a-model-is-not-ready)。
:::

## 附加你自己运行的运行时 {#attach-to-a-runtime-you-run}

在 Router 能访问到的任何地方启动一个运行时，然后让部署指向它：

```bash title="On the runtime host"
vllm-srun serve vllm-sr/d3-edge --device rocm:0 --host 0.0.0.0 --port 8100
```

```yaml title="config.yaml"
global:
  model_catalog:
    deployments:
      gpu-edge:
        provider: model_runtime
        artifact: vllm-sr/d3-edge
        endpoint: http://gpu-runtime.example.com:8100
        model: d3-edge
```

- `model` 是运行时在 `GET /v1/models` 中列出的 ID。运行时以去掉组织名的仓库名列出已发布的模型，这里是 `d3-edge`，除非 `--served-model-name` 为它指定了另一个名称。没有 `model` 时，Router 请求的是部署名称；运行时不提供的名称会让部署不可用。
- `endpoint` 接受 `http://host:port`、`https://host:port` 或 Unix 套接字 `unix:///path/to/runtime.sock`。
- Router 不会启动、重启或停止附加的运行时；它检查其健康状况，在它宕机期间把它视为不可用。
- 附加的运行时应与 Router 来自同一个发行版。Router 只接受恰好提供其契约版本的运行时，其他版本会被报告为 `incompatible`，原因中包含两个版本。

:::warning
以这种方式启动的运行时默认监听 `127.0.0.1`，除非你传入 `--host`。它自身没有认证：只在私有网络中暴露它。
:::

在大型主机上，当这样的运行时在可选的 ONNX Runtime 引擎上提供模型（ONNX 包或 Omni 包）时，给它设置 `--threads`，或在 cpuset 中运行它：该模型的每个计算图都运行自己的线程池，最多 `--threads` 个 CPU 线程；没有这个选项时，最多可用进程能运行的所有 CPU（一个 Omni 包有四个计算图）。托管的 CPU worker 从 [CPU 预算](./placement-and-scaling.md#cpu-threads)中获得 `--threads`。

要在集群中让多个 Router 共享 GPU 模型，参见[在 Kubernetes 上运行](./kubernetes.md#share-a-runtime-between-routers)。

## 发布原生 API {#publish-a-native-api}

对于 `vllm-sr/vela3-fast` 的 `primary` 部署，在独立的监听器上添加显式授权，让客户端可以直接向它提问：

```yaml title="config.yaml"
listeners:
  - name: http
    address: 0.0.0.0
    port: 8899
    systemone:
      models: [vllm-sr/vela3-fast]
global:
  model_catalog:
    deployments:
      primary:
        provider: model_runtime
        artifact: vllm-sr/vela3-fast
```

- 用 `GET /v1/systemone/models` 发现已发布的原生模型，然后用那个 `model` ID 发送 `POST /v1/systemone` 或 `POST /v1/decisions`。
- 部署使用其 Hub 制品 ID，除非你设置了 `public_name`；本地制品需要显式的公共名称。
- 当客户端必须认证时，添加监听器的 `api_keys`。
- 原生授权与 Chat 的 `listeners[].models` 允许列表相互独立，在两种启动模式下都有效。替换或扩展模型时，已有授权保持不变；当你打算发布另一个模型时再更新它。

完整的原生请求见[快速开始](../get-started/quickstart.md#3-send-a-request)。发布并不保证就绪；用 `vllm-sr instance --config config.yaml models` 检查。

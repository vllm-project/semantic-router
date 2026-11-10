---
title: 快速开始
sidebar_position: 2
description: 安装模型运行时，运行一个模型，向它发送请求，再让路由器使用它。
is_mtpe: false
translation:
  source_commit: "0811a3cb7de0e30e3dc2282233f573d554691bf1"
  source_file: "docs/model-runtime/get-started/quickstart.md"
  outdated: false
---

import Tabs from '@theme/Tabs'
import TabItem from '@theme/TabItem'

# 快速开始 {#quickstart}

大约十分钟内，你将安装 `vllm-sr` CLI，运行一个决策模型，向它提问，然后让 Router 用同一个模型做路由。

**你需要** Linux、macOS 或 WSL2，以及 Docker 或 Podman、Python 3.10 或更高版本、12 GiB 内存，还要有大约 9 GB 的可用磁盘空间存放 Router 镜像和下载的模型。不需要 GPU。

:::note
Engine 模式（`vllm-sr serve ARTIFACT --engine`）比 0.4.0 发行版更新；参见[发布渠道说明](../../installation/installation.md)。
:::

## 1. 安装 {#1-install}

把包含 Engine 模式的开发渠道安装到安装器的隔离虚拟环境中。`--no-launch` 把启动留到下一步：

```bash
curl -fsSL https://vllm-sr.ai/install.sh | bash -s -- --channel dev --no-launch
```

模型运行时，即 `vllm-srun` Python 包，只随 Router 镜像发布，因此你的机器上不会安装其他东西。CLI 在第一次使用时拉取镜像。镜像带有该发行版自己的 PyTorch 构建，因此其答案就是模型发布时的答案（[选择模型](./choose-a-model.md#hardware)）。

## 2. 运行模型 {#2-serve-a-model}

启动 Vela 3.0 vela3-fast，即 Router 的默认决策模型，它以 FP32 在 CPU 上运行。决策模型回答你用自然语言写的问题。

<Tabs groupId="platform">
<TabItem value="cpu" label="CPU" default>

```bash
vllm-sr serve vllm-sr/vela3-fast --engine --platform cpu
```

在 macOS 上运行时在 CPU 上运行，因为那里的容器无法使用 GPU。

</TabItem>
<TabItem value="rocm" label="AMD GPU">

```bash
vllm-sr serve vllm-sr/vela3-fast --engine --platform rocm --device-ids 0
```

`vllm-sr-rocm` 镜像首次使用时需要下载 6.5 GB。`--device-ids N` 在启动时选择一块主机 GPU；在规范 YAML 中用 `rocm:N` 选择。

vela3-fast 的参考答案只为 CPU 录制，因此在 GPU 上它的自检显示 [`unverified`](../deploy/troubleshooting.md#a-ready-models-self-check-says-unverified)。

</TabItem>
<TabItem value="cuda" label="NVIDIA GPU">

```bash
vllm-sr serve vllm-sr/vela3-fast --engine --platform cuda --device-ids 0
```

`--platform cuda` 选择 `vllm-sr-cuda` 镜像并透传 GPU。

vela3-fast 的参考答案只为 CPU 录制，因此在 GPU 上它的自检显示 [`unverified`](../deploy/troubleshooting.md#a-ready-models-self-check-says-unverified)。

</TabItem>
</Tabs>

在 Engine 模式下，CLI 只启动持久前端和一个托管的模型 worker，都在一个 Router 容器中：

- 第一次启动会下载模型；之后的启动会复用它。
- 启动时打印一段简短的摘要，最后一行是带 System One URL 的 `Ready:`（`--verbose` 打印所有细节）。
- `vllm-sr status` 查看整个栈，`vllm-sr stop` 停止它。
- `--dashboard` 加入 Dashboard，`--observability` 加入 Jaeger、Prometheus 和 Grafana，`--no-minimal` 启动整个栈。

在第二个终端中列出公开的原生模型：

```bash
curl -s localhost:8899/v1/systemone/models
```

:::tip 更大的模型和多模态模型
Decision 3.0 模型还能回答关于图像和视频的问题。在 AMD GPU 上用 `vllm-sr serve vllm-sr/d3-edge -e --platform rocm` 启动最小的 d3-edge，并按[Router 与 Engine 模式](https://github.com/vllm-project/semantic-router/blob/main/src/vllm-sr/INSTANCE_MODES.md#ask-about-images-and-videos)中的说明发送图像和视频。`vllm-sr/d3-turbo` 对每个输入只读取一次就回答它的所有问题，还能在图像和视频中定位物体（[Box 和 track 问题](https://github.com/vllm-project/semantic-router/blob/main/src/vllm-sr/INSTANCE_MODES.md#find-things-in-images-and-videos)）；`vllm-sr/d3-ultra` 会把文本请求的问题列表放在其状态之前读取。[支持的模型](../supported-models/overview.md)列出了每个模型。
:::

<details>
<summary>Engine 模式把状态保存在哪里，以及它发布哪些模型</summary>

- 没有 `--config`，且工作目录中也没有 `config.yaml` 时，Engine 模式把每个模型的配置、状态和下载内容保存在 `~/.local/state/vllm-sr/engine/<model>`。
- 初始的 Engine 配置在默认监听器上发布所选模型。已有的配置会保留其显式的 `listeners[].systemone.models` 允许列表和 API 密钥；以另一种模式重启或选择模型都不会扩大它。
- 每次不带 `--engine` 启动都使用 Router 模式；仅指定 MODEL 不会选择 Engine 模式。不带参数的 `vllm-sr serve` 保留已配置的默认判断部署，在新配置中则初始化 Vela 3.0 vela3-fast。

</details>

## 3. 发送请求 {#3-send-a-request}

就一个请求同时问两个问题：它属于哪种工作（一个 **choice**），以及它是否需要多步推理（一个 **是/否** 问题，称为 **noul**）。

```bash
curl -s localhost:8899/v1/systemone -H 'content-type: application/json' -d '{
  "model": "vllm-sr/vela3-fast",
  "state": "Write a Python function that merges two sorted lists.",
  "questions": {
    "kind": {"type": "choice", "instructions": "What kind of work is this?",
             "criteria": {"code": "Writing or fixing code", "math": "Mathematics", "chat": "Anything else"}},
    "reasoning": {"type": "noul", "instructions": "Does answering this need multi-step reasoning?"}
  }
}'
```

`kind` 的答案给出选中的选项和每个选项的概率。`reasoning` 的答案是回答为“是”的概率：

```json title="Response"
{
  "model": "vllm-sr/vela3-fast",
  "answers": {
    "kind": {"type": "choice", "choice": "code", "probabilities": {"code": 0.98, "math": 0.005, "chat": 0.015}, "confidence": 0.97},
    "reasoning": {"type": "noul", "noul": 0.457}
  },
  "usage": {"input_tokens": 68, "output_tokens": 0}
}
```

在 AMD EPYC 9575F 的 16 个核上，这个请求大约需要 0.1 秒。

- 在请求中加入 `"options": {"return_meta": true}` 还会得到 `meta`：回答它的修订版本、精度（`dtype`）和设备，批处理是否会改变其答案（`numerics`），以及耗时。
- `POST /v1/decisions` 接受相同的请求和相同的显式公共模型 ID。它就地回答失败的问题，而 `/v1/systemone` 像 System One 一样把请求作为整体回答。
- 原生发现使用 `/v1/systemone/models`。`/v1/models` 仍是 Chat 发现；在 Engine 模式下，它还会以 `models: [{name, description, release_date}]` 列出 System One 模型。

worker 的其他 API 是独立的；参见 [HTTP API 参考](../reference/http-api.md)。压缩的请求体、multipart 图像上传、assets、Box、track 和长上下文见 [System One 兼容性](../reference/http-api.md#system-one-compatibility)。

## 4. 在 Router 中使用 {#4-use-it-from-the-router}

Router 会为你运行模型。用 `provider: model_runtime` 把模型声明为一个**部署**，然后在 `systemone` 信号中向它提问。把下面的内容保存为 `config.yaml`，并把 `host.docker.internal:8000` 换成一个回答你用户的 OpenAI 兼容后端：

```yaml title="config.yaml"
version: v0.5
listeners:
  - name: http
    address: 0.0.0.0
    port: 8899
    systemone:
      models: [vllm-sr/vela3-fast]
providers:
  defaults:
    model: answer-model
  models:
    - name: answer-model
      backend_refs:
        - name: answer
          endpoint: host.docker.internal:8000
          protocol: http
recipes:
  - name: default
    signals:
      systemone:
        - name: needs_reasoning
          deployment: primary
          question:
            type: noul
            instructions: Does answering this request need multi-step reasoning?
          predicate:
            gte: 0.7
    decisions:
      - name: think-first
        priority: 100
        rules:
          operator: AND
          on_unknown: no_match
          conditions:
            - type: systemone
              name: needs_reasoning
        modelRefs:
          - model: answer-model
            use_reasoning: true
      - name: default-route
        priority: 1
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: answer-model
            use_reasoning: false
global:
  model_catalog:
    deployments:
      primary:
        provider: model_runtime
        artifact: vllm-sr/vela3-fast
        device: cpu
```

验证文件，然后使用这份配置以 Router 模式重启：

```bash
vllm-sr config validate --config config.yaml
vllm-sr serve --config config.yaml --replace-active-config
```

`--replace-active-config` 用你刚写的文件覆盖为该目录保存的任何配置；之后的重启可以省略它以保留 Dashboard 中的修改。`primary` 部署既回答路由问题，也回答直接的 System One 请求，监听器显式发布其原生模型名称。

通过 Router 发送一个请求，看看它走了哪条路由：

```bash
curl -s -D - -o /dev/null localhost:8899/v1/chat/completions \
  -H 'content-type: application/json' -H 'x-vsr-debug: true' \
  -d '{"model": "vllm-sr/auto", "messages": [{"role": "user", "content": "Plan a three-step proof that there are infinitely many primes."}]}' \
  | grep -i '^x-vsr-'
```

`x-vsr-selected-decision` 给出路由名称，`x-vsr-trace` 的 `signals` 成员列出匹配的 `systemone` 信号。

:::note
如果运行时之后重启，在模型恢复之前该信号为未知，期间 `on_unknown: no_match` 会把请求发往 `default-route`。
:::

Router 和 Engine 模式共享一个托管实例。如果要附加一个独立运维的 worker，请使用显式的 `endpoint` 和 `model`；参见[连接 Router](../deploy/connect-the-router.md#attach-to-a-runtime-you-run)。

## 下一步 {#next-steps}

- [选择模型、规模和硬件](./choose-a-model.md)
- 开启一个内置功能：[应用信号](../../tutorials/signal/overview.md#application-signals)、[你自己的问题](../../system-one/use/custom-questions.md)、[用于召回的嵌入](../guides/embeddings.md)
- [连接 Router](../deploy/connect-the-router.md)以及[放置与扩展 worker](../deploy/placement-and-scaling.md)：GPU、Kubernetes、共享运行时
- [故障排查](../deploy/troubleshooting.md)

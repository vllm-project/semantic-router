---
title: 命令参考
sidebar_label: 命令
sidebar_position: 1
description: vllm-sr serve 和 vllm-srun serve 的选项、其他 vllm-srun 命令，以及 models 文件。
is_mtpe: true
translation:
  source_commit: "bc5c6bfb747ffe9c5f35d98b203427dada8c4c56"
  source_file: "docs/model-runtime/reference/commands.md"
  outdated: true
---

# 命令参考 {#command-reference}

`vllm-sr serve` 启动 Router 的持久前端及其托管的运行时 worker；`vllm-srun serve` 直接启动一个 worker。本页列出两者的每个选项，以及 worker 可以读取的 models 文件。

## `vllm-sr serve` 与 `vllm-srun serve` {#vllm-sr-serve}

`vllm-sr serve MODEL ...`（引擎模式）在前台跑 runtime，跑在 router 镜像拉起的 Docker 或 Podman 容器里。不带 `MODEL` 参数时，`vllm-sr serve` 起的是 router。

| 选项 | 默认 | 含义 |
| --- | --- | --- |
| `MODEL ...` | | Hub 仓库、内置模型名或本地包目录，容器通过只读挂载读它们。多个模型共用一个进程。`MODEL@REVISION` 钉 revision。 |
| `--models FILE` | | 用 models 文件代替 `MODEL` 参数。其中的本地包也会挂载进去。 |
| `--revision SHA` | | 要加载的 40 位 commit，对一个 `MODEL` 生效。 |
| `--platform` | `cpu` | 镜像与 GPU 透传：`cpu`（`vllm-sr`）、`amd`（`vllm-sr-rocm`，带 ROCm 设备）或 `nvidia`（`vllm-sr-cuda`，带 NVIDIA GPU）。macOS 只能跑 `cpu`。 |
| `--device` | `auto` | `auto`，或镜像跑得动的：`cpu`、`--platform amd` 时的 `rocm[:N]`、`--platform nvidia` 时的 `cuda[:N]`，或镜像里装了插件时的插件加速器。 |
| `--host` | `127.0.0.1` | runtime 端口发布在哪个宿主地址。 |
| `--port` | `8100` | runtime 发布在哪个宿主端口。 |
| `--max-num-seqs`、`--max-num-batched-tokens`、`--batch-window-ms`、`--dtype`、`--enable-shared-context` | 已配置的值，或 runtime 默认值 | 默认部署的[引擎参数](../deploy/engine-args.md)，与副本数和放置位置无关。 |
| `--image` | 平台对应的镜像 | 换一个镜像，比如装了插件的。 |
| `--image-pull-policy` | `always` | `always`、`ifnotpresent` 或 `never`。 |
| `--container-runtime` | 自动探测 | `docker` 或 `podman`。 |
| `--log-level` | `info` | runtime 的日志级别。 |

引擎模式把 runtime 下载和编译的东西留在 `~/.cache/vllm-sr/models`（设了 `$XDG_CACHE_HOME` 时是 `$XDG_CACHE_HOME/vllm-sr/models`）；`VLLM_SR_ENGINE_CACHE_DIR` 可以搬走它。`HF_TOKEN`、`HF_ENDPOINT` 和 `HF_HUB_OFFLINE` 按名字进容器，绝不上它的命令行。

`vllm-srun serve` 是同一个服务，选项最全。它是镜像里跑的那条命令；在你自己机器上用它需要一份源码检出（`make model-runtime-install`）。

| 选项 | 默认 | 含义 |
| --- | --- | --- |
| `MODEL ...` | | Hub 仓库、内置模型名或本地包目录。多个模型共用一个进程。`MODEL@REVISION` 钉 revision。 |
| `--models FILE` | | 用 models 文件代替 `MODEL` 参数。 |
| `--revision SHA` | | 要加载的 40 位 commit，对一个 `MODEL` 生效。 |
| `--device` | `auto` | `auto`，或带可选索引的加速器：`cpu`、`cuda[:N]`、`rocm[:N]`、`xpu[:N]`、`npu[:N]`、`mps`，或插件加的。 |
| `--host` | `127.0.0.1` | 监听地址。 |
| `--port` | `8100` | 监听端口。 |
| `--uds PATH` | | 监听 Unix socket，不走 TCP。 |
| `--engine` | `auto` | 引擎插件：`auto`（首个能跑这个模型的，native 优先）、`native`（PyTorch）或 `onnxruntime`（`onnx` extra）。 |
| `--family` | 自动探测 | 强制指定模型家族插件。 |
| `--served-model-name` | 模型名 | API 报告的模型 ID，对一个 `MODEL` 生效。 |
| `--threads` | 全部核 | 进程的 CPU 线程数。ONNX Runtime 模型的每张图各跑一个至多这么多线程的池，大机器上值得设。 |
| `--memory-budget GIB` | | 预估大小超过这个值的模型，拒绝加载。 |
| `--max-queue` | `256` | 每个模型排多少请求之后，runtime 开始答 429。 |
| `--max-queued-tokens` | `4194304` | 每个模型排多少 token 之后答 429。 |
| `--max-num-seqs` | `8192` | 一次前向最多装多少行（跨请求计）：决策请求每个问题一行，其他请求每个输入或窗口一行。`1` 表示每行单独跑。默认值在默认 token 预算下永远不会成为限制。见[引擎参数](../deploy/engine-args.md)。 |
| `--max-num-batched-tokens` | `65536` | 一次前向最多装多少个补齐后的 token：行数乘以最长那一行；对打包行的模型也这样计数。 |
| `--batch-window-ms` | `2.0` | `--max-num-seqs` 大于 1 时，第一个请求到达后调度器再等多久收集更多请求。 |
| `--dtype` | `auto` | `auto`（模型发布时的精度）、`float32`、`bfloat16` 或 `float16`（仅 GPU）。 |
| `--enable-shared-context`、`--no-enable-shared-context` | 由模型家族决定 | 多问题决策请求的共享状态只算一次。只有 Decision 2.0 默认开启。 |
| `--max-request-bytes` | `8388608` | 请求体上限。 |
| `--max-bundle-tasks` | `64` | 一个 `/v1/bundle` 请求最多带多少任务。 |
| `--load-attempts` | `5` | 一个模型加载几次之后就算 `failed`；包损坏或自检失败不重试。 |
| `--load-retry-seconds` | `5` | 加载失败的模型第一次重载前等多久，之后翻倍，最多 300 秒。 |
| `--result-cache-entries` | `16384` | 每个模型按内容留多少条近期结果；`0` 关掉缓存。 |
| `--cache-dir` | `HF_HUB_CACHE` | Hugging Face 缓存目录。 |
| `--offline` | 关 | 只用缓存里已有的文件。 |
| `--base-path DIR` | | 适配器包需要的钉版基模型的本地副本，免去下载。文件会按包里记的哈希核对。 |
| `--accept-licence ID` | | 接受受限许可证（可重复）。 |
| `--log-level` | `info` | `debug`、`info`、`warning` 或 `error`。 |
| `--autotune-cache DIR` | `$VLLM_SRUN_AUTOTUNE_CACHE` | 把编译好的 GPU kernel、以及没有记录选择的模型的 kernel 调优留在多次运行之间。内置模型在 MI300X 和 MI325X 上跑的是记录在案的选择，不调优。 |

## 其他命令 {#other-commands}

其他命令：`vllm-srun models` 列内置模型及钉住的 revision；`vllm-srun plugins` 列已装的家族、引擎和加速器；`vllm-srun devices` 以 JSON 打印本机有哪些设备、`--device auto` 先试哪个。

## models 文件 {#models-file}

models 文件列出一个进程的模型，各自带选项：

```yaml title="models.yaml"
models:
  - model: vllm-sr/Vela-1.0-Encoder-307M-Domain
    name: vela-domain
    device: cpu
  - model: vllm-sr/Decision-2.0-Kai-0.6B
    name: decision-kai
    device: auto
    enable_shared_context: false
```

每条目收 `model`（必填）、`revision`、`name`、`device`、`engine`、`family`、`memory_budget_gib`、`options`（家族选项；Vela 2.0 模型还收 `max_scan_tokens`，即它的[扫描预算](./http-api.md#long-inputs)），以及[引擎参数](../deploy/engine-args.md) `max_num_seqs`、`max_num_batched_tokens`、`batch_window_ms`、`dtype` 和 `enable_shared_context`；条目没写的取进程命令行上的值。router 为它托管的进程写这个文件。

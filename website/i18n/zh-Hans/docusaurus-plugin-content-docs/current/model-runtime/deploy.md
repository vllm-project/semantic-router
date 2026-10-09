---
title: 让 router 来跑模型
description: router 如何替你跑模型、如何把模型放上 GPU 或独立进程，以及如何挂上你自己运行的 runtime。
translation:
  source_commit: "fddd53d7c446c30ae183d16b6d63ec54f7a00a3e"
  source_file: "docs/model-runtime/deploy.md"
  outdated: false
is_mtpe: true
---

# 让 router 来跑模型 {#run-it-with-the-router}

router 有两种跑模型的方式：

- **托管（默认）。** router 把 runtime 当子进程拉起来，崩了就重启，router 停的时候它也跟着停。
- **挂接。** runtime 你自己跑——比如放在 GPU 机器上或作为 Kubernetes 服务——router 用 `endpoint` 连上去。

两种方式用的是同一份配置、同一批模型，答案也一样。

## 内置功能不用配任何东西 {#built-in-features-need-no-configuration}

打开内置功能时（比如 `domain` 信号或语义缓存），router 已经知道该用哪个 Vela 模型，并把它跑在 CPU 的托管 runtime 里。在功能模块上设 `use_cpu: false`，就能让 runtime 自己挑 GPU（`device: auto`）。

只有活跃功能用到的模型才会被启动。声明一个没人用的部署，不会加载它。

## 描述一个部署 {#describe-a-deployment}

想自己挑模型、设备或进程，就写一个部署。部署就是 `global.model_catalog.deployments` 下一个有名字的模型：

```yaml
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
      decision-kai:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Kai-0.6B
        device: auto
        process: decisions
```

| 字段 | 含义 |
| --- | --- |
| `provider` | 对 runtime 服务的模型来说，永远是 `model_runtime`。 |
| `artifact` | Hugging Face 仓库，或本地副本的绝对路径。 |
| `revision` | 要加载的 40 位 commit。内置模型已钉好版本；其他仓库需要指定。 |
| `device` | `auto`（默认）、`cpu`、`cuda:N`、`rocm:N`、`xpu:N`、`mps`，或插件添加的加速器。 |
| `profile` | `exact`（默认），或选择加入的更快档位。见[档位](./profiles.md)。 |
| `input` | 对任务模型：最长输入（`max_tokens`，token 数），以及超长输入怎么办（`overflow`：`reject`、`truncate` 或 `window`）。 |
| `process` | 同名部署在同一个 runtime 进程里跑。 |
| `endpoint` | 挂到一个你自己跑的 runtime，不新起进程。 |
| `served_name` | 挂接的 runtime 同时服务多个模型时，本模型在那边叫什么（默认用部署名）。 |

然后用一个**绑定**把功能连到部署上：

```yaml
global:
  model_catalog:
    bindings:
      domain_classifier:
        deployment: vela-domain
        contract: label_distribution.v1
```

每个功能读一种答案，也就是它的 `contract`：标签概率（`label_distribution.v1`）、互相独立的标签分（`label_scores.v1`）、文本片段（`token_spans.v1`）、向量（`embedding.v1`）或相关性分（`relevance_scores.v1`）。每个功能的绑定写法，看对应的任务指南。启动时 router 会把每个绑定和模型自身的说明（它的头、标签、维度和输入上限）对一遍，对不上就拒绝服务流量。

写在 `global.model_catalog.bindings` 下的绑定全局生效。配方可以在自己的 `routing.model_bindings` 下覆盖它。

## 把模型编进进程 {#group-models-into-processes}

默认同一块 GPU 上的模型共用一个 runtime 进程，`rocm:0` 上的所有模型一次调用内出结果，显存也用得省。它们排队用 GPU，一次一个设备调用；哪个模型不能等别人，就给它一块独占的 GPU。CPU 模型会摊到多个进程上，一个模型一个，最多到 router 可用核数的一半，这样一个请求的几个模型能并行跑；每个进程分到一样多的核当线程。`VLLM_SRUN_CPU_PROCESSES` 给 CPU 进程数封顶，设 `1` 就把所有 CPU 模型关在一个进程里。能不动就不动：挤在一个进程里，可选的 ONNX Runtime 引擎上的模型和 PyTorch 模型要抢 CPU 线程，负载一高其中一个就答得慢。

`device: auto`（默认）的部署，按 auto 在 router 宿主机上挑中的设备分组。宿主机没有 GPU 那就是 CPU，于是每个部署各占一个 CPU 进程，跟 `device: cpu` 一样。有 GPU 的宿主机上，它们和显式指定这块 GPU 的部署共用首个 GPU（比如 `rocm:0`）的进程；GPU 显存不够时 runtime 仍可能把某个模型放到别的设备上，被放去 CPU 的模型可能会用上 router 的每一个核。`vllm-srun devices` 会告诉你 auto 先挑哪个设备。router 只问一次——某个 auto 部署第一次跑的时候问；runtime 答不上来，auto 部署就共用一个进程直到 router 重启，并且 router 会记一笔 `auto_device_unresolved`。

某个部署不想和别人共担故障域或显存，就给它自己的 `process` 名，比如大型决策模型：

```yaml
global:
  model_catalog:
    deployments:
      decision-lux:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Lux-9B
        device: rocm:0
        process: large-decisions
```

这个进程崩了或者显存耗尽，只有它的部署不可用，router 重启它即可；其他模型照常答题。

## 挂到你自己跑的 runtime {#attach-to-a-runtime-you-run}

在 router 够得着的地方起一个 runtime，然后把部署指过去：

```bash
vllm-sr serve vllm-sr/Decision-2.0-Lux-9B vllm-sr/Vela-1.0-Encoder-307M-PII --platform amd --device rocm:0 --host 0.0.0.0 --port 8100
```

```yaml
global:
  model_catalog:
    deployments:
      gpu-lux:
        provider: model_runtime
        endpoint: http://gpu-runtime.internal:8100
        served_name: vllm-sr/Decision-2.0-Lux-9B
      gpu-pii:
        provider: model_runtime
        endpoint: http://gpu-runtime.internal:8100
        served_name: vllm-sr/Vela-1.0-Encoder-307M-PII
```

`endpoint` 收 `http://host:port`、`https://host:port` 或 Unix socket（`unix:///path/to/runtime.sock`）。挂接的 runtime  router 不启动、不重启、不停——它只探活，runtime 挂了就当作不可用。

这样起出来的 runtime 默认只听 `127.0.0.1`，除非传 `--host`。只把它暴露在私有网络上：它自身没有任何认证。

在大机器上，这样的 runtime 若要服务可选的 ONNX Runtime 引擎上的模型（一个 ONNX 包，或一个 Omni 包），就给它 `--threads` 或者跑进 cpuset：该模型每张图各占一个至多 `--threads` 的 CPU 线程池，不设的话会占到进程可用的每一个 CPU（一个 Omni 包有四张图）。router 拉起的 runtime 总会按核数分到自己的 `--threads`。

首次启动时，runtime 会把 router 用到的模型下载进它的模型卷（`/app/models`，即 chart 的 `persistence` 声明，默认 10 GiB）。按你路由的模型来定这个声明的大小：光 Vela Omni Mini 就要 4.3 GB。在离线的集群里，先把它们拉进去（见[故障排查](model-runtime/troubleshooting.md#the-runtime-stays-in-loading-or-warming)）。

### 在 Kubernetes 上 {#on-kubernetes}

router 镜像自带 CPU runtime，托管部署在任何集群里都能跑。ROCm 版 router 镜像 `ghcr.io/vllm-project/semantic-router/vllm-sr-rocm` 带的 runtime 含 ROCm 版 PyTorch，`vllm-sr-cuda` 带的含 CUDA 版 PyTorch：给 router pod 一张 GPU（资源限额里写 `amd.com/gpu` 或 `nvidia.com/gpu`），部署上设 `device: rocm:0` 或 `device: cuda:0`，这个模型就由 router 自己在 GPU 上跑。`vllm-sr serve --target kubernetes --platform amd|nvidia` 会把镜像和 GPU 限额写进 chart 的 values。

ROCm 镜像里留在 CPU 上跑的模型，用的是它的 ROCm 版 PyTorch。对多数 Vela 任务模型，这个构建过不了 runtime 的加载期检查（单独答和批量答要一致），所以在 `exact` 下它们排队逐个答请求：答案一样，负载下吞吐低些。模型全在 CPU 上跑的 router，该用 CPU 镜像。ROCm 镜像的 PyTorch 没有 CPU LAPACK，带 gated delta rule 层的解码器（Qwen3.5 一系的模型，比如 Decision 2.0 Lux-9B 和 Vela 2.0 4B）在它的 CPU 上根本加载不了：runtime 在启动时、权重加载前就拒绝 `device: cpu`，并说明原因。这类模型放 GPU 上跑，或者用 CPU 镜像跑。

多个 router 想共享 GPU 模型，就用同一个镜像把 runtime 单独跑成一个 Deployment，所有 router 挂它的 Service。镜像里的 `vllm-srun` 命令就是干这个的：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-runtime
spec:
  replicas: 1
  selector:
    matchLabels: {app: model-runtime}
  template:
    metadata:
      labels: {app: model-runtime}
    spec:
      containers:
        - name: runtime
          image: ghcr.io/vllm-project/semantic-router/vllm-sr-rocm:latest
          command: ["vllm-srun"]
          args: ["serve", "vllm-sr/Decision-2.0-Lux-9B", "--device", "rocm:0", "--host", "0.0.0.0", "--port", "8100"]
          resources:
            limits: {amd.com/gpu: 1}
          ports:
            - {name: http, containerPort: 8100}
          readinessProbe:
            httpGet: {path: /health, port: http}
          livenessProbe:
            httpGet: {path: /health/live, port: http}
---
apiVersion: v1
kind: Service
metadata:
  name: model-runtime
spec:
  selector: {app: model-runtime}
  ports:
    - {name: http, port: 8100, targetPort: http}
```

部署的 `endpoint` 设成 `http://model-runtime.<namespace>.svc.cluster.local:8100`。就绪探针用 `/health`：只有每个模型都加载完并通过自检，它才成功。

## 模型还没就绪的时候 {#when-a-model-is-not-ready}

router 只有把它为这个配置跑的模型都加载好了，才开始伺候这个配置。启动时它等每一个托管部署：路由用到的任务模型（domain、PII、guard、safety、fact-check、feedback 和 hallucination 模型，以及你自己的分类器），加上 `decision` 信号和 `decision` 选择算法问的决策模型。挂接 runtime 的任务模型它也等，所以要赶在 router 前面起那个 runtime。等的时候，`/health` 应声，`/ready` 返 `503`；`/startup-status` 逐个托管部署列它的状态（`starting`、`loading`、`warming`、`ready`……），`vllm-sr serve` 打印它还在等的那些。这次等待——含首次下载——由 `VLLM_SRUN_READY_TIMEOUT` 兜底（默认 10 分钟）。有模型加载失败或没能及时就绪，router 就不启动；它的日志和它最后的启动状态（`phase: error`）会点出部署和原因（见[故障排查](model-runtime/troubleshooting.md#the-runtime-reports-failed)）。

router 开始服务后，请求绝不会等一个答不上来的模型：

| 情况 | 功能看到什么 |
| --- | --- |
| 模型还在下载或加载 | 未知 |
| 答案在功能超时之后才到 | 未知 |
| runtime 过载 | 未知 |
| runtime 进程崩了 | 未知，直到 router 重启它（退避从 1 秒到 60 秒） |
| 一个进程里的模型全部加载失败 | 未知；router 按同样的退避重启该进程，直到模型加载成功 |
| 输入超过部署允许的长度且 `overflow: reject` | 这条输入报错 |

未知的信号不参与匹配。决策的 `rules.on_unknown` 决定未知信号对这条路由意味着什么（默认 `no_match`，或 `fail_request`），guard 和 PII 模块有 `on_error: allow | block`。`decision` 选择算法回退到 `modelRefs` 里的第一个模型。

## 看看什么在跑 {#check-what-is-running}

- router 在 metrics 端口导出 `vsr_model_runtime_ready{deployment="..."}`（部署能答题时为 1）、`vsr_model_runtime_restarts_total`、按结果分的 `vsr_model_runtime_requests_total` 和按原因分的 `vsr_model_runtime_unknown_answers_total`。
- router 的管理 API 列出每个部署及其进程、状态、重启次数和所服务的模型（标签、设备、档位）：`curl -s localhost:8080/api/v1/inventory/model-runtime`。
- 挂接的 runtime 直接应答 `GET /v1/models` 和 `GET /health`。

[参考](./reference.md#router-managed-runtimes) 列出能改 runtime 命令、socket 目录和托管 runtime 模型缓存的环境变量。

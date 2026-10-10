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

## 放置与扩容副本 {#place-and-scale-replicas}

![按部署的副本调度与模型 worker 内的各界面](/img/architecture/system-one/03-serving-engine-and-workers.svg)

用 `-dp` 设置默认部署的独立 worker 数。数值档位和放置位置分开管：

```bash
# 两个 worker 放在一块显式指定的宿主机 GPU 上。
vllm-sr serve vllm-sr/Vela-2.0-4B -e --platform rocm -dp 2 --device-ids 0
# 两个 worker 分放两块宿主机 GPU。
vllm-sr serve vllm-sr/Vela-2.0-4B -e --platform rocm -dp 2 --device-ids 0,1
```

不给设备 ID 时，新的 GPU 放置用前 N 块可用 GPU，容量不够就拒绝，不会悄悄把每个 worker 都塞到一块卡上。已写明的放置按声明顺序缩改。只指定 MODEL 时保留配置好的副本和档位。这些 flag 写的是下面同一份规范化资源，没有单独的 DP 状态。Kubernetes 上用配置里的分配序，不用宿主机 ID。

每个受管副本各有自己的进程。一个部署的全部副本共用同一制品、revision、档位和能力。一个请求的融合问题批在逻辑模型选定后作为一个单元调度。池子挑就绪且待处理请求字节最少的 worker，并列时挑最近最少分配的。这是本地负载估计，不是挂接 runtime 的队列长度，也不是实测 token 数。原生请求和路由任务共享同一个物理 worker 的负载计量。

```yaml
global:
  router:
    enabled: false
  model_catalog:
    deployments:
      primary:
        provider: model_runtime
        artifact: vllm-sr/Vela-2.0-4B
        profile: exact
        replicas:
          - device: rocm:0
          - device: rocm:1
    system:
      decision_model:
        deployment: primary
```

用不同的 GPU 加算力。显式重复同一设备就在该 GPU 上起多个 worker；每个 worker 要为自己的模型留显存。选布局前，用你的输入长度和并发实测吞吐和尾延迟。`batching` 能提高单个 worker 内的跨请求利用率；`max_speed` 还会改变数值，需要单独做质量评估。副本不做权重分片，也不是张量或流水并行。

`replicas` 不要和部署级的 `device`、`endpoint` 或 `served_name` 一起用。省略 `replicas` 保留单 worker 简写，调度和状态契约与显式单副本相同。单 worker 部署保留前端结果缓存；多副本用各自独立的 worker 缓存。副本的放置与绑定无关，路由消费方的变化不会改动 worker 身份。重配先准备好兼容的池再发布，退场的 worker 会排空已有请求。

池子每个 worker 最多接纳 32 个在途 HTTP 请求，没有池级等待队列。就绪副本全满时返回过载。这个上限不是 GPU 前向并发；前向已经开始的，调用方超时后它仍可继续。清单报告期望与就绪副本数、各 worker 的就绪情况、在途请求和重启次数。降级的池仍能靠健康副本继续服务。

### CPU 线程 {#cpu-threads}

每个受管 CPU worker 默认分到 router 可用 CPU 预算的一半（向下取整），至少一个线程、至多 16 个。要改每个 worker 的线程数，在启动栈之前设一个正整数：

```bash
VLLM_SRUN_CPU_THREADS=8 vllm-sr serve --platform cpu --config config.yaml
```

该设置受可用 CPU 预算封顶，它不设 worker 数。每个部署或副本仍各有独立的 worker，模式变化不在活跃消费方之间重新分配线程。从默认值开始，用实际输入长度和并发对比延迟和吞吐。几个 CPU 模型都忙时，少些线程能减少争抢；专用核上的长输入可能适合更高的覆盖值。这是每 worker 上限，几个忙 worker 仍可能争同一批核。共享宿主机时给其他服务留核。通过 `endpoint` 挂接的 runtime 自管线程。

## 发布原生 API {#publish-a-native-api}

用上面的 `primary` 部署，给独立 listener 加一份显式授权，客户端就能直接问它：

```yaml
listeners:
  - name: http
    address: 0.0.0.0
    port: 8899
    systemone:
      models: [vllm-sr/Vela-2.0-4B]
```

用 `GET /v1/systemone/models` 发现已发布的原生模型，然后带那个 `model` ID 发 `POST /v1/systemone` 或 `POST /v1/decisions`。部署用它的 Hub 制品 ID，除非设了 `public_name`；本地制品需要显式公开名。客户端要认证时就加 listener `api_keys`。

原生授权与 Chat 的 `listeners[].models` 白名单是两回事，两种启动模式下都有效。模型被替换或扩容时，既有授权不变；想发布另一个模型时再更新它。完整的原生请求见 [quickstart](model-runtime/quickstart.md#3-send-a-request)。发布不保证就绪；用 `vllm-sr instance --config config.yaml models` 检查。

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

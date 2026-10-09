---
title: 故障排查与常见问题
sidebar_label: 故障排查与常见问题
description: 修复模型运行时的常见问题，并解答常见疑问。
translation:
  source_commit: "5c3869fc7b8c7585f4a0dcbc89a4da1ccb34d1c9"
  source_file: "docs/model-runtime/troubleshooting.md"
  outdated: false
---

# 故障排查与常见问题

先看运行时对自身的报告。对于在 8100 端口独立运行的 `vllm-srun` worker：

```bash
curl -s localhost:8100/health
curl -s localhost:8100/v1/models
```

对于路由器托管的运行时，查看路由器的指标和日志：

```bash
curl -s localhost:9190/metrics | grep '^vsr_model_runtime'
```

`vsr_model_runtime_ready{deployment="..."} 1` 表示该 deployment 可以作答。
路由器日志会列出每个托管运行时进程以及它停止的原因。

使用 `vllm-sr serve ARTIFACT --engine` 时，应访问前端 listener 的 `/v1/systemone/models`。公网 `/v1/models` 列出 Chat 模型，不是私有 worker 清单；配置了 listener API key 时还需要携带相应凭据。

## 启动会等待模型 {#startup-waits-for-the-models}

只有当路由器为当前配置托管的每个模型都加载完成后，它才开始服务。在此之前，`/ready` 返回 `503`，
`vllm-sr serve` 会一直等待，并打印路由器还在等待什么：

```text
Waiting for Router-managed model deployments, 0 of 1 ready: decision-kai (vllm-sr/Decision-2.0-Kai-0.6B) loading
```

`/startup-status` 报告 `phase: loading_model_deployments`，并在 `model_deployments` 中列出每个
deployment 及其状态：

```bash
curl -s localhost:8080/startup-status
```

离线的附加运行时不会阻塞自定义 `decision` 问题或显式 `decision.v1` 任务绑定的准备；当前能力信息就绪且兼容之前，这些任务不可用。隐式绑定和原生任务头仍可能需要在启动时读取模型元数据。见[部署就绪规则](/docs/model-runtime/deploy#when-a-model-is-not-ready)。

首次启动要下载模型，因此比之后的启动更久。等待在 `VLLM_SRUN_READY_TIMEOUT`（默认 10 分钟）后结束，
并报告 `did not become ready within 10m0s`；再次启动即可从缓存继续下载，或在路由器的环境中调大该超时。
模型加载失败时，等待会立即结束并给出原因（见下文）。

## 信号从不匹配 {#a-signal-never-matches}

很可能是模型还没就绪，或者它的答案到得太晚。

1. 查看该 deployment 的 `vsr_model_runtime_ready`。它为 `0` 时，使用它的每个信号都为未知，不会匹配。
2. 按 `reason` 查看 `vsr_model_runtime_unknown_answers_total`。`timeout` 表示答案晚于信号的
   `timeout_ms`：调大它，或改用更小的模型或 GPU。`unavailable` 表示运行时未就绪或不可达。
   `overloaded` 表示它的队列已满。
3. 打开调试并发送同样的文本，读取 `x-vsr-matched-*` 响应头：

   ```bash
   curl -s -D - -o /dev/null localhost:8899/v1/chat/completions \
     -H 'content-type: application/json' -H 'x-vsr-debug: true' \
     -d '{"model": "vllm-sr/auto", "messages": [{"role": "user", "content": "your text"}]}'
   ```

4. 预览同一段文本的路由，而不生成回答。返回中的 `signal_errors` 列出未知的信号及原因，
   例如 `decision_timeout`：

   ```bash
   curl -s 'localhost:8080/api/v1/routing/preview?trace=true' \
     -H 'content-type: application/json' \
     -d '{"model": "vllm-sr/auto", "text": "your text"}' \
     | jq '{decision: .decision_result.decision_name, matched: .decision_result.matched_signals, signal_errors}'
   ```

## 运行时一直处于 `loading` 或 `warming` {#the-runtime-stays-in-loading-or-warming}

- **首次启动：** 模型正在下载，大模型需要几分钟。路由器日志和 `GET /health` 会显示当前阶段。
- **没有网络：** 运行时从 Hugging Face Hub 下载。离线时，先把模型复制到缓存中，或把 `artifact` 指向本地副本。
  例如，把固定修订版的 Vela Omni Nano 下载到路由器镜像所管理的运行时的缓存（即模型卷）中，
  然后以 `HF_HUB_OFFLINE=1` 启动：

  ```bash
  hf download vllm-sr/Vela-1.0-Omni-Nano \
    --revision 2ff2d66385dbdd661a560ec3e8bcb45a0527d92e \
    --cache-dir /app/models/model-runtime
  ```

  Mini 的修订版是 `801bae3ad28df6891408f0e0441c676b30e132e3`。路由器镜像不再内置预先导出的 Omni 包，
  因此在离线集群中按图片路由的路由器需要先这样下载一次。
- **CPU 上长时间处于 `warming`：** 自检会让模型跑几次请求。大型决策模型在 CPU 上很慢；
  请使用 GPU 或 `vllm-sr/Decision-2.0-Kai-0.6B`。
- **`loading` 且原因中有 "retrying after ..."：** 模型因可能自行消失的原因加载失败，例如 GPU 被占用、可用内存不足或下载中断。运行时最多重试五次，首次等待 5 秒，之后每次加倍，期间同一进程中的其他模型照常服务（`--load-attempts`、`--load-retry-seconds`）。包损坏或自检失败会立即报告 `failed`。

## 运行时报告 `failed` {#the-runtime-reports-failed}

`GET /v1/models` 会给出每个模型失败的原因。由路由器运行模型时，路由器日志会带有同样的原因，例如
`model runtime is not ready: model @domain_classifier failed to load: ...`。
路由器运行的某个运行时进程中所有模型都加载失败时，路由器会重启该进程（首次等待 1 秒，之后最长间隔 60 秒），
因此 GPU 被占用、磁盘已满这类暂时性原因消除后，模型会自行恢复。与仍在服务的模型同处一个进程的模型加载失败时，由运行时自己重试，该进程会继续运行。路由器托管的模型重试三次仍然失败时，路由器无法启动；配置重新加载时，新配置会被拒绝，上一份配置继续服务。常见原因：

| 原因提示 | 处理方法 |
| --- | --- |
| a file hash does not match | 下载已损坏或仓库发生了变化。从缓存中删除该模型后重新启动。 |
| a revision is required | 非内置仓库需要用 40 位 commit 设置 `revision`。 |
| access denied, gated or private | 用 `hf auth login` 登录，或为有访问权限的账号设置 `HF_TOKEN`。 |
| does not fit, out of memory | 换用更小的模型、显存更大的 GPU，或把副本分配到不同 GPU。 |
| device not available | 指定的 GPU 不存在，或已安装的 PyTorch 不支持它。使用 `device: auto`，或安装正确的 PyTorch 版本。 |
| built without LAPACK | 该模型在 CPU 上需要 LAPACK，而当前的 PyTorch（ROCm 镜像中的版本）没有。把模型放到 GPU 上（`device: rocm:0`），或用 CPU 镜像运行 CPU 上的模型。运行时不会重试。 |
| no family recognizes the package | 不支持该模型的架构。见[选择模型](model-runtime/choose-a-model.md#your-own-models)。 |
| a licence must be accepted | 该模型的许可证限制了使用。确认你可以使用后，传入 `--accept-licence <id>`。 |

同一进程中一个模型失败时，该进程中的其他模型继续提供服务。

## 已就绪模型的自检显示 `unverified` {#a-ready-models-self-check-says-unverified}

`GET /v1/models` 在 `golden` 下给出每个模型的自检结果。`matched` 表示它的回答与该类设备上发布版的回答一致。
`unverified` 表示模型可以提供服务，但运行时无法担保这一点：

- **没有 `reason`，且 `golden.reference` 为空：** 该设备没有发布版的回答，例如非内置模型，或没有验证记录的 GPU（CUDA）。
  运行时只检查了回答格式正确，且每次运行都相同。
- **`reason` 以 `kernel choices not applied` 开头：** 在 AMD Instinct MI300X 和 MI325X GPU 上，Decision 2.0 和 Vela 2.0
  使用发布时用 FLA 0.5.2 记录的内核配置运行，因此每个进程给出相同的回答。这些配置无法在当前环境运行时，模型仍会加载，
  日志会写 `... runs without its recorded kernel choices: ...`，它的回答可能与发布版不同。

| `reason` 提示 | 处理方法 |
| --- | --- |
| `FLA is not installed` | 安装 `fla-core==0.5.2`，或使用自带它的路由器 ROCm 镜像。 |
| `they were recorded with FLA 0.5.2, not ...` | 安装 `fla-core==0.5.2`。 |
| `failed to import` | 修复 FLA 安装；消息中给出了具体错误。 |
| `has no autotuned kernel ...` | 安装 `fla-core==0.5.2`。 |

`--autotune-cache` 会在重启之间保留已编译的内核，但不能代替记录的配置。

## 路由器拒绝配置 {#the-router-refuses-the-configuration}

| 消息提到 | 处理方法 |
| --- | --- |
| `removed model execution fields`、`candle`、`ort`、`openvino`、`precision`、`variant`、`use_mmbert_32k`、`use_nli`、`polarity_guard` | 运行 `vllm-sr config migrate`。见[迁移](model-runtime/migrate.md)。 |
| a binding does not match the model | 模型的标签、头、维度与该功能所需不同，或输入上限更小。绑定为该功能训练的模型，或修改 `input`。 |
| `artifact must be a Hub repository ID or an absolute package path` | Hub 模型用 `owner/name`，本地副本用绝对路径。 |
| `revision must be a 40-hex commit` | 使用完整的 commit 哈希，而不是分支或标签。 |

## 长输入被拒绝 {#a-long-input-is-rejected}

任务 deployment 会拒绝超过 `input.max_tokens` 的输入，而不是悄悄截断。
把 `max_tokens` 调大到模型上限，或选择其他策略：

```yaml
global:
  model_catalog:
    deployments:
      vela-pii:
        provider: model_runtime
        artifact: vllm-sr/Vela-1.0-Encoder-307M-PII
        input:
          max_tokens: 32768
          overflow: window
```

`truncate` 保留文本开头。`window` 用相互重叠的窗口读取全文并合并结果，PII 和安全扫描应使用它，以免漏检。

`window` 最多读取 `max_tokens`：更长的输入以 `scan_budget_exceeded` 失败。
通过 Vela 2.0 时，路由类问题只读取长请求的前若干 token（Vela 2.0 0.3B 在 CPU 上为 8,192 个），
安全类问题则在模型的扫描预算内读取全文（CPU 上为四个输入）。要让安全类问题读取更多，
请给 deployment 设置扫描预算：

```yaml
global:
  model_catalog:
    deployments:
      vela2:
        provider: model_runtime
        artifact: vllm-sr/Vela-2.0-0.3B
        input:
          max_tokens: 131072
          overflow: window
```

对于 prompt guard 和 safety 信号，如果原生模型因单次输入过长而拒绝处理，路由器会以有限并发
重试覆盖全文的重叠窗口，取各窗口的最高风险，并要求每个窗口确认已完整读取输入。
未完成的扫描仍是未确定状态；扫描预算和截止时间继续生效。直接调用 `/v1/decisions` 时，
仍遵循模型本身的输入长度限制。

模型没有完整读取的内容会让越狱或 PII 规则匹配：在 `reject` 下超过模型 `max_tokens`
的输入、超过其上限的输入、被截断的输入，或未能在信号截止时间内扫描完的输入。无论
`on_error` 如何设置，匹配的类型都是 `unscanned`，并给出原因（`input_limit`、`scan_budget`
或 `deadline`），因此填充提示词无法让攻击或个人数据绕过检查。在模块上设置
`on_unscanned: allow` 可让这类内容改为遵循 `on_error`。其他信号报告原因，并遵循各自的
`on_error`（[参考](model-runtime/reference.md#long-inputs)）。

## 请求在等待慢模型 {#a-request-waits-on-a-slow-model}

在信号截止时间前没有返回的模型运行时信号会按其策略处理，因此一个慢模型不会让整个请求失败：
路由类信号遵循 `on_error`，安全类信号按未扫描处理。截止时间是请求的截止时间减去剩余时间的十分之一，
已部署服务的请求没有路由器可见的截止时间，则为 45 秒；设置 `global.model_catalog.signal_timeout_ms`
可以缩短它。在 CPU 上，Vela 2.0 0.3B 用四个核每秒约读取 1,000 个 token，因此长请求可能无法在截止时间内完成安全扫描。

## 路由器无法访问挂载的运行时 {#the-router-cannot-reach-an-attached-runtime}

- 除非用 `--host 0.0.0.0` 启动，运行时只监听 `127.0.0.1`。
- 在容器内部，`localhost` 指容器自身。请使用宿主机地址、Docker 网络名或 Kubernetes Service。
- 从路由器所在的机器执行 `curl <endpoint>/health` 必须有响应。

## 请求比预期慢 {#requests-are-slower-than-expected}

- 对于路由器托管的运行时，对比路由器上较慢 deployment 的 `vsr_model_runtime_server_seconds`（按 `phase`）
  和 `vsr_model_runtime_transport_seconds`（见[参考](model-runtime/reference.md#metrics)）。时间大多在 `forward`
  说明模型本身在该设备上就慢。时间在 `queue` 说明模型已饱和：增加 GPU、改用更小的模型或另起一个进程。
  传输占比大则指向宿主机（CPU 争用、远程 endpoint）。
- 对于你自己启动的运行时，查看它 `/metrics` 上的 `vllm_srun_request_duration_seconds` 和
  `vllm_srun_queue_duration_seconds`，或其响应的 `Server-Timing` 头。
- 在多核 CPU 主机上，先在启动前用 cpuset 或亲和性掩码
  [限制路由器的 CPU 预算](model-runtime/deploy.md#bound-the-cpu-budget-on-many-core-hosts)。
  托管运行时的子进程（包括后加载的进程）会继承允许使用的 CPU 范围。
  只设置线程上限可能让延迟更差；`GOMAXPROCS` 和 `--threads` 并不限制 CPU 范围。
  对于你自己启动的运行时，单独设置 CPU 范围，再按该范围设置线程数。

- 托管 CPU worker 默认使用路由器可用 CPU 预算的一半，向下取整，最少 1 个、最多 16 个线程。
  在 `vllm-sr serve` 启动前设置正整数 `VLLM_SRUN_CPU_THREADS`，即可指定每个 worker 的线程数，上限为完整 CPU 预算。
  用实际输入长度和并发量比较效果：多个模型同时忙碌时，减少线程可能降低争用；有专用核心时，长输入可能受益于更多线程。
  参见 [CPU 线程](model-runtime/deploy.md#cpu-threads)。自行启动并通过 `endpoint` 挂载的运行时使用自己的
  `--threads` 设置，同一进程中的模型共享这些线程。
- 当其他工作占用了部分核心时，CPU 模型会明显变慢，因为每个线程都要等最慢的那个。使用 ROCm GPU
  的进程即使空闲也可能让一个 CPU 核心一直忙碌，同一主机上的 LLM 服务也一样：给 CPU 模型留出专用核心。
- GPU 上的决策模型可以使用 `shared_context` 或 `batching`；见 [Profiles](model-runtime/profiles.md)。

## 常见问题 {#faq}

**需要 GPU 吗？** 不需要。所有内置任务模型以及较小的决策模型都能在 CPU 上运行。

**运行时会把我的数据发到别处吗？** 不会。它只从 Hugging Face Hub 下载模型文件，从不把请求文本发出去。
它不记录请求文本，指标中也不包含请求内容。

**可以使用非内置的模型吗？** 可以，只要架构受支持：提供带 `revision` 的 Hub 仓库，或本地绝对路径。
对于它不认识的模型，运行时会计算并报告其身份。

**从 Hub 加载模型安全吗？** 运行时从不执行模型仓库中附带的代码，并会在加载前对照固定的哈希校验内置模型的每个文件。

**多个路由器可以共享一个运行时吗？** 可以。启动一次，并给每个路由器配置同一个 `endpoint`。

**一个运行时可以提供多个模型吗？** 可以。给 `vllm-srun serve` 传入多个模型，再通过 `endpoint` 挂载各个 deployment。
路由器托管的 deployment 和 replica 各自使用独立的 worker。

**更换 embedding 模型后，我缓存和存储的向量会怎样？** 它们与新向量隔离，不会被复用。
见[更换 embedding 模型时重新向量化](model-runtime/migrate.md#re-embed-when-the-embedding-model-changes)。

**模型存放在哪里？** 你自己启动的运行时存放在 Hugging Face 缓存（`HF_HUB_CACHE`）中，
托管运行时存放在路由器的模型目录下；见[参考](model-runtime/reference.md#router-managed-runtimes)。

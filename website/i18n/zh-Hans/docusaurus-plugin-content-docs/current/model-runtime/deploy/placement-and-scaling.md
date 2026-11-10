---
title: 放置与扩展 worker
sidebar_label: 放置与扩展 worker
sidebar_position: 2
description: 在一块或多块 GPU 上运行一个部署的多个独立副本，了解请求如何分派，并确定 CPU 线程数。
is_mtpe: false
translation:
  source_commit: "bc5c6bfb747ffe9c5f35d98b203427dada8c4c56"
  source_file: "docs/model-runtime/deploy/placement-and-scaling.md"
  outdated: false
---

# 放置与扩展 worker {#place-and-scale-workers}

一个逻辑部署可以作为一个或多个独立的**副本**运行，每个副本都是自己的 worker 进程，并持有自己的一份权重。用副本在更多 GPU 上增加计算能力；用 CPU 线程数确定 CPU 上 worker 的规模。

```bash title="Two workers of the default deployment on two GPUs"
vllm-sr serve vllm-sr/Vela-2.0-4B -e --platform rocm -dp 2 --device-ids 0,1
```

![按部署分派副本，以及模型 worker 内部的接口](/img/architecture/system-one/03-serving-engine-and-workers.svg)

## 放置与扩展副本 {#place-and-scale-replicas}

`-dp` 设置默认部署的独立 worker 数量。其引擎参数与放置相互独立：

```bash
# Two workers on one explicitly selected host GPU.
vllm-sr serve vllm-sr/Vela-2.0-4B -e --platform rocm -dp 2 --device-ids 0
# Two workers across two host GPUs.
vllm-sr serve vllm-sr/Vela-2.0-4B -e --platform rocm -dp 2 --device-ids 0,1
```

- 没有设备 ID 时，新的 GPU 放置使用前 N 块可用 GPU，容量不足时拒绝；它不会悄悄把所有 worker 放到同一块卡上。已有的放置按其声明的顺序调整规模。
- 只指定 MODEL 会保留已配置的副本和引擎参数。
- 这些参数写入下面所示的同一个规范资源；没有单独的 DP 状态。
- 在 Kubernetes 上，请在配置中使用分配序号，而不是主机 ID。

```yaml title="config.yaml"
global:
  router:
    enabled: false
  model_catalog:
    deployments:
      primary:
        provider: model_runtime
        artifact: vllm-sr/Vela-2.0-4B
        replicas:
          - device: rocm:0
          - device: rocm:1
    system:
      decision_model:
        deployment: primary
```

:::warning
副本不会切分权重，也不是张量并行或流水线并行。用不同的 GPU 来增加计算能力：显式重复同一个设备会在该 GPU 上启动多个 worker，而每个 worker 都需要为其模型准备内存。在一个 worker 内部，并发请求的行已经共享前向计算（[引擎参数](./engine-args.md)）。在选择布局之前，请用你的输入长度和并发量测量吞吐量和尾延迟。
:::

### 副本共享什么 {#what-replicas-share}

- 一个部署的所有副本使用相同的制品、修订版本、引擎参数和能力，部署未设置的批处理上限除外：每个 GPU worker 在启动时校准自己的 token 预算，因此各副本报告的 `max_num_batched_tokens` 可能不同。
- 设置 `max_num_seqs` 和 `max_num_batched_tokens` 可以让每个副本使用相同的上限；报告值与设置值不同的副本不会参与分派。
- 不要把 `replicas` 与部署级别的 `device` 或 `endpoint` 组合使用；附加的副本用自己的 `model` 指定其提供的模型。
- 省略 `replicas` 保留单 worker 的简写形式，其分派和状态契约与显式的单个副本相同。
- 单 worker 部署保留前端结果缓存；多副本使用各自独立的 worker 缓存。
- 副本的放置与绑定无关，因此更改使用某个部署的信号会保留不变的 worker 身份。重新配置会在发布之前准备好兼容的池，退役的 worker 会处理完已有请求。

### 请求如何分派 {#how-requests-are-dispatched}

在选定逻辑模型后，一个请求合并后的问题批次作为一个整体分派。池选择未完成请求字节数最少的就绪 worker，平局时选择最久未被分配的 worker。这是本地的负载估计，而不是附加运行时的队列长度或实测的 token 数。原生请求和路由任务共享物理 worker 的负载统计。

- 池对每个 worker 最多接纳 32 个进行中的 HTTP 请求，并且没有池级等待队列。当所有就绪副本都已满时，它返回过载。
- 这个上限不是 GPU 前向计算的并发度：已经在运行的前向计算可能在调用方超时后继续。
- Inventory 报告期望和就绪的副本数、每个 worker 的就绪状态、进行中的请求和重启次数。降级的池可以通过健康的副本继续服务。

## CPU 线程 {#cpu-threads}

每个托管的 CPU worker 默认使用 Router 可用 CPU 预算的一半（向下取整），至少 1 个线程，最多 16 个。要为每个 worker 选择不同的线程数，请在启动栈之前设置一个正整数：

```bash
VLLM_SRUN_CPU_THREADS=8 vllm-sr serve --platform cpu --config config.yaml
```

- 这个设置受可用 CPU 预算限制；它不设置 worker 的数量。每个部署或副本仍然拥有一个独立的 worker，模式变化也不会在活动 worker 之间重新分配线程。
- 先使用默认值，然后用你实际的输入长度和并发量比较延迟和吞吐量。当多个 CPU 模型同时繁忙时，较少的线程可以减少争用；在专用核上处理较长输入时，较大的覆盖值可能更好。
- 这是每个 worker 的上限，因此多个繁忙的 worker 仍可能争用同样的核。与其他服务共享主机时，请为它们预留核。
- 你通过 `endpoint` 附加的运行时自己管理线程。

### 从流量中学习线程数 {#learn-the-thread-count-from-traffic}

托管的 CPU 运行时还可以学习其流量所需的算子内线程数：短输入最快的线程数往往低于长输入，而在共享主机上它会随邻居的负载变化。

| 设置 | worker 的行为 |
| --- | --- |
| `VLLM_SRUN_CPU_ADAPTIVE_THREADS=1` | 学习并报告。它以配置的线程数服务每个请求，同时把允许的各线程数作为计时的影子计算运行，这些计算必须逐位重现所服务的答案；然后为每个输入规模区间记录最快的线程数。把 `--threads` 设为学到的最优值即可应用它。 |
| `VLLM_SRUN_CPU_ADAPTIVE_THREADS=2` | 在过渡窗口内，学到的线程数通过与所服务答案的影子校验后，也用它们来服务。 |

<details>
<summary>学习如何保持精确</summary>

- 只有在启动时的黄金检查通过后才开始探索。以配置线程数进行一次预热计算，可以把缓存排除在比较之外；在这台主机上只要有一个线程数出现漂移，就永久停止学习。
- 使用 `1` 时，worker 的答案始终来自配置的线程数。在任何主机上，单一线程数都是安全的，因为只有线程数的变化才可能改变答案。
- 改变线程数后答案是否保持不变，是主机及其内核的属性：在测量过的 Arm 主机上逐位一致，在 x86 上取决于输入。`2` 适用于有记录证明跨线程数逐位一致的主机，因为只在窗口从未见过的输入上出现漂移的主机，仍可能改变精确的答案。
- 所有前向计算都在每个进程共享的上下文中声明其线程数，因此同一进程中的多个 worker 永远不会继承彼此的线程数。
- 探索会占用最初的一些请求，学到的设置反映了 worker 在学习期间看到的负载；在负载变化的主机上，重启 worker 即可重新学习。

</details>

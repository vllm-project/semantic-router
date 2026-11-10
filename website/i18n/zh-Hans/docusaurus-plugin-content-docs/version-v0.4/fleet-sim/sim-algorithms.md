---
title: 仿真模型
translation:
  source_commit: "33349fdab9ad294da19ebd11588f8adbe8771b4a"
  source_file: "docs/fleet-sim/sim-algorithms.md"
  outdated: false
---

# 仿真模型

Fleet Sim 结合快速排队近似与请求级离散事件模拟器（DES）。本页说明这些模型计算什么，以及结果在何处需要外部验证。

## 模型流程 {#model-flow}

一次典型的 `optimize` 运行中，Fleet Sim：

1. 把工作负载 CDF 拆成短分布和长分布；
2. 根据 GPU profile 估计服务时间分布；
3. 用 M/G/c 排队近似为每个池做规模估算；
4. 按建模小时成本对候选排序；以及
5. 对最多 `--verify-top` 个候选运行 DES。

`simulate` 跳过搜索，对调用方提供的池规模运行 DES。`pareto` 用解析方式评估 CDF 断点。`disagg` 使用单独的阶段级解析模型，不运行请求级 DES。

## 工作负载模型 {#workload-model}

### CDF 采样 {#cdf-sampling}

CDF 点 `[t, f]` 表示比例为 `f` 的请求最多有 `t` 个总 token。样本在每个 CDF 区间内均匀抽取。默认将采样得到的总数按 80% 输入和 20% 输出 token 划分。

合成工作负载还会分配三类之一：

| 类别 | 默认比例 | 用途 |
| --- | ---: | --- |
| `prose` | 0.60 | 压缩后路由的资格 |
| `code` | 0.25 | 视为不安全压缩 |
| `rag` | 0.15 | 压缩后路由的资格 |

到达遵循速率为 `lambda` 的泊松过程。随机种子使给定命令可复现，但单个种子不是置信区间。

独立 HTTP 服务在运行作业前会把上传的轨迹转换成总 token CDF。该路径不会回放原来的到达间隔、输入/输出比例和路由顺序。Python 库还包含 `TraceWorkload`，用于程序化的带时间戳回放。

### 规划含义 {#planning-implication}

长度路由器使用 `input_tokens + output_tokens`。在线路由器在生成前不知道最终输出长度，因此这是容量规划预言，而不是可直接部署的路由规则。评估真实策略时，请使用请求时可用的 token 估计并测量其误差。

## GPU profile {#gpu-profiles}

每个池使用一个 profile，提供迭代延迟、prefill 延迟、KV-cache 容量、最大并发和小时成本。

### 手工 profile {#manual-profiles}

`ManualProfile` 包含测量或估计的常量：

| 字段 | 含义 |
| --- | --- |
| `W` | 基础迭代延迟，单位为秒 |
| `H` | 在 `calibration_ctx` 下每个活跃序列的延迟 |
| `calibration_ctx` | 校准 `H` 时的序列长度 |
| `chunk` | 每次迭代处理的 prefill token 数 |
| `blk_size` | 一个 KV-cache 块中的 token 数 |
| `total_kv_blks` | KV-cache 块预算 |
| `max_slots` | 已校准的并发上限 |
| `cost_per_hr` | 一个 profile 单元的建模小时成本 |

在活跃并发 `n` 和平均序列长度 `L` 时，迭代模型为：

```text
H_effective = H * L / calibration_ctx
iteration_time = W + H_effective * n
```

槽位数取 KV-cache 上限与按池最大上下文缩放后的 `max_slots` 上限中的较小值。手工 profile 对 prefill 和 decode 使用同一迭代模型，因为它们不包含硬件 FLOP 规格。

### 计算 profile {#computed-profiles}

`ComputedProfile` 由硬件规格、模型架构以及张量并行、dtype、chunk 大小和 KV 利用率等服务配置构建。它推导 `W`、`H` 和 KV 块预算。

对 prefill，它估计投影、注意力和前馈 FLOP，并返回计算时间和内存时间中较慢的一方。这种 roofline 计算对敏感性分析有用，但省略了许多运行时效应，仍须对照负载测试校准。

### profile 代表什么 {#what-a-profile-represents}

profile 描述完整的 **模型 + 硬件 + 并行布局 + 服务配置** 组合。仅有 GPU 名称不够。更改模型规模、量化、张量并行、最大上下文、分块或 vLLM 版本，可能同时使 `W`、`H`、容量、成本和功耗值失效。

## 服务时间模型 {#service-time-model}

对输入长度为 `L_in`、输出长度为 `L_out`、prefill chunk 为 `C` 的请求，模型使用：

```text
prefill_iterations = ceil(L_in / C)
prefill_time = prefill_iterations * prefill_iteration_time
decode_time = L_out * decode_iteration_time
raw_service_time = prefill_time + decode_time
TTFT = queue_wait + prefill_time
```

DES 按满槽并发推导的有效完成时间调度，同时记录用于输出 token 时序的物理完成估计。它不仿真每一次调度器迭代或 token 事件。

## 解析规模估算 {#analytical-sizing}

Fleet Sim 从每个池的归一化 CDF 采样 3,000 个服务时间，并计算：

- 平均服务时间；
- 变异系数平方（`CV^2`）；
- 每个 profile 单元的 KV-cache 槽位数；以及
- 平均 prefill 时间。

它按如下方式估计 profile 吞吐：

```text
mu_gpu = slots / mean_service_time
```

池等待时间使用 Erlang-C 概率和 Kimura M/G/c P99 近似。每个 KV 槽被视为一个排队服务器。所选池规模必须同时满足 P99 等待目标和默认利用率上限 0.85。报告的解析 TTFT 将平均 prefill 时间加到估计的 P99 等待上。

该近似假设平稳泊松到达流和独立的服务时间分布。基于 token 的路由、突发流量和共享 GPU 调度可能违反这些假设，因此选定候选应用 DES 和真实负载测试校验。

## 离散事件仿真 {#discrete-event-simulation}

DES 在到达和建模完成之间推进。每个池包含相同实例，默认使用最短队列。实例仅在同时拥有空闲逻辑槽和足够 KV 块时才准入请求。

若 KV 准入会超出块预算，实例会抢占最长的活跃请求并将其放回队列头部。再次准入时模型会重启服务计算；它不保留逐 token 进度。

主要 DES 指标是 P50/P99 TTFT、P99 队列等待、完成吞吐、SLO 合规率和平均利用率。百分位和 SLO 合规率仅对**已完成请求**计算。务必比较 `total_completed` 与请求的仿真数量；否则队列拒绝或不完整排空会使延迟百分位看起来好于整个到达总体。

CLI 目前不会从报告指标中去掉初始预热段。在尾延迟重要时，请运行足够请求、检查多个种子，并与稳态负载测试比较。

## 路由模型 {#routing-models}

| 路由器 | 行为 | 可用性 |
| --- | --- | --- |
| `LengthRouter` | 将请求发往最小可容纳池，或配置的短/长拆分 | CLI 和库 |
| `CompressAndRouteRouter` | 压缩 `(B_short, gamma * B_short]` 中的合格请求，并发送到短池 | 通过 `--gamma` 用于 `optimize`、`simulate`，以及库 |
| `SpilloverRouter` | 当短池压力越过阈值时，将短流量发往长池 | 库 |
| `LeastLoadedRouter` | 选择相对槽位而言活跃加排队负载最低的池 | 多池 CLI 和库 |
| `ModelRouter` | 将 `request.model_id` 映射到池，带回退到配置池或第一个池 | 多池 CLI 和库 |
| `SemanticRouter` | 调用用户提供的分类器函数，输出未知时回退 | 库；CLI JSON 可以选择它，但不提供分类器函数 |
| `RandomRouter` | 均匀选择一个池 | CLI 基线和库 |

`compare-routers` 仅比较长度、固定 `gamma=1.5` 的压缩后路由，以及随机路由。它不会调用在线的 vLLM Semantic Router。

### 压缩后路由假设 {#compress-and-route-assumptions}

DES 将 `prose`、`rag` 和 `mixed` 视为安全类别，将 `code` 视为不安全。合格输入会被缩短到短池预算。解析优化器用有效压缩概率（默认 `0.75`）处理边界带中的流量。

这些是模拟器假设。它们不测量语义保持、类别错误或真实压缩器的延迟。请单独验证质量，并用工作负载证据替换安全比例。

## 阈值搜索 {#threshold-search}

`pareto` 把每个具有非平凡短流量比例的非末端 CDF 断点作为候选 `B_short`。对每个点，它解析地为两个池做规模估算，并标记在建模成本和最差池 P99 上均未被支配的点。

输出是一组权衡，不是自动的生产阈值。阈值还应能抵抗估计误差、工作负载漂移和短池的上下文容量。

## 拆分 prefill 和 decode {#disaggregated-prefill-and-decode}

`disagg` 命令估计独立的 prefill 和 decode 吞吐，然后扫描 worker 数量。系统吞吐取两个阶段速率中的较小值。它应用固定退化因子（prefill 为 `0.90`，decode 为 `0.92`），并将基础 prefill 时间乘以 `1.80` 作为 TTFT 估计。

这些常量是内置假设。模型不仿真 KV 传输大小、拓扑、网络争用、放置或阶段队列。在判定拆分设计满足 TTFT 或 TPOT 目标之前，请使用测得的传输和阶段行为。

## 已知边界 {#known-boundaries}

Fleet Sim 目前不建模：

- token 级连续批处理或精确的 vLLM 调度器行为；
- 张量并行集合通信、网络拓扑、主机开销或内核启动效应；
- 前缀缓存命中分布、推测解码、量化内核或 adapter 切换（除非间接体现在已校准 profile 中）；
- CLI CDF 工作流中的非泊松突发；
- 模型答案质量或路由分类器错误；
- 在线故障、修复队列、发布容量或自动扩缩反应时间；
- 将拒绝计入 SLO 合规分母；或
- 当请求超过每个已配置池的最大上下文时的硬拒绝——长度路由器会把它发往最大的池。

用模拟器比较显式假设。用生产形态的负载测试接受部署。

基于该性能模型的能耗计算，请继续阅读[功耗模型](./power-model)。

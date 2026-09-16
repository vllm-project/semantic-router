---
title: 容量规划
translation:
  source_commit: "33349fdab9ad294da19ebd11588f8adbe8771b4a"
  source_file: "docs/fleet-sim/use-cases.md"
  outdated: false
---

# 容量规划工作流

Fleet Sim 最适合作为一系列问题来使用。先从工作负载开始，校准一条基线，只有在基线说明为何需要时，再引入新的池或策略。

下面的命令演示工作流。它们有意不包含示例节省或推荐 GPU 数量：这些值取决于你提供的工作负载和 profile 假设。

## 1. 描述工作负载 {#1-describe-the-workload}

在为机队做规模估算之前，收集：

- 提示词和输出 token 数；
- 请求时间戳或到达率范围；
- 延迟目标及其测量方式；
- 若流量已使用语义路由，所选模型或路由；以及
- 平均速率会掩盖的突发、日周期和故障时段。

CLI 接受**总 token** 的累积分布：

```json
{
  "cdf": [
    [512, 0.25],
    [2048, 0.70],
    [8192, 0.95],
    [32768, 1.0]
  ]
}
```

阈值必须递增，累积比例应以 `1.0` 结束。从 CDF 采样的 CLI 工作负载假定 80% 输入 token、20% 输出 token 以及泊松到达。如果该拆分或到达过程不像你的流量，请仅把结果当作敏感性研究，或使用 Python 库的 `TraceWorkload` 与 `Fleet` API。

独立 HTTP 服务接受 JSONL 和 CSV 轨迹上传，并汇总提示词、输出、到达和路由分布。对仿真作业，它会把上传的长度转换成 CDF 并生成泊松到达；它不会回放原来的时间戳或路由标签。上传前请去掉提示词文本和用户标识，因为只需要数值规划字段。

## 2. 建立固定机队基线 {#2-establish-a-fixed-fleet-baseline}

当你已经知道当前短池和长池数量时，使用 `simulate`：

```bash
vllm-sr-sim simulate \
  --cdf data/azure_cdf.json \
  --lam 200 \
  --slo 500 \
  --b-short 6144 \
  --n-s 24 \
  --n-l 8 \
  --n-req 30000 \
  --out baseline.json
```

分别查看每个池。机队范围的百分位可能掩盖一个队列很深的小长池。至少检查：

- 按池的 P99 TTFT 和 P99 队列等待；
- 完成率和 SLO 合规比例；
- 平均利用率；
- 抢占或未完成的请求；以及
- 对随机种子和请求数量的敏感性。

第一次运行不是校准。调整 profile，使建模 TTFT、吞吐、并发和 KV 容量与同一模型和服务配置的受控负载测试匹配。

## 3. 寻找短/长阈值 {#3-find-a-shortlong-threshold}

只有当各池的服务特征有实质差异，且工作负载在拆分两侧都有足够流量时，双池设计才有帮助。

用 `pareto` 把 CDF 断点评估为候选阈值：

```bash
vllm-sr-sim pareto \
  --cdf data/lmsys_cdf.json \
  --lam 200 \
  --slo 500 \
  --gpu-short a100 \
  --gpu-long h100 \
  --out threshold-sweep.json
```

按运营原因选择阈值，不要只看最低建模成本：

- 两个池都应留下有用余量；
- 短池必须能服务路由到它的每一条请求；
- 路由分类和 token 估计在边界附近必须稳定；
- 若各池服务不同模型，答案质量必须仍然可接受；以及
- 小的工作负载偏移不应导致机队数量大幅跳跃。

选定候选后，用得到的数量运行 `simulate`，不要只依赖解析行。

## 4. 搜索双池机队 {#4-search-a-two-pool-fleet}

`optimize` 用解析方式为池做规模估算，并可对最低成本候选做 DES 校验：

```bash
vllm-sr-sim optimize \
  --cdf data/azure_cdf.json \
  --lam 200 \
  --slo 500 \
  --b-short 6144 \
  --gpu-short a100 \
  --gpu-long h100 \
  --verify-top 3 \
  --n-sim-req 30000 \
  --out candidates.json
```

搜索还会扫描压缩后路由的 `gamma` 带。该模型假定一部分边界流量可以安全缩短。CLI 不知道你真实的类别组合或压缩质量。在代表性请求上测量压缩资格、延迟和任务质量之前，把 `gamma > 1` 带来的任何收益视为有条件的。

## 5. 规划流量增长 {#5-plan-for-traffic-growth}

用 `whatif` 找出池需要再增加一个容量单元或失去延迟余量的速率：

```bash
vllm-sr-sim whatif \
  --cdf data/azure_cdf.json \
  --lam-range 100 150 200 300 400 \
  --slo 500 \
  --b-short 6144 \
  --gpu-short a100 \
  --gpu-long h100 \
  --out arrival-sweep.json
```

使用多种工作负载形态，不要只扫多种速率。即使每秒总请求数不变，长上下文占比变化也可能压垮长池。为故障、发布和突发吸收加入自己的运营储备；CLI 优化不会从机队遥测推断该储备。

## 6. 比较路由策略 {#6-compare-routing-policies}

对固定双池机队，`compare-routers` 在同一组生成到达上运行三种 CLI 策略：长度路由、`gamma=1.5` 的压缩后路由，以及均匀随机路由。

```bash
vllm-sr-sim compare-routers \
  --cdf data/agent_heavy_cdf.json \
  --lam 200 \
  --slo 500 \
  --b-short 6144 \
  --n-s 24 \
  --n-l 8 \
  --n-req 30000
```

该命令是受控的模拟器比较，不是仓库中所有路由器实现的基准。尤其是，它不运行在线语义分类器，也不包含其延迟和错误。

对已路由的生产数据，为每个所选模型或池推导一条 CDF 和到达比例，然后使用带每池 `workloads` 的 `model` 拓扑。这保留观察到的汇总路由组合，而不假装重新运行分类器或复现请求顺序。

## 7. 建模超过两个池 {#7-model-more-than-two-pools}

对特定模型的池或任意异构拓扑使用 `simulate-fleet`。最小 JSON 文件如下：

```json
{
  "pools": [
    {
      "id": "general",
      "gpu": "a100",
      "n_gpus": 12,
      "max_ctx": 8192
    },
    {
      "id": "long-context",
      "gpu": "h100",
      "n_gpus": 8,
      "max_ctx": 65536
    }
  ],
  "router": "length"
}
```

```bash
vllm-sr-sim simulate-fleet fleet.json \
  --cdf data/azure_cdf.json \
  --lam 200 \
  --slo 500 \
  --n-req 30000 \
  --out fleet-result.json
```

支持的 CLI JSON 路由器值为 `length`、`model`、`semantic`、`random` 和 `least_loaded`。对模型路由的 CLI 研究，省略 `--cdf` 并为每个池提供 `workloads` 条目；传入 `--cdf` 会覆盖这些每池流。请验证回退池，以免缺失的模型名称悄悄扭曲程序化研究。

## 8. 评估拆分 prefill 和 decode {#8-evaluate-disaggregated-prefill-and-decode}

当 prefill 和 decode 需要不同容量或硬件时，拆分才相关。它也会引入 KV 传输、网络和协调成本。

```bash
vllm-sr-sim disagg \
  --cdf data/azure_cdf.json \
  --lam 200 \
  --slo-ttft 500 \
  --slo-tpot 100 \
  --gpu-prefill h100 \
  --gpu-decode a100 \
  --mean-isl 2048 \
  --mean-osl 256 \
  --out disagg.json
```

优化器使用内置退化和传输校正因子。那些是假设，不是你的网络或拆分运行时的测量。在选择 prefill/decode 比例之前，用部署测试替换该结论。

## 9. 仅在性能校准后再加入功耗 {#9-add-power-only-after-performance-calibration}

`tok-per-watt` 和 `grid-flex` 建立在同一性能 profile 加上功耗曲线之上。在两者都已针对目标模型测量之后，它们对比较场景有用。

```bash
vllm-sr-sim tok-per-watt \
  --cdf data/azure_cdf.json \
  --lam 200 \
  --slo 500 \
  --gpus h100 a100
```

```bash
vllm-sr-sim grid-flex \
  --cdf data/azure_cdf.json \
  --lam 200 \
  --n-gpus 32 \
  --gpu h100 \
  --slo 500 \
  --verify-des 20000 \
  --out flex-curve.json
```

不要把内置的 A10G 与 A100/H100 `tok-per-watt` 输出当作仅硬件比较：捆绑的 profile 代表不同的模型规模和并行布局。即使 profile 标注为同一模型，也要在相同测试条件下校准整机功耗、吞吐和输出长度。

`grid-flex` 估计并发上限降低建模功耗时会发生什么。它不会对 vLLM 施加上限，也不参与需求响应系统。

## 使用结果之前 {#before-using-a-result}

每次决策都记录以下内容：

- 工作负载来源和观察窗口；
- 模型、精度、张量并行、GPU SKU 和 vLLM 设置；
- profile 常量及其测量方式；
- 模拟器版本、命令、种子和请求数量；
- 解析和 DES 结果，包括任何不一致；以及
- 接受或拒绝该设计的负载测试结果。

这样可以把一次模拟器运行变成可审阅的容量假设，而不是没有依据的性能主张。

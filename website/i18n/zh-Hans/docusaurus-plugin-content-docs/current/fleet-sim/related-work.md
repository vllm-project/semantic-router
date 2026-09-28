---
title: 研究背景
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/fleet-sim/related-work.md"
  outdated: false
---

# 研究背景

Fleet Sim 是机队规划工具。它借用排队、推理仿真、异构服务以及拆分 prefill/decode 研究中熟悉的抽象，但并不复现任何单一研究系统。

本页帮助选择合适的工具层级。它有意避免把论文基准数字抄进产品指导；性能主张都绑定到各论文自己的工作负载和评估环境。

## Fleet Sim 的位置 {#where-fleet-sim-fits}

| 层级 | 主要问题 | Fleet Sim 覆盖范围 |
| --- | --- | --- |
| 服务引擎 | 单个副本应如何批处理和调度 token？ | 通过已校准 profile 表示，不以内核保真度仿真 |
| 副本配置 | 单个副本应使用哪些张量/流水线并行和运行时设置？ | 输入假设；`ComputedProfile` 可探索粗略敏感性 |
| 机队规划 | 需要多少池实例，流量应如何拆分？ | 主要范围 |
| 运行时控制 | 在线机队何时扩缩、溢出流量或降低负载？ | 可评估静态场景；不操作控制器 |
| 机房能耗 | 整机功耗和对电网的影响是什么？ | 仅 GPU 板级功耗估计 |

用分析器或高保真引擎模拟器校准副本，用 Fleet Sim 比较机队拓扑，再用生产负载测试接受结果。

## 相邻研究 {#adjacent-research}

### 异构机队选择 {#heterogeneous-fleet-selection}

[Mélange](https://arxiv.org/abs/2404.14527) 研究跨工作负载切片的成本感知 GPU 类型选择。其核心问题是组合哪些已测量的硬件 profile。Fleet Sim 在提供性能 profile 和成本之后，聚焦池数量、路由和排队。

共同教训是：没有工作负载规模、到达率、SLO、模型和价格，就不能给 GPU SKU 排名。因此 Fleet Sim 的内置 profile 名称应视为待替换的输入，而不是通用硬件排名。

### 逐副本与引擎仿真 {#per-replica-and-engine-simulation}

[Vidur](https://arxiv.org/abs/2405.05465) 用分析得到的算子级模型仿真 LLM 服务引擎并搜索其配置。这比 Fleet Sim 的 `W`/`H` 请求级 profile 保真度更高。

[AIConfigurator](https://arxiv.org/abs/2601.06288) 用硬件级和算子级性能信息探索模型和引擎配置。Fleet Sim 的计算 profile 有类似 roofline 的分解，但这并不使它成为 AIConfigurator 或内核数据库的已验证替代。

这类工具的测量或选定设置可以转换成 `ManualProfile`，再做机队规模估算。

### 拆分 prefill 和 decode {#disaggregated-prefill-and-decode}

[DistServe](https://arxiv.org/abs/2401.09670) 和 [Splitwise](https://arxiv.org/abs/2311.18677) 研究将 prefill 与 decode 阶段分离的系统。它们建模或测量阶段干扰、并行配置、放置和 KV 传输等细节。

Fleet Sim 的 `disagg` 命令窄得多：它在扫描 prefill 和 decode worker 数量时应用固定的阶段退化和 TTFT 校正因子。它对第一次敏感性研究有用，但不能验证网络拓扑或 KV 传输延迟。

### 自动扩缩与突发控制 {#autoscaling-and-burst-control}

[SageServe](https://arxiv.org/abs/2502.14617) 考虑感知预测的运行时容量控制，[TokenScale](https://arxiv.org/abs/2512.03416) 考虑用阶段级 token 需求对拆分推理做运行时扩缩。

Fleet Sim 不实现其中任何控制循环。其 `whatif` 输出可以帮助识别独立控制器应测试的静态速率和机队形态，但不建模启动时间、预测误差、控制延迟或在线反压。

### 功耗感知推理 {#power-aware-inference}

功耗感知服务工作（包括 [GPU-to-Grid](https://arxiv.org/abs/2602.05116)）促使研究并发上限对延迟的影响。Fleet Sim 的 `grid-flex` 命令将其表示为拟合或估计的批次与板级功耗曲线，再加上解析延迟和可选 DES 延迟。

它不测量机房功耗，也不实现需求响应。校准和报告边界见[功耗模型](./power-model)。

## 排队基础 {#queueing-foundation}

解析规模估算使用 Erlang-C 等待概率和 Kimura 风格的 M/G/c 尾部近似。随后 DES 用显式合成到达和 KV 槽准入评估选定候选。

这些模型回答的问题不同于内核或服务引擎仿真：它们在已知服务时间分布后估计机队排队。其假设（平稳泊松到达、近似独立的服务时间、逻辑 KV 槽）必须对照部署检查。

## 选择工具 {#selecting-a-tool}

| 若你需要决定…… | 从这里开始…… |
| --- | --- |
| 单个副本的批处理、调度器或并行设置 | 服务引擎基准或高保真模拟器 |
| 哪个已测量的 GPU/模型 profile 最便宜 | 异构配置搜索 |
| 短/长池数量和路由敏感性 | Fleet Sim `optimize`、`pareto` 和 `simulate` |
| prefill/decode worker 数量敏感性 | Fleet Sim `disagg`，再跟拆分系统测试 |
| 逐秒扩缩行为 | 运行时自动扩缩/控制器模型 |
| 并发对板级功耗的敏感性 | 测量后再用 Fleet Sim 功耗命令 |
| 生产容量审批 | 生产形态的负载测试 |

本表中没有任何工具可以省去把工作负载、模型、硬件、运行时和测量假设与结果一并保存。

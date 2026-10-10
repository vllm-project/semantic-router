---
title: 开放智能指数
sidebar_label: 智能指数
description: Model Arena 排名和路由质量证据背后的版本化评测层级。
translation:
  source_commit: "e56591a9cb24f073bf159927e87116ba6d278741"
  source_file: "docs/benchmarking/open-intelligence-index.md"
  outdated: false
---

# 开放智能指数 {#open-intelligence-index}

开放智能指数是 Model Hub、Model Arena 和路由共用的目录证据图，保留版本化的已报告测量和缺失覆盖。它不是 [sr-bench 1.0](sr-bench) 的实时评测分数。新的单模型/MoM 能力、成本和延迟应由 sr-bench 实测；目录记录不能证明当前部署的新结果。

## Intelligence 1.0 {#intelligence-10}

```text
Intelligence 1.0
├── General                 20%
│   └── MMLU-Pro           100%
├── Reasoning               40%
│   ├── GPQA Diamond        50%
│   └── HLE 1.0 text-only   50%
├── Coding                  20%
│   ├── LiveCodeBench v6    50%
│   └── SciCode             50%
└── Agentic                 20%
    └── Terminal-Bench 2.1 100%
```

| 能力 | 基准 | 指标 | 来源 |
| --- | --- | --- | --- |
| General | MMLU-Pro | Accuracy | [仓库](https://github.com/TIGER-AI-Lab/MMLU-Pro) · [论文](https://arxiv.org/abs/2406.01574) · [数据](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) |
| Reasoning | GPQA Diamond | Accuracy | [仓库](https://github.com/idavidrein/gpqa) · [论文](https://arxiv.org/abs/2311.12022) · [数据](https://huggingface.co/datasets/idavidrein/gpqa) |
| Reasoning | HLE 1.0 纯文本 | 冻结的 2158 题纯文本子集上的 Accuracy | [仓库](https://github.com/centerforaisafety/HLE) · [论文](https://arxiv.org/abs/2501.14249) · [数据](https://huggingface.co/datasets/cais/hle) |
| Coding | LiveCodeBench v6 | Pass@1，代码生成 | [仓库](https://github.com/LiveCodeBench/LiveCodeBench) · [论文](https://arxiv.org/abs/2403.07974) · [数据](https://huggingface.co/datasets/livecodebench/code_generation_lite) |
| Coding | SciCode | 可执行子问题分数 | [仓库](https://github.com/scicode-bench/SciCode) · [论文](https://arxiv.org/abs/2407.13168) · [数据](https://huggingface.co/datasets/SciCode1/SciCode) |
| Agentic | Terminal-Bench 2.1 | Resolved rate | [任务](https://github.com/harbor-framework/terminal-bench-2) · [数据集](https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1) · [runner](https://github.com/harbor-framework/harbor) |

基准输入、runner 和评分路径是公开的。每条被接纳的记录仍标识精确版本、配置文件、模型检查点、推理力度、harness、工具、运行条件、日期和来源。Agentic 结果比较冻结的模型与 Agent 系统，而不是孤立的模型名。
对于 HLE，已发布的 `no-tools` 标签并不能证明多模态题目已被排除。这些记录仍然可见，但只有在冻结的 2158 道纯文本题目上的显式运行才会进入 Intelligence 1.0。

## 分数与缺失数据 {#score-and-missing-data}

原始指标在聚合前规范化到 `[0, 1]`：

```text
General   = MMLU-Pro
Reasoning = 0.50 × GPQA Diamond + 0.50 × HLE
Coding    = 0.50 × LiveCodeBench + 0.50 × SciCode
Agentic   = Terminal-Bench 2.1

Intelligence 1.0 = 100 × (
    0.20 × General
  + 0.40 × Reasoning
  + 0.20 × Coding
  + 0.20 × Agentic
)
```

每个类别和 Overall 都使用 `require_all`：

- `available` 表示同一模型和推理力度下每个子项都存在；
- `partial` 有部分子项、空分数、精确覆盖率以及缺失列表；
- `missing` 没有被接纳的子项，且分数为空。

分数绝不会被填补、设为零、按已报告组件重新规范化，或从另一个检查点或推理力度借用。部分 Overall 仍可包含可用的类别分数。该模型可以出现在匹配的类别或基准排名中，并可为类别感知路由服务。

## Model Arena {#model-arena}

Arena 对同一份数据提供三个视图：

1. Overall Intelligence 排名；
2. General、Reasoning、Coding 和 Agentic 排名；
3. 六个原始基准排名。

每个模型显示最高可用推理力度结果，并在该行暴露该力度。物理模型和虚拟模型使用相同的排名逻辑。URL 参数保留层级、所选能力或基准、模型范围和所选模型，因此每个视图都可分享。

## 路由质量证据 {#routing-quality-evidence}

为决策选择合适的能力，而不是强迫每条路由都使用 Overall：

```yaml
algorithm:
  type: multi_factor
  multi_factor:
    quality:
      index: vllm-sr/coding@1.0.0
      on_missing: exclude
    weights:
      quality: 0.4
      latency: 0.2
      cost: 0.2
      load: 0.2
```

`exclude` 只接纳具有可用精确力度结果的候选。`disable_quality` 保留完整候选池，但若任一候选缺少所选指数，则从整个比较中移除质量。运维 SLO、延迟、成本和负载继续生效；不会做候选本地的权重更改。

`quality.min_coverage` 可以要求比指数自身缺失数据策略更多的证据，`quality.min_score` 是硬质量下限。这让部署可以在有意部分的运维指数上路由，而不削弱完整情形的 1.0 Overall 契约。YAML 契约见[自定义评测](custom-evaluations)，Balanced、Accuracy-first 和 Cost-first 目标见 [Multi Factor](../tutorials/algorithm/selection/multi-factor)。

## 新测量

通过 [sr-bench](sr-bench) 的真实路由入口评测 MoM。模型池分数或逐题 oracle 不能证明路由策略质量。发布运维评测记录前保留数据、模型、配置和价格身份。

目录图由 `config/catalog/resources/indices.yaml` 定义。改变权重或基准契约需要新的版本身份。

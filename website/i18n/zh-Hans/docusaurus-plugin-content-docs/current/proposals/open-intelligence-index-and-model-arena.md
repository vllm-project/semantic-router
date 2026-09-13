---
title: Open Intelligence 架构
description: 连接可复现评估、统一 Model Arena 和证据感知路由的版本化能力层级。
created: 2026-09-09
status: Implemented
translation:
  source_commit: "e56591a9cb24f073bf159927e87116ba6d278741"
  source_file: "docs/proposals/open-intelligence-index-and-model-arena.md"
  outdated: false
---

> **状态：** 已实现 · **跟踪：** [#3577](https://github.com/vllm-project/semantic-router/issues/3577)

## 决策 {#decision}

vLLM Semantic Router 为三类消费者使用同一版本化评估图：

1. Model Hub 保留每一条有效基准结果及其来源；
2. Arena 从该图渲染 Overall、能力和基准排名；
3. 路由显式选择 Overall 或能力指数，并只消费候选精确模型和推理力度的 `available` 结果。

物理模型和虚拟模型遵循同一契约。缺失数据保持可见，但从不估计、悄悄重加权，或跨模型变体复制。

## 能力路线图 {#capability-roadmap}

版本 1.0 是活动的文本智能契约。其 HLE 叶子是冻结的 2,158 题 2025 年 5 月纯文本子集；带图像的 HLE 题目从不进入 1.0 分数。版本 1.5 和 2.0 是设计锁定，不是活动目录指数；各自仅在其基准修订、运行器、评分器和比较队列冻结后激活。

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

```text
Intelligence 1.5
├── General                 20%
│   └── MMLU-Pro           100%
├── Reasoning               40%
│   ├── GPQA Diamond        50%
│   └── HLE 1.0 text-only   50%
├── Coding                  20%
│   ├── LiveCodeBench v6    50%
│   └── SciCode             50%
└── Agentic                 20%
    ├── Terminal-Bench 4.0  50%
    └── SWE-bench Live      50%  (immutable snapshot)
```

```text
Intelligence 2.0
├── General                 15%
│   └── MMLU-Pro           100%
├── Reasoning               30%
│   ├── GPQA Diamond        50%
│   └── HLE 1.0 text-only   50%
├── Coding                  15%
│   ├── LiveCodeBench v6    50%
│   └── SciCode             50%
├── Agentic                 15%
│   ├── Terminal-Bench 4.0  40%
│   ├── SWE-bench Live      40%  (immutable snapshot)
│   └── CyberGym Level 1    20%
├── Multimodal              15%
│   ├── MMMU-Pro            50%
│   ├── MathVista           25%
│   └── OCRBench v1         25%
└── Safety                  10%  + eligibility gate
    ├── HarmBench Robustness 50%
    └── XSTest Safe Helpfulness 50%
```

CyberGym 属于 **Agentic / Security Engineering**：它测量可执行环境中的自主漏洞复现。它不测量模型是否行为安全，因此不属于 Safety。

## 基准契约 {#benchmark-contract}

### 1.0 中活动 {#active-in-10}

| 能力 | 基准 | 测量内容 | 公开来源 |
| --- | --- | --- | --- |
| General | MMLU-Pro | 宽泛多领域知识与推理 | [仓库](https://github.com/TIGER-AI-Lab/MMLU-Pro)，[论文](https://arxiv.org/abs/2406.01574)，[数据](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) |
| Reasoning | GPQA Diamond | 研究生级科学推理 | [仓库](https://github.com/idavidrein/gpqa)，[论文](https://arxiv.org/abs/2311.12022)，[数据](https://huggingface.co/datasets/idavidrein/gpqa) |
| Reasoning | Humanity's Last Exam（纯文本） | 前沿、跨领域封闭答案推理，覆盖冻结的 2,158 道纯文本题 | [仓库](https://github.com/centerforaisafety/HLE)，[论文](https://arxiv.org/abs/2501.14249)，[数据](https://huggingface.co/datasets/cais/hle) |
| Coding | LiveCodeBench v6 | 近期竞赛代码生成 | [仓库](https://github.com/LiveCodeBench/LiveCodeBench)，[论文](https://arxiv.org/abs/2403.07974)，[数据](https://huggingface.co/datasets/livecodebench/code_generation_lite) |
| Coding | SciCode | 可执行科学编程问题 | [仓库](https://github.com/scicode-bench/SciCode)，[论文](https://arxiv.org/abs/2407.13168)，[数据](https://huggingface.co/datasets/SciCode1/SciCode) |
| Agentic | Terminal-Bench 2.1 | 终端环境中的长程工作 | [任务](https://github.com/harbor-framework/terminal-bench-2)，[数据集](https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1)，[运行器](https://github.com/harbor-framework/harbor) |

全部六个都有公开输入、可执行评估代码和公开评分路径。每条目录记录仍固定精确基准修订、配置文件、检查点、推理力度、工具链、工具、运行条件、日期和来源。
对 HLE，仅有 `no-tools` 不能证明已排除带图像题目：此类已发布结果在 `published-no-tools` 下保持可见，但没有显式的 2,158 题纯文本协议就不能进入 Intelligence 1.0。
终端和智能体基准是模型和冻结智能体工具链的联合测量；仅模型名称从不标识此类结果。

### 为 1.5 和 2.0 保留 {#reserved-for-15-and-20}

| 版本 | 基准 | 激活要求 | 公开来源 |
| --- | --- | --- | --- |
| 1.5 | Terminal-Bench 4.0 | 固定 `v4.0.0` 任务、Harbor 版本、智能体、限制、重试和环境 | [仓库](https://github.com/harbor-framework/terminal-bench)，[发布](https://github.com/harbor-framework/terminal-bench/releases/tag/v4.0.0)，[数据集](https://huggingface.co/datasets/harborframework/terminal-bench) |
| 1.5 | SWE-bench Live | 固定一个不可变已验证快照、任务 ID、镜像、评分器、智能体和重试策略 | [仓库](https://github.com/microsoft/SWE-bench-Live)，[论文](https://arxiv.org/abs/2505.23419)，[数据](https://huggingface.co/SWE-bench-Live) |
| 2.0 | CyberGym Level 1 | 固定全部 Level-1 任务、环境资产、智能体、预算和二进制验证器 | [仓库](https://github.com/sunblaze-ucb/cybergym)，[论文](https://arxiv.org/abs/2506.02548)，[数据](https://huggingface.co/datasets/sunblaze-ucb/cybergym) |
| 2.0 | MMMU-Pro | 固定标准多模态拆分和评估器 | [仓库](https://github.com/MMMU-Benchmark/MMMU)，[论文](https://arxiv.org/abs/2409.02813)，[数据](https://huggingface.co/datasets/MMMU/MMMU_Pro) |
| 2.0 | MathVista | 固定公开 `testmini` 拆分、提示词、提取和评分器 | [仓库](https://github.com/lupantech/MathVista)，[论文](https://arxiv.org/abs/2310.02255)，[数据](https://huggingface.co/datasets/AI4Math/MathVista) |
| 2.0 | OCRBench v1 | 固定 1,000 个公开项和确定性 v1 评分器；不要用私有测试 v2 替代 | [仓库](https://github.com/qywh2023/OCRbench)，[论文](https://arxiv.org/abs/2305.07895)，[数据](https://huggingface.co/datasets/echo840/OCRBench) |
| 2.0 | HarmBench | 固定行为、攻击、生成预算和开放分类器 | [仓库](https://github.com/centerforaisafety/HarmBench)，[论文](https://arxiv.org/abs/2402.04249)，[数据](https://huggingface.co/datasets/walledai/HarmBench) |
| 2.0 | XSTest | 固定公开提示词集和开放安全合规评分器 | [仓库](https://github.com/paul-rottger/xstest)，[论文](https://arxiv.org/abs/2308.01263) |

此处“开放”意味着独立贡献者可以访问输入、运行评估，并用固定版本复现分数。激活前，每个产物记录许可证和再分发边界；没有可运行输入或评分器的公开排行榜不够。

## 层级分数 {#hierarchical-score}

每个原始指标归一化到 `[0, 1]`。叶子能力是其基准的加权平均；Overall 是能力分数的加权平均。对 1.0：

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

`require_all` 应用于每个节点。仅当同一模型和推理力度的全部叶子都可用时，能力才可用；仅当全部四个能力都可用时，Overall 才可用。否则结果为 `partial` 或 `missing`，其分数为 null，覆盖率和缺失组件保持显式。没有零/均值填补、代理分数、部分重归一化或跨力度借用。

这并不使不完整证据无用。缺失 General 的模型仍可以有可用的 Coding 分数，并参与 Coding 排名或编码特定路由。它只是不能声称可比较的 Overall 分数。

## 运营方自有证据 {#operator-owned-evidence}

发布证据和部署本地证据使用同一类型化图。运营方在顶层 `evaluation` 下定义新基准语义、指数 DAG 和模型关联测量。内置基准无需重新声明。

运营方资源 ID 必须命名空间化并版本化。它们不能遮蔽内置基准或指数。定义固定配置文件、指标范围、方向、归一化、组件权重，以及一种显式缺失数据策略：

- `require_all` 仅在每个组件都存在时产生分数；
- `require_coverage` 在声明的覆盖率阈值之后产生分数；
- `reported_only` 从任何已报告组件产生分数。

后两种策略是有意的运营方指数语义，不是隐式填补。每个结果仍暴露其覆盖率。路由可以施加比指数定义更严格的 `quality.min_coverage`。未声明基准的记录保留在 `evaluation.records[]` 下，但在声明其语义之前不能进入指数。

## 物理模型与虚拟模型 {#physical-and-virtual-models}

虚拟模型通过冻结端点在同一套件的每个任务上评估。回执额外记录配方修订、每任务路由、失败、token、延迟和成本。其智能分数从不从成员模型分数、路由份额或预言选择组装。

## Arena {#arena}

两个 Hub 表面从同一目录渲染三个独立视图：

- **Overall** — 一个完整案例 Intelligence 排行榜；
- **Capabilities** — General、Reasoning、Coding 和 Agentic 排行榜；
- **Benchmarks** — 六个原始 1.0 基准排行榜。

每个视图使用竞赛排名，暴露所选推理力度，同等对待物理和虚拟模型，并支持可分享的范围和层状态。公开 Hub 还在其 URL 中保留目录过滤器和模型详情。不完整模型出现在其证据有效的地方，而不是出现在伪造的 Overall 排名中。

基准是 Arena 的第三层。先前独立的基准浏览器已从公开 Hub 和控制面板移除，以免同一证据的两种呈现之间排名、过滤器和 URL 状态分歧。

生成的快照当前包含 101 张模型卡片和 1,510 条评估记录。具有可用 1.0 结果的唯一模型为：General 35、Reasoning 86、Coding 24、Agentic 77，以及 Overall 21。因此 Overall 达到了初始 20 模型目标，同时保持严格完整性。新的 GLM-5.3 和 Qwen3.8 卡片即使因缺失叶子而不能获得 Overall 资格，仍在其支持的能力和基准视图中可见。

## 路由 {#routing}

`multi_factor` 可以选择任何版本化 Overall、能力或运营方指数。它将硬资格与优化目标分开。

平衡路由使用归一化权重：

```yaml
algorithm:
  type: multi_factor
  multi_factor:
    quality:
      index: vllm-sr/coding@1.0.0
      on_missing: exclude
      min_coverage: 1.0
    weights:
      quality: 0.4
      latency: 0.2
      cost: 0.2
      load: 0.2
```

Accuracy-first 和 Cost-first 使用同一通用字典序引擎，而不是产品特定分支：

```yaml
# Accuracy-first: keep models within 3% of the best quality, then minimize cost.
objective:
  strategy: lexicographic
  priorities:
    - {factor: quality, tolerance: 0.03}
    - {factor: cost, tolerance: 0.05}
    - {factor: latency, tolerance: 0.05}
```

```yaml
# Cost-first: enforce a quality floor, then choose within the cheapest band.
quality:
  index: vllm-sr/intelligence@1.0.0
  on_missing: exclude
  min_coverage: 1.0
  min_score: 65
objective:
  strategy: lexicographic
  priorities:
    - {factor: cost, tolerance: 0.05}
    - {factor: quality, tolerance: 0.03}
    - {factor: latency, tolerance: 0.05}
```

质量查找使用候选的精确推理力度，并只接受 `status: available`：

- `exclude` 移除缺失该指数的候选。若没有剩余，应用现有 `on_no_candidates` 策略；`fail` 返回 HTTP 503，而不回退到未评估候选。
- `disable_quality` 保留每个候选；若任何候选缺少所选指数，选择器对整个候选池禁用质量，并只使用延迟、成本、负载和配置的 SLO。它从不对一个模型与另一个模型施以不同重加权。

通用决策可以选择 `vllm-sr/intelligence@1.0.0`；已分类的编码决策可以选择 `vllm-sr/coding@1.0.0`。未来基准作为版本化叶子添加，并组合进新的能力/指数版本，而不向路由器添加基准特定分支。

成本使用当前输入 token 估计和请求的最大输出 token 预算，以及分开的输入/输出价格。硬 SLO 和质量下限在任一目标之前运行。因此未来的 Safety-first 和 Cybersecurity-first 配方添加策略/资格门并选择相关版本化指数；它们不需要另一选择算法。

## 后续收尾与路线图 {#follow-up-closure-and-roadmap}

该实现在配置、运行时、控制面板、公开 Website 和文档上关闭当前契约：

- 自定义基准定义、指数 DAG、模型关联记录、校验、规范往返和精确力度路由是一条路径；
- Balanced、Accuracy-first 和 Cost-first 是同一选择器的配置；
- Overall、Capabilities 和 Benchmarks 是两个 Hub 表面上仅有的 Arena 层，带可分享的 Arena 状态和完整的公开 Hub URL 状态；
- 配置、评估和算法指南发布相同字段和缺失数据行为。

以下工作仍是版本化后续，不是 1.0 中的隐藏行为：

- **MoM：** 在当前 Balanced 配方旁发布 Cost-first、Accuracy-first、Safety-first 和 Cybersecurity-first 虚拟模型，然后从在线结果加上评估、推理和研究证据优化冻结配方；
- **Evaluation：** 仅在其开放套件固定后激活 1.5 和 2.0，扩展能力指数和 Arena，并在迁移期间保持旧基准/指数版本可查询；
- **Inference：** 通过同一模型/提供商/证据契约接纳额外当前模型，包括 Kimi K3 和 DSV4 Flash Vision Exp；
- **Research：** 将路由计划作为九条具体轨道推进：
 1. 从大型开放和封闭模型池中选择模型；
 2. 切换时跨模型复用 KV cache；
 3. 确定切换模型时保留哪些上下文；
 4. 通过缓存并复用 LLM 推理轨迹来改进对相似请求的 SLM；
 5. 学习失败模式，用于 SLM 与 LLM 之间自我改进的蒸馏路由；
 6. 从模型内部潜在统计路由；
 7. 路由到模型时转发嵌入或其他超出提示词的表示；
 8. 使用模型协作做测试时扩展；以及
 9. 构建带路由记忆的自我改进路由器。

## 版本化与迁移 {#versioning-and-migration}

基准身份、配置文件和指数定义不可变。Terminal-Bench 4.0 在 Intelligence 1.5 中替换 2.1；它们的分数从不在同一个 Agentic 节点中共存。在新版本构建完整的物理与虚拟队列期间，旧评估和指数版本保持可查询。仅在替换契约可复现且覆盖足够之后，目录默认值才变更。

机器可读的 1.0 图位于 `config/catalog/resources/indices.yaml`；目录生成是运行时和 UI 投影使用的唯一评分实现。

---
title: 自定义评测
description: 通过一个规范 evaluation 分区添加版本化基准定义、模型关联记录和路由指数。
translation:
  source_commit: "2b7519a84aec96963b02a3534e82908beba33f76"
  source_file: "docs/benchmarking/custom-evaluations.md"
  outdated: false
---

# 自定义评测 {#custom-evaluations}

运维评测数据只有一个规范位置：

```yaml
evaluation:
  benchmarks: [] # definitions not built into this release
  indices: []    # operator-owned index DAGs
  records: []    # measurements linked to Model Card identities
```

内置数据在生成的目录快照中使用同一逻辑模型：基准定义、指数定义和与模型关联的评测记录是分开的集合。将测量值放在 Model Card 之外，可避免重复身份元数据，并让一个模型拥有多次基准运行、配置文件和推理力度。

发布所属源按职责拆分在 `config/catalog` 下：
Model Card 在 `resources/models/`，基准契约在
`resources/benchmarks.yaml`，测量值在 `resources/evaluations/`，指数 DAG 在 `resources/indices.yaml`。生成会校验并嵌入一份不可变快照。用户 `evaluation` 数据为本地部署扩展该快照；它不会复制或覆盖发布所属记录。

## 为自定义模型添加证据 {#add-evidence-for-a-custom-model}

对于自定义模型，省略 `providers.models[].catalog`。此时 provider 模型名就是其规范 Model Card 身份，也被 `evaluation.records[].model` 使用：

```yaml
version: v0.3

providers:
  models:
    - name: private-chat
      provider_model_id: private-chat-awq
      api_format: openai
      backend_refs:
        - name: primary
          provider: vllm
          endpoint: model-gateway.example:8000
          protocol: http

evaluation:
  records:
    - model: private-chat
      benchmark: livecodebench/livecodebench@6.0.0
      benchmark_profile: independent-code-generation
      reasoning_effort: high
      metrics: {pass_at_1: 0.61}
      measured_at: 2026-09-09
      source: https://benchmarks.example/runs/private-chat-lcb6

routing:
  modelCards:
    - name: private-chat
      display_name: Private Chat
      capabilities: [chat, tools, coding]
```

对于由内置卡片支撑的别名，将 `model` 设为规范的 `providers.models[].catalog` ID，而不是面向请求的别名。因此同一检查点的多个别名共享同一份模型证据。

该配置是路由证据，与 [sr-bench 1.0](sr-bench) 的执行账本分开。sr-bench 固定自身的目标、数据和价格身份。应先审查完整 live 报告，再显式发布运维评测记录；目录值不能代替当前部署的新测量。

## 定义基准和指数 {#define-a-benchmark-and-index}

内置基准 ID 无需重新声明。仅为运维自有基准定义语义：

```yaml
evaluation:
  benchmarks:
    - id: acme/clinical-reasoning@1.0.0
      display_name: ACME Clinical Reasoning
      domain: medical_reasoning
      source: https://benchmarks.example/clinical-reasoning/v1
      default_profile: heldout
      profiles:
        - id: heldout
          display_name: Held-out set
          description: Frozen v1 cases with deterministic scoring.
      metrics:
        - id: accuracy
          unit: proportion
          direction: higher_is_better
          range: [0, 1]

  indices:
    - id: acme/clinical-quality@1.0.0
      display_name: Clinical Quality
      aggregation: weighted_mean
      scale: [0, 100]
      missing: {policy: require_coverage, minimum: 0.5}
      domains: {medical_reasoning: 0.5, scientific_reasoning: 0.5}
      components:
        - benchmark: acme/clinical-reasoning@1.0.0
          benchmark_profile: heldout
          metric: accuracy
          weight: 0.5
          normalization: {type: identity}
        - benchmark: idavidrein/gpqa-diamond@1.0.0
          benchmark_profiles: [independent-standard, published-standard]
          metric: accuracy
          weight: 0.5
          normalization: {type: identity}

  records:
    - model: private-chat
      benchmark: acme/clinical-reasoning@1.0.0
      benchmark_profile: heldout
      reasoning_effort: high
      metrics: {accuracy: 0.74}
```

资源 ID 必须小写、带命名空间且版本化。运维定义不能遮蔽内置基准或指数。当任务、配置文件、评分器、指标含义、规范化或权重变化时，发布新 ID。

未声明的带命名空间基准记录会被保留，并在规范导出中往返，但在定义其指标范围、方向和配置文件之前不能进入指数。

## 选择缺失数据语义 {#choose-missing-data-semantics}

| 策略 | 指数何时可用 | 聚合 |
| --- | --- | --- |
| `require_all` | 覆盖率为 1.0 | 所有已声明组件权重 |
| `require_coverage` | 覆盖率达到 `minimum` | 已报告组件权重 |
| `reported_only` | 任一组件存在 | 已报告组件权重 |

没有任何策略会编造缺失分数。部分策略仅在指数定义显式要求时才重新规范化已报告权重；覆盖率和缺失组件仍然可见。

## 按指数路由 {#route-on-the-index}

```yaml
algorithm:
  type: multi_factor
  multi_factor:
    quality:
      index: acme/clinical-quality@1.0.0
      on_missing: exclude
      min_coverage: 0.5
      min_score: 60
    objective:
      strategy: lexicographic
      priorities:
        - {factor: quality, tolerance: 0.03}
        - {factor: cost, tolerance: 0.05}
```

Router 使用候选精确推理力度的 `available` 结果，然后应用路由级覆盖率和分数门槛。平衡、质量优先和成本优先目标见 [Multi Factor](../tutorials/algorithm/selection/multi-factor)。

在控制面板中打开 **Build → Models → Evaluation Records** 以添加或编辑记录。基准和指数定义仍保持为显式 YAML，因为更改其语义会创建版本化评分契约，而不是普通 Model Card 元数据。

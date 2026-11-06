---
title: 分类请求
description: 按请求的领域、是否需要事实核查、用户反馈和请求的输出模态路由，或按你自己分类器的标签路由。
translation:
  source_commit: "439c22531380dfd51abf2ce33ce6dc4db87b1fb7"
  source_file: "docs/model-runtime/guides/classify.md"
  outdated: true
is_mtpe: true
---

# 分类请求 {#classify-requests}

分类器给每个请求贴个标签：它的主题、要不要事实核查、用户对上一条答案满不满意、或者他们是不是在要图。路由就按这些标签匹配。

| 信号 | 模型 | 标签 |
| --- | --- | --- |
| [`domain`](../../tutorials/signal/learned/domain.md) | Vela 1.0 Domain | 14 个领域：`biology`、`business`、`chemistry`、`computer science`、`economics`、`engineering`、`health`、`history`、`law`、`math`、`other`、`philosophy`、`physics`、`psychology` |
| [`fact_check`](../../tutorials/signal/learned/fact-check.md) | Vela 1.0 FactCheck | `FACT_CHECK_NEEDED`、`NO_FACT_CHECK_NEEDED` |
| [`user_feedback`](../../tutorials/signal/learned/user-feedback.md) | Vela 1.0 Feedback | satisfied、need clarification、wrong answer、want different、no feedback |
| [`modality`](../../tutorials/signal/learned/modality.md) | Vela 1.0 Modality | `AR`（文本）、`DIFFUSION`（图像）、`BOTH` |
| [`classifier`](../../tutorials/signal/learned/systemone.md) | 你自己的模型 | 你的标签 |

表里列的是专用模型。没有绑定时，内置的 domain、fact-check、feedback 和 modality 信号使用决策模型，默认是 vela3-fast。下面的绑定显式选用 Vela 1.0。见[选择模型](../choose-a-model#by-task)和[为信号选择模型](../../installation/choose-a-model-for-a-signal.md)。

## 打开它 {#turn-it-on}

把信号加上，在路由里用起来。模型由 router 在 CPU 上替你跑，不用别的：

```yaml
routing:
  signals:
    domains:
      - name: math
        description: Mathematics and quantitative reasoning.
        mmlu_categories: [math]
      - name: computer science
        description: Programming and computer science.
        mmlu_categories: [computer science]
  decisions:
    - name: math-route
      priority: 100
      rules:
        operator: AND
        conditions:
          - type: domain
            name: math
      modelRefs:
        - model: math-model
```

## 选它跑在哪 {#choose-where-it-runs}

想自己挑设备、输入上限或换个模型，就写个部署，把信号绑上去。绑定键是信号类型 `domain`、`fact_check`、`user_feedback` 和 `modality`，每个都读取其模型的标签概率：

```yaml
global:
  model_catalog:
    deployments:
      vela-domain:
        provider: model_runtime
        artifact: vllm-sr/Vela-1.0-Encoder-307M-Domain
        device: cpu
        input:
          max_tokens: 2048
          overflow: truncate
    bindings:
      domain: vela-domain
```

`overflow: truncate` 拿长请求的前 2,048 个 token 分类，并如实报告截断了；`reject`（默认）则让超限输入的信号保持未知。

## 验一下 {#check-it}

直接问模型。单独起一个，或用任意服务它的 runtime：

这些 worker 级示例在含有 `vllm-srun` 的环境里运行（例如 Router 镜像）。Classify、embeddings、rerank 和 bundle 是 worker API；实例前端发布的是 System One 和 decision 请求。

```bash
vllm-srun serve vllm-sr/Vela-1.0-Encoder-307M-Domain --device cpu --port 8100
curl -s localhost:8100/v1/classify -H 'content-type: application/json' \
  -d '{"input": ["What is the derivative of x squared?", "Fix this segfault in my C code."]}'
```

每条结果带最可能的 `label` 和按 `labels` 顺序排的全部 `probabilities`。经过 router 的话，发请求时带 `x-vsr-debug: true` 头：`x-vsr-matched-domains` 列出匹配上的领域，`x-vsr-selected-decision` 点名走了哪条路由。

## 用自己的分类器 {#use-your-own-classifier}

Hugging Face 上的 ModernBERT 或 mmBERT 序列分类器，可以直接当 `classifier` 信号用，标签随你定。钉住它的 revision、在规则上指定它的部署、按它的标签路由：

```yaml
routing:
  signals:
    classifiers:
      - name: ticket_topic
        deployment: ticket-topics
        labels: [billing, shipping, other]
  decisions:
    - name: billing-desk
      priority: 100
      rules:
        operator: AND
        conditions:
          - type: classifier
            name: ticket_topic
            label: billing
            predicate:
              gte: 0.5
      modelRefs:
        - model: support-model
global:
  model_catalog:
    deployments:
      ticket-topics:
        provider: model_runtime
        artifact: your-org/ticket-topic-classifier
        revision: 0123456789abcdef0123456789abcdef01234567
        device: cpu
        input:
          max_tokens: 512
          overflow: reject
```

`labels` 要按模型自己的标签顺序列（它的 `id2label`）。router 从 runtime 读模型的标签，和规则对不上就拒绝启动。互相独立的标签、又有公开工作点的模型，用部署的 `operating_point` 固定工作点；见[分类器信号](../../tutorials/signal/learned/systemone.md)。

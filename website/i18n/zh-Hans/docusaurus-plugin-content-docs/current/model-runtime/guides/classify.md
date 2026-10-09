---
title: 分类请求
description: 按请求的领域、是否需要事实核查、用户反馈和请求的输出模态路由，或按你自己分类器的标签路由。
translation:
  source_commit: "439c22531380dfd51abf2ce33ce6dc4db87b1fb7"
  source_file: "docs/model-runtime/guides/classify.md"
  outdated: false
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
| [`classifier`](../../tutorials/signal/learned/classifier.md) | 你自己的模型 | 你的标签 |

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

想自己挑设备、输入上限或换个模型，就写个部署，把功能绑上去。绑定名叫 `domain_classifier`、`fact_check_classifier`、`feedback_detector` 和 `modality_detector`，它们读的都是标签概率（`label_distribution.v1`）：

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
      domain_classifier:
        deployment: vela-domain
        contract: label_distribution.v1
```

`overflow: truncate` 拿长请求的前 2,048 个 token 分类，并如实报告截断了；`reject`（默认）则让超限输入的信号保持未知。

## 验一下 {#check-it}

直接问模型。单独起一个，或用任意服务它的 runtime：

```bash
vllm-sr serve vllm-sr/Vela-1.0-Encoder-307M-Domain --device cpu --port 8100
curl -s localhost:8100/v1/classify -H 'content-type: application/json' \
  -d '{"input": ["What is the derivative of x squared?", "Fix this segfault in my C code."]}'
```

每条结果带最可能的 `label` 和按 `labels` 顺序排的全部 `probabilities`。经过 router 的话，发请求时带 `x-vsr-debug: true` 头：`x-vsr-matched-domains` 列出匹配上的领域，`x-vsr-selected-decision` 点名走了哪条路由。

## 用自己的分类器 {#use-your-own-classifier}

Hugging Face 上的 ModernBERT 或 mmBERT 序列分类器，可以直接当 `classifier` 信号用，标签随你定。钉住它的 revision、绑到规则上、按它的标签路由：

```yaml
routing:
  signals:
    classifiers:
      - name: ticket_topic
        type: local
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
  model_bindings:
    classifier.ticket_topic:
      deployment: ticket-topics
      contract: label_distribution.v1
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

`labels` 要按模型自己的标签顺序列（它的 `id2label`）。router 从 runtime 读模型的标签，和规则对不上就拒绝启动。互相独立的标签、又有公开工作点的模型，用 `label_scores.v1`；见[分类器信号](../../tutorials/signal/learned/classifier.md)。

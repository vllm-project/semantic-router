---
title: 检测 PII
description: 用精确的字符位置找出请求里的个人信息，并按它路由或拦截。
translation:
  source_commit: "fddd53d7c446c30ae183d16b6d63ec54f7a00a3e"
  source_file: "docs/model-runtime/guides/pii.md"
  outdated: false
is_mtpe: true
---

# 检测 PII {#detect-pii}

Vela 1.0 PII 找出请求里的个人信息，还告诉你它在哪：每个片段都有类型、精确字符和一个概率。[`pii` 信号](../../tutorials/signal/learned/pii.md)在发现你不允许的类型时匹配，路由随即可把请求扣在私有模型上，或直接拒绝。

模型认得 17 种类型：`AGE`、`CREDIT_CARD`、`DATE_TIME`、`DOMAIN_NAME`、`EMAIL_ADDRESS`、`GPE`（地点）、`IBAN_CODE`、`IP_ADDRESS`、`NRP`（国籍、宗教或政治团体）、`ORGANIZATION`、`PERSON`、`PHONE_NUMBER`、`STREET_ADDRESS`、`TITLE`、`US_DRIVER_LICENSE`、`US_SSN` 和 `ZIP_CODE`。

## 打开它 {#turn-it-on}

```yaml
routing:
  signals:
    pii:
      - name: personal_data
        threshold: 0.85
        pii_types_allowed: [EMAIL_ADDRESS]
  decisions:
    - name: keep-private
      priority: 200
      rules:
        operator: AND
        conditions:
          - type: pii
            name: personal_data
      modelRefs:
        - model: private-model
```

Vela PII 由 router 跑在 CPU 上。整个请求最多 32,768 个 token，按 512 token 的滑动窗口读完，所以长文档末尾的名字和开头的一样找得准。

## 选它跑在哪 {#choose-where-it-runs}

绑定名叫 `pii_classifier`，读的是文本片段（`token_spans.v1`）：

```yaml
global:
  model_catalog:
    deployments:
      vela-pii:
        provider: model_runtime
        artifact: vllm-sr/Vela-1.0-Encoder-307M-PII
        device: cpu
        input:
          max_tokens: 32768
          overflow: window
    bindings:
      pii_classifier:
        deployment: vela-pii
        contract: token_spans.v1
    modules:
      classifier:
        pii:
          window:
            size: 512
            overlap: 255
```

PII 就用 `overflow: window`：模型把整个请求按 512 token 的窗口读，窗和窗重叠 255 个 token。`truncate` 只扫开头，`reject` 会让长请求的信号变成未知。

### 关于 Vela 2.0

Vela 2.0 用它的 router span 头找出同样的 17 种类型，作为一个现成的问题。把 `pii_classifier` 绑到一个 Vela 2.0 部署上：PII 问题与请求就同一文本问该部署的[决策问题](model-runtime/guides/decisions.md)在同一次调用里发出。

```yaml alternative
global:
  model_catalog:
    deployments:
      vela2:
        provider: model_runtime
        artifact: vllm-sr/Vela-2.0-0.3B
        device: cpu
    bindings:
      pii_classifier:
        deployment: vela2
        contract: token_spans.v1
```

模型自己读整个请求，所以部署不带 `input`，PII 模块也不带 `window`。规则阈值如何作用于它的片段，见[`pii` 信号](../../tutorials/signal/learned/pii.md)。

## 扫描做不完的时候 {#when-the-scan-cannot-finish}

模型没就绪或扫描失败，信号就是未知的。模型没读到的内容（超出它的输入或[扫描上限](model-runtime/reference.md#long-inputs)、被截断，或没赶上信号的截止时间）匹配为 `unscanned`，除非模块设了 `on_unscanned: allow`。PII 模块的 `on_error` 决定未知信号意味着什么：`allow`（默认）把没读到的文本当干净，`block` 则把它当 `classification_error` 匹配上——没检查过的文本，别想以干净的身份过关。

```yaml
global:
  model_catalog:
    modules:
      classifier:
        pii:
          on_error: block
```

## 验一下 {#check-it}

这些 worker 级示例在含有 `vllm-srun` 的环境里运行（例如 Router 镜像）。Classify、embeddings、rerank 和 bundle 是 worker API；实例前端发布的是 System One 和 decision 请求。

```bash
vllm-srun serve vllm-sr/Vela-1.0-Encoder-307M-PII --device cpu --port 8100
curl -s localhost:8100/v1/classify -H 'content-type: application/json' \
  -d '{"input": ["Hi, I am Tom Baker, write to tom.baker@example.com."]}'
```

每个片段带 `label`、`start` 和 `end`（按字符，尾巴不含）、`text` 和 `probability`。经过 router 时，`x-vsr-matched-pii` 列出匹配上的 PII 规则。

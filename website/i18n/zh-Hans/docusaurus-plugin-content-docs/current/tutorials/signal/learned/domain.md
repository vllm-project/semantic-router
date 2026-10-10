---
translation:
  source_commit: "9156d5bc1ed9edff626b95a2b8260a77cb1712c5"
  source_file: "docs/tutorials/signal/learned/domain.md"
  outdated: false
---

# 领域信号 {#domain-signal}

## 概览 {#overview}

`domain` 对请求主题族分类。在 `routing.signals.domains` 下定义领域规则。

检测器使用 `global.model_catalog.modules.classifier` 下配置的分类器，以及 `global.model_catalog.system` 下的模型绑定。

## 主要优势 {#key-advantages}

- 按主题路由，无需把每个短语硬编码进关键词列表。
- 领域策略可在多条决策间复用。
- 支持稳定、易审计的类别族。
- 适合作为路由图中第一个学习型信号。

## 解决什么问题？ {#what-problem-does-it-solve}

关键词路由在提示被改写、或领域边界宽于少量短语时会失效。

`domain` 将主题分类映射为命名路由信号，决策可将其与复杂度、安全或插件逻辑组合。

## 何时使用 {#when-to-use}

在以下情况使用 `domain`：

- 路由按主题族组织
- 词法匹配过于脆弱
- 同一主题边界应供给多条决策
- 希望在加入更专信号前先有稳定的学习分类器

## 配置 {#configuration}

```yaml
routing:
  signals:
    domains:
      - name: business
        description: Business and management related queries.
        mmlu_categories: [business]
      - name: law
        description: Legal questions and law-related topics.
        mmlu_categories: [law]
      - name: psychology
        description: Psychology and mental health topics.
        mmlu_categories: [psychology]
      - name: health
        description: Health and medical information queries.
        mmlu_categories: [health]
      - name: other
        description: General fallback traffic.
        mmlu_categories: [other]
```

保持领域名稳定，因为决策直接引用这些名称。

只有已声明的领域会匹配。分类器标签匹配在 `mmlu_categories` 中列出它的领域，或与它同名的领域。没有任何领域列出的标签计为 `other`，因此像上面那样列出 `other` 的领域，就是所有未声明主题的回退；没有这样的领域时，这类请求不匹配任何领域。

### 本地与远程分类器选择 {#local-and-remote-classifier-selection}

没有 `backend` 时，领域分类器（默认 Vela 2.0 0.3B）运行在[模型运行时](../../../model-runtime/guides/classify)中。运行时从模型包读取架构，因此无需设置本地选择器；`vllm-sr config migrate` 会移除早期的 `variant`、`use_modernbert` 和 `use_mmbert_32k` 键。

远程类别分类器使用共享 backend 块。其 `model` 是 `global.model_catalog.external[]` 中的显式名称，该目录条目必须有 `model_role: classification`。Category 目前只接受 `http_classify` 协议与 `label_distribution.v1` 约定，以便完整标签分布继续供给领域匹配与路由决策。

```yaml
global:
  model_catalog:
    external:
      - name: domain-service
        model_role: classification
        llm_endpoint:
          address: domain-classifier.default.svc
          port: 8080
        llm_model_name: domain-intent-v1
    modules:
      classifier:
        domain:
          backend:
            protocol: http_classify
            contract: label_distribution.v1
            model: domain-service
            deadline_ms: 5000
```

## 依赖与限制 {#dependencies-and-limitations}

领域分类使用已配置的分类器模块并处理请求文本。把 `other` 声明为回退，并在分类器变更时重新评估标签与阈值。完整示例见：
[`config/fragments/signal/domain/mmlu.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/domain/mmlu.yaml)。

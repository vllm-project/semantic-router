---
is_mtpe: false
translation:
  source_commit: "9156d5bc1ed9edff626b95a2b8260a77cb1712c5"
  source_file: "docs/tutorials/signal/learned/domain.md"
  outdated: false
---

# 领域信号 {#domain-signal}

## 概览 {#overview}

`domain` 对请求主题族分类。在 `routing.signals.domains` 下定义领域规则。

除非 `domain` 绑定指定了其他部署，否则由决策模型回答它；见[为信号选择模型](../../../installation/choose-a-model-for-a-signal.md)。`global.model_catalog.modules.classifier.domain` 保存它的设置，例如阈值。

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

### 本地与远程模型 {#local-and-remote-models}

默认由决策模型（vela3-fast）在[模型运行时](../../../model-runtime/guides/classify.md)中回答 domain 信号。运行时从模型包读取架构，因此无需设置本地选择器；`vllm-sr config migrate` 会移除早期的 `variant`、`use_modernbert` 和 `use_mmbert_32k` 键。

远程领域分类器是 `domain` 所绑定的 `classify` 部署。Router 把请求文本发送到 `<endpoint>/classify`，并读取完整的标签分布，用于领域匹配与路由决策。

```yaml
global:
  model_catalog:
    deployments:
      domain-service:
        provider: classify
        endpoint: https://domain-classifier.example.com
        auth: {api_key_env: DOMAIN_SERVICE_KEY}
        timeout_ms: 5000
    bindings:
      domain: domain-service
```

## 依赖与限制 {#dependencies-and-limitations}

领域分类在回答该信号的部署上处理请求文本。把 `other` 声明为回退，并在该部署变更时重新评估标签与阈值。完整示例见：
[`config/fragments/signal/domain/mmlu.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/domain/mmlu.yaml)。

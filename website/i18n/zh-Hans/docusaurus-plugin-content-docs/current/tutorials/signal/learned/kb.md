---
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/tutorials/signal/learned/kb.md"
  outdated: false
---

# 知识库信号 {#knowledge-base-signal}

## 概览 {#overview}

`kb` 将路由信号绑定到命名知识库实例的输出。在 `routing.signals.kb` 下定义这些绑定。

用于在 Router 启动时加载、并在多条路由间复用的嵌入知识库。

## 主要优势 {#key-advantages}

- 同一示例集可被多条路由复用。
- 标签、分组与数值指标显式，不依赖魔法运行时名。
- 支持胜者式与阈值式信号绑定。
- 投影可消费连续知识库指标，而不把信号变成脚本表面。

## 解决什么问题？ {#what-problem-does-it-solve}

部分路由策略依赖精选示例集，而非单一关键词或嵌入候选列表。一个知识库可以把请求分类为隐私、安全、情感或偏好标签，而路由配置只暴露决策需要的标签或分组。

`kb` 明确这一分层：

- `global.model_catalog.kbs[]` 持有可复用知识库包
- `routing.signals.kb[]` 将特定标签或分组绑定为命名路由信号
- `routing.projections` 可消费如 `best_score`、`best_matched_score` 或配置的分组 margin 等知识库指标

## 何时使用 {#when-to-use}

在以下情况使用 `kb`：

- 请求必须对照精选示例集分类
- 一个启动加载的知识库结果应供给多条路由
- 需要稳定路由级分组而不重复示例
- 需要显式绑定而非隐式信号名

## 配置 {#configuration}

```yaml
global:
  model_catalog:
    kbs:
      - name: privacy_kb
        source:
          path: knowledge_bases/privacy/
          manifest: labels.json
        threshold: 0.55
        prototype_scoring:
          enabled: true
          cluster_similarity_threshold: 0.9
          max_prototypes: 8
          best_weight: 0.75
          top_m: 2
          margin_threshold: 0.05
        label_thresholds:
          prompt_injection: 0.7
        groups:
          privacy_policy: [proprietary_code, internal_document, pii]
          security_containment: [prompt_injection, credential_exfiltration]
          private: [proprietary_code, internal_document, pii, prompt_injection, credential_exfiltration]
          public: [generic_coding, general_knowledge]
        metrics:
          - name: private_vs_public
            type: group_margin
            positive_group: private
            negative_group: public

routing:
  signals:
    kb:
      - name: privacy_policy
        kb: privacy_kb
        target:
          kind: group
          value: privacy_policy
        match: best
      - name: proprietary_code
        kb: privacy_kb
        target:
          kind: label
          value: proprietary_code
        match: threshold
```

保持知识库名稳定，因为 `kb` 信号直接绑定这些名称。

启用 `prototype_scoring` 时，KB 从标签示例构建每标签原型库。运行时分类随后从这些标签自有原型给标签打分，而不是让一个原始示例永远主导整个标签。

## 匹配语义 {#match-semantics}

`routing.signals.kb[]` 支持：

- `target.kind: label` 或 `group`
- `match: best` 或 `threshold`

含义：

- `label + best`：仅当该标签是知识库最佳标签时匹配
- `label + threshold`：标签分数超过有效阈值时匹配
- `group + best`：仅当该分组是知识库最佳分组时匹配
- `group + threshold`：任一成员标签超过阈值时匹配

## 投影指标 {#projection-metrics}

知识库信号是布尔路由输入。数值输出留在 `routing.projections`。

例如：

```yaml
routing:
  projections:
    scores:
      - name: privacy_bias
        method: weighted_sum
        inputs:
          - type: kb_metric
            kb: privacy_kb
            metric: private_vs_public
            value_source: score
            weight: 1.0
```

命名知识库指标在 `global.model_catalog.kbs[].metrics[]` 下声明。内置指标 `best_score` 与 `best_matched_score` 始终可用。

## 依赖与限制 {#dependencies-and-limitations}

- 知识库包从 `global.model_catalog.kbs[].source` 加载。请将其清单与文件一起版本化。
- 请求文本通过共享语义嵌入运行时嵌入。因此远程嵌入提供方会收到该文本。
- 标签、分组、阈值与嵌入模型构成一个校准单元；任一部分变更时请一起重新评估。
- 知识库信号示例见
 [`config/fragments/signal/kb/privacy.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/kb/privacy.yaml)，
 完整 KB 声明见
 [`config/config.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml)。

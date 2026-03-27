---
translation:
  source_commit: "baa07413"
  source_file: "docs/tutorials/signal/learned/kb.md"
  outdated: false
---

# Knowledge Base 信号

## 概览

`kb` 将路由信号绑定到命名知识库实例的输出。映射到 `config/signal/kb/`，在 `routing.signals.kb` 中声明。

该信号族面向**维护型**、启动时加载的嵌入知识库，并在多条路由间复用。

## 主要优势

- 同一维护示例包可被多条路由复用。
- 标签、分组与数值指标显式，不依赖魔法运行时名。
- 支持胜者式与阈值式信号绑定。
- 投影可消费连续知识库指标，而不把信号变成脚本表面。

## 解决什么问题？

部分路由策略依赖精选示例集，而非单一关键词或嵌入候选列表。可能希望在启动加载的知识库上将请求分类为隐私、安全、情感或偏好标签，再只暴露路由图应看到的标签或分组绑定。

`kb` 明确这一分层：

- `global.model_catalog.kbs[]` 持有可复用知识库包
- `routing.signals.kb[]` 将特定标签或分组绑定为命名路由信号
- `routing.projections` 可消费如 `best_score`、`best_matched_score` 或配置的分组 margin 等指标

## 何时使用

在以下情况使用 `kb`：

- 请求必须对照维护的示例包分类
- 一个启动加载的知识库结果应供给多条路由
- 需要稳定路由级分组而不重复示例
- 需要显式绑定而非隐式信号名

## 配置

源片段族：`config/signal/kb/`

```yaml
global:
  model_catalog:
    kbs:
      - name: privacy_knowledge_base
        source:
          path: kb/privacy/
          manifest: labels.json
        threshold: 0.55
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
        kb: privacy_knowledge_base
        target:
          kind: group
          value: privacy_policy
        match: best
      - name: proprietary_code
        kb: privacy_knowledge_base
        target:
          kind: label
          value: proprietary_code
        match: threshold
```

保持知识库名稳定，`kb` 信号直接绑定这些名称。

## 匹配语义

`routing.signals.kb[]` 支持：

- `target.kind: label` 或 `group`
- `match: best` 或 `threshold`

含义：

- `label + best`：仅当该标签是知识库最佳标签时匹配
- `label + threshold`：标签分数超过有效阈值时匹配
- `group + best`：仅当该分组是知识库最佳分组时匹配
- `group + threshold`：任成员标签超过阈值时匹配

## 投影指标

知识库信号是布尔路由输入；数值输出留在 `routing.projections`。

例如：

```yaml
routing:
  projections:
    scores:
      - name: privacy_bias
        method: weighted_sum
        inputs:
          - type: kb_metric
            kb: privacy_knowledge_base
            metric: private_vs_public
            value_source: score
            weight: 1.0
```

命名知识库指标在 `global.model_catalog.kbs[].metrics[]` 声明。内置指标 `best_score` 与 `best_matched_score` 始终可用。

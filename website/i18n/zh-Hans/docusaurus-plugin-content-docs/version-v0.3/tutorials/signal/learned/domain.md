---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/signal/learned/domain.md"
  outdated: false
---

# Domain 信号

## 概览

`domain` 对请求主题族分类。映射到 `config/signal/domain/`，在 `routing.signals.domains` 中声明。

该族为学习型：路由器使用 `global.model_catalog.modules.classifier` 下的领域分类路径，以及 `global.model_catalog.system` 中稳定的领域系统模型绑定。

## 主要优势

- 按主题路由，无需把每个短语硬编码进关键词列表。
- 领域策略可在多条决策间复用。
- 支持稳定、易审计的类别族。
- 适合作为路由图中第一个学习型信号。

## 解决什么问题？

关键词路由在改写提示或领域边界宽于短语集合时会失效。

`domain` 将主题分类映射为命名路由信号，可与复杂度、安全或插件逻辑组合。

## 何时使用

在以下情况使用 `domain`：

- 路由按主题族组织
- 词法匹配过于脆弱
- 同一主题边界应供给多条决策
- 希望在加入更专信号前先有稳定的学习分类器

## 配置

源片段族：`config/signal/domain/`

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

保持领域名稳定，决策直接引用这些名称。

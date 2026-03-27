---
translation:
  source_commit: "485c74ba"
  source_file: "docs/tutorials/signal/learned/embedding.md"
  outdated: false
---

# Embedding 信号

## 概览

`embedding` 通过语义相似度将请求与代表性示例匹配。映射到 `config/signal/embedding/`，在 `routing.signals.embeddings` 中声明。

该族为学习型：依赖 `global.model_catalog.embeddings` 中的语义嵌入资产。

## 主要优势

- 比纯关键词规则更好处理改写。
- 团队可用示例短语调路由，不必重训分类器。
- 适合支持意图、产品流程与语义 FAQ 路由。
- 是从纯词法信号平滑升级的一步。

## 解决什么问题？

关键词路由会漏掉措辞不同但语义相近的提示。完整领域分类在路由依赖窄意图时也可能过粗。

`embedding` 在嵌入空间中将新提示与示例候选匹配。

## 何时使用

在以下情况使用 `embedding`：

- 措辞变但意图稳定
- 希望语义路由而不引入完整自定义分类器
- 示例比领域标签更易维护
- 支持或工作流意图需要比关键词更好的召回

## 配置

源片段族：`config/signal/embedding/`

```yaml
routing:
  signals:
    embeddings:
      - name: technical_support
        threshold: 0.75
        aggregation_method: max
        candidates:
          - how to configure the system
          - installation guide
          - troubleshooting steps
          - error message explanation
          - setup instructions
      - name: account_management
        threshold: 0.72
        aggregation_method: max
        candidates:
          - password reset
          - account settings
          - profile update
          - subscription management
          - billing information
```

阈值与候选列表一起调；质量比堆砌低质示例更重要。

排序回退行为在路由器自有嵌入目录下单独配置：

```yaml
global:
  model_catalog:
    embeddings:
      semantic:
        embedding_config:
          enable_soft_matching: true
          top_k: 1
          min_score_threshold: 0.5
```

路由器按聚合相似度排序嵌入规则，并至多返回 `top_k` 条进入路由。默认 `top_k` 为 `1`，即仅返回最强嵌入信号，除非显式提高上限。若需要旧版「返回所有匹配嵌入规则」行为，可将 `top_k` 设为 `0`。

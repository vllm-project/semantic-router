---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/algorithm/selection/router-dc.md"
  outdated: true
---

# Router DC

## 概览

`router_dc` 用于**语义查询到模型匹配**的选择算法。

对应 `config/algorithm/selection/router-dc.yaml`。

## 主要优势

- 使用语义相似度，而非仅显式排序规则。
- 学习选择器阈值在配置中可见。
- 提示语义比静态优先级更重要时很合适。

## 解决什么问题？

部分路由需要所选模型与提示风格或任务语义紧密匹配。`router_dc` 将该学习语义选择器作为决策的一部分，而非外部隐藏步骤。

## 何时使用

- 最佳候选取决于提示与模型画像之间的语义相似度
- 需要学习选择器但不需要完整在线探索
- 单条路由应按语义契合而非仅成本或延迟路由

## 配置

在 `routing.decisions[].algorithm` 中使用：

```yaml
algorithm:
  type: router_dc
  router_dc:
    temperature: 0.2
    dimension_size: 384
    min_similarity: 0.7
    use_query_contrastive: true
```

---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/algorithm/selection/knn.md"
  outdated: false
---

# KNN

## 概览

`knn` 用于**最近邻**模型选择的选择算法。

对应 `config/algorithm/selection/knn.yaml`。

## 主要优势

- 在配置中显式表达基于示例的路由。
- 路由级策略紧凑。
- 相似提示应选择相似模型时很合适。

## 解决什么问题？

部分路由策略更易表述为「选择与历史相似提示效果好的模型」。`knn` 在决策中直接暴露最近邻选择器。

## 何时使用

- 有历史提示到模型的示例
- 相似提示通常应对应同一候选模型
- 路由应使用检索式选择而非固定排序

## 配置

在 `routing.decisions[].algorithm` 中使用：

```yaml
algorithm:
  type: knn
```

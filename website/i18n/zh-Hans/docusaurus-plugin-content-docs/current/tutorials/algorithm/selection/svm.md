---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/algorithm/selection/svm.md"
  outdated: false
---

# SVM

## 概览

`svm` 用于采用 **SVM 选择器**的路由。

对应 `config/algorithm/selection/svm.yaml`。

## 主要优势

- 在同一算法表面暴露经典分类器式选择。
- 选择器逻辑在模型内时路由配置很小。
- 轻量学习分类器即足够时很合适。

## 解决什么问题？

对部分工作负载，经典分类器比更大学习选择器或在线探索策略更易维护。`svm` 使基于分类器的路由选择显式。

## 何时使用

- 有面向该路由的基于 SVM 的选择器产物
- 轻量学习分类对模型选择足够
- 需要学习选择但无需额外编排

## 配置

在 `routing.decisions[].algorithm` 中使用：

```yaml
algorithm:
  type: svm
```

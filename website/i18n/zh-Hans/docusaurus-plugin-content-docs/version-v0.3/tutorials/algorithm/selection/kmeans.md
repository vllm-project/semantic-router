---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/algorithm/selection/kmeans.md"
  outdated: false
---

# KMeans

## 概览

`kmeans` 用于依赖**基于聚类**选择器的路由。

对应 `config/algorithm/selection/kmeans.yaml`。

## 主要优势

- 将基于聚类的选择与决策规则分离。
- 聚类由选择器自身处理时路由配置最小。
- 提示流量自然聚成若干簇时很合适。

## 解决什么问题？

当提示流量落入少数重复簇时，基于聚类的选择器可比手写优先级规则更干净地选模型。`kmeans` 在路由层暴露该选择器。

## 何时使用

- 有面向候选模型的基于聚类选择器
- 提示流量自然分成可重复类别
- 路由应使用学习到的簇而非静态排序

## 配置

在 `routing.decisions[].algorithm` 中使用：

```yaml
algorithm:
  type: kmeans
```

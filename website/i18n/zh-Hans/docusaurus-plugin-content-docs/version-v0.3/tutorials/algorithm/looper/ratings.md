---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/algorithm/looper/ratings.md"
  outdated: false
---

# Ratings

## 概览

`ratings` 是循环算法：在复用路由级评分信号的同时协调多个候选。

对应 `config/algorithm/looper/ratings.yaml`。

## 主要优势

- 在有并发上限的情况下支持多模型执行。
- 评分感知编排局部在单条路由内。
- 错误处理行为显式。

## 解决什么问题？

部分路由需要多个候选参与，但仍需受控循环而非无界扇出。`ratings` 暴露这种有界多模型协调策略。

## 何时使用

- 同一路由内应运行多个候选
- 路由级评分应影响循环
- 并发需要硬上限

## 配置

在 `routing.decisions[].algorithm` 中使用：

```yaml
algorithm:
  type: ratings
  ratings:
    max_concurrent: 3
    on_error: skip
```

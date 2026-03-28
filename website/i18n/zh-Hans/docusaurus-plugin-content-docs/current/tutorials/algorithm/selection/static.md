---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/algorithm/selection/static.md"
  outdated: false
---

# Static

## 概览

`static` 是最简单的选择算法：路由保留候选列表，选择策略固定。

对应 `config/algorithm/selection/static.yaml`。

## 主要优势

- 确定性、易推理。
- 无学习选择器状态或运行时调参。
- 候选顺序已有意图时的良好默认。

## 解决什么问题？

有时路由只需稳定胜者策略，无需额外打分。`static` 显式表达这一点，而非让选择行为隐式。

## 何时使用

- 路由匹配后应始终同一候选胜出
- 模型顺序已在算法层外精心编排
- 需要最简单的路由本地选择策略

## 配置

在 `routing.decisions[].algorithm` 中使用：

```yaml
algorithm:
  type: static
```

---
is_mtpe: false
translation:
  source_commit: "72b3d17c5fdcaa506e603b1697fdb2d4b6318a6e"
  source_file: "docs/tutorials/algorithm/selection/router-dc.md"
  outdated: false
---

# Router DC（已移除）

## 概览

`router_dc` 选择器已被移除，其 `global.router.model_selection.router_dc` 设置也一并移除。[System One](./systemone) 选择器按候选的描述进行选择。

## 它解决什么问题？

它对请求和每个模型描述进行嵌入，然后选择相似度最强的候选。`systemone` 选择器向决策模型询问应由哪个已描述的候选作答。

## 何时使用

不要再配置它：仍然使用它的配置会在加载时失败。请使用 [System One](./systemone)。

## 配置

运行 `vllm-sr config migrate` 来改写它。有两个或更多候选的决策会变为 `systemone` 选择器；只有一个候选的决策会变为 `static`。请在模型卡片或 `algorithm.systemone.candidates` 中描述每个候选：

```yaml
algorithm:
  type: systemone
  systemone:
    instructions: Which candidate model best fits this request?
```

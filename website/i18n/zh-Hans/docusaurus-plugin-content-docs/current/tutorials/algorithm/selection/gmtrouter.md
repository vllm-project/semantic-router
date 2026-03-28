---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/algorithm/selection/gmtrouter.md"
  outdated: false
---

# GMT Router

## 概览

`gmtrouter` 基于历史与学习模型的个性化路由选择算法。

对应 `config/algorithm/selection/gmtrouter.yaml`。

## 主要优势

- 支持按用户或租户个性化。
- 个性化模型路径在配置中显式。
- 单条路由可选用历史感知选择而不影响其他路由。

## 解决什么问题？

部分工作负载需要选择反映**先前用户交互**，而非全局平均。`gmtrouter` 为已匹配决策增加学习个性化层。

## 何时使用

- 路由应适应用户或租户历史
- 有可加载的训练选择器产物
- 静态或仅反馈排序对工作负载不足

## 配置

在 `routing.decisions[].algorithm` 中使用：

```yaml
algorithm:
  type: gmtrouter
  gmtrouter:
    enable_personalization: true
    history_sample_size: 50
    model_path: models/gmtrouter.pt
```

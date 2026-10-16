---
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/tutorials/algorithm/selection/static.md"
  outdated: false
---

# 静态选择

## 概览

`static` 在没有指标或已学习状态的情况下提供确定性模型选择。
它默认选择决策 `modelRefs` 中的第一项。匹配的领域可以提供固定 `model_scores`；当分数不同于默认哨兵 `1.0` 时，选择器使用最高分。

## 主要优势

- 确定性，易于审计。
- 没有选择器模型、指标或存储依赖。
- 为其他选择策略提供稳定基线。

## 解决什么问题？

有些路由已经有有意的模型顺序或固定的按领域分数，不需要在线排序策略。Static 让这种选择变得显式。

## 何时使用

将 Static 用于确定性路由、比较选择器时的基线，或外部进程拥有候选排序时。如果决策只有一个候选，通常可以完全省略算法。

## 配置

```yaml
algorithm:
  type: static
```

把预期的回退胜者放在 `modelRefs` 第一位。要用领域 `model_scores` 排序时，为每个候选打分，并避免 `1.0`，因为选择器把该值保留给第一候选回退。
完整示例见：
[`config/fragments/algorithm/selection/static.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/selection/static.yaml)。

## 依赖与限制

- 没有外部依赖，除普通决策匹配外也不处理请求内容。
- 它不会故障转移到后续候选，不会对负载做出反应，也不会从结果中学习。后端可用性仍由普通提供商路径负责。

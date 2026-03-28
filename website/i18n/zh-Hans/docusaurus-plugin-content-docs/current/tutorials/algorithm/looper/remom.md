---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/algorithm/looper/remom.md"
  outdated: false
---

# ReMoM

## 概览

`remom` 是用于**广度受控**的多模型编排循环算法。

对应 `config/algorithm/looper/remom.yaml`。

## 主要优势

- 按调度广度模式协调多个候选模型。
- 中间响应行为显式。
- 适合需要比简单升级更丰富编排的路由。

## 解决什么问题？

部分路由不仅要升级，还要控制每阶段参与模型数量。`remom` 在配置中直接给出广度调度。

## 何时使用

- 一条路由应在多轮中协调多个模型
- 需要可配置的广度调度而非单步升级
- 中间响应应显式包含或排除

## 配置

在 `routing.decisions[].algorithm` 中使用：

```yaml
algorithm:
  type: remom
  remom:
    breadth_schedule: [3, 2, 1]
    model_distribution: round_robin
    temperature: 0.7
    max_concurrent: 3
    include_intermediate_responses: false
    on_error: skip
```

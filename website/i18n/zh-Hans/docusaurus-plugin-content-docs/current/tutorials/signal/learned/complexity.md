---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/signal/learned/complexity.md"
  outdated: false
---

# Complexity 信号

## 概览

`complexity` 估计提示是否需要更难推理路径或更便宜的简易路径。映射到 `config/signal/complexity/`，在 `routing.signals.complexity` 中声明。

该族为学习型：分类器用嵌入相似度将请求与难/易候选比较，并可选用多模态候选。

## 主要优势

- 将推理升级与领域分类解耦。
- 同一复杂度策略可被多条决策复用。
- 难/易示例随时间调参较简单。
- 简单提示留在便宜模型，复杂提示升级。

## 解决什么问题？

仅凭主题无法判断是否需要强推理。同一领域内两个问题推理深度可能差异很大。

`complexity` 通过示例驱动的规则直接估计任务难度。

## 何时使用

在以下情况使用 `complexity`：

- 部分提示需要更强推理或更长思维链
- 简单流量应留在更便宜模型
- 希望升级策略独立于领域
- 多模态推理请求与简单提示需不同处理

## 配置

源片段族：`config/signal/complexity/`

```yaml
routing:
  signals:
    complexity:
      - name: needs_reasoning
        threshold: 0.75
        description: Escalate multi-step reasoning or synthesis-heavy prompts.
        hard:
          candidates:
            - solve this step by step
            - compare multiple tradeoffs
            - analyze the root cause
        easy:
          candidates:
            - answer briefly
            - quick summary
            - simple rewrite
```

使用有代表性的难/易示例配置 `complexity`，使学习边界与真实路由成本画像一致。

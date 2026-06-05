---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/signal/heuristic/context.md"
  outdated: false
---

# Context 信号

## 概览

`context` 检测需要更大**有效上下文窗口**的请求。映射到 `config/signal/context/`，在 `routing.signals.context` 中声明。

该族为启发式：按 token 窗口需求路由，而非分类器推理。

## 主要优势

- 长上下文路由显式化，而非埋在模型默认里。
- 短提示不必承担超大上下文模型成本。
- 同一上下文阈值可被多条决策复用。
- 与领域或复杂度信号配合良好。

## 解决什么问题？

两个提示可能主题相同，但所需上下文窗口差异很大。若只按领域路由，长文档可能落到会截断或失败的模型上。

`context` 将上下文窗口需求作为一等路由输入。

## 何时使用

在以下情况使用 `context`：

- 部分路由需要 32K、128K 或更大上下文
- 长文档流量应使用不同模型族
- 希望短请求留在更便宜或更快的模型上
- 路由依赖上下文规模而非仅主题

## 配置

源片段族：`config/signal/context/`

```yaml
routing:
  signals:
    context:
      - name: long_context
        min_tokens: 32K
        max_tokens: 256K
        description: Requests that need a larger effective context window.
```

当路由器应根据提示长度或预期上下文需求切换候选时，使用 `context`。

---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/signal/heuristic/keyword.md"
  outdated: false
---

# Keyword 信号

## 概览

`keyword` 匹配请求中的显式词法模式。映射到 `config/signal/keyword/`，在 `routing.signals.keywords` 中声明。

该族为启发式：用配置的词、短语或轻量检索方式匹配，而非学习型意图分类器。

## 主要优势

- 对明显词法情况给出确定性路由。
- 支持简单类正则匹配及更强的 BM25 或 n-gram 变体。
- 触发短语显式，易于审计。
- 往往是构建有用路由图的最快路径。

## 解决什么问题？

有些路由不需要完整分类器，只需识别稳定词如账单、重置密码、紧急支持。

`keyword` 将词法匹配变成可复用命名信号，而不是在决策里散落字符串判断。

## 何时使用

在以下情况使用 `keyword`：

- 词法线索稳定且信噪比高
- 需要对支持或策略关键词做确定性路由
- 在学习型信号之前需要低延迟第一道筛选
- 措辞比语义改写覆盖更重要

## 配置

源片段族：`config/signal/keyword/`

```yaml
routing:
  signals:
    keywords:
      - name: code_keywords
        operator: OR
        method: bm25
        keywords: ["code", "function", "debug", "algorithm", "refactor"]
        bm25_threshold: 0.1
        case_sensitive: false
      - name: urgent_keywords
        operator: OR
        method: ngram
        keywords: ["urgent", "immediate", "asap", "emergency"]
        ngram_threshold: 0.4
        ngram_arity: 3
        case_sensitive: false
```

简单场景用纯关键词列表；精确文本过脆时再使用 `method: bm25` 或 `method: ngram`。

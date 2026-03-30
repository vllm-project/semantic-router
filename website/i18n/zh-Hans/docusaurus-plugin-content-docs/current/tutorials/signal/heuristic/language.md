---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/signal/heuristic/language.md"
  outdated: false
---

# Language 信号

## 概览

`language` 检测请求语言并暴露为路由信号。映射到 `config/signal/language/`，在 `routing.signals.language` 中声明。

在本教程分类中属启发式：使用轻量语言检测器，而非路由器自有分类模型。

## 主要优势

- 多语言流量路由无需按区域复制决策。
- 语言处理在路由图中显式。
- 与模态、上下文及模型族约束配合良好。
- 仅关心区域时不必支付领域分类器成本。

## 解决什么问题？

若忽略语言，多语言流量可能落到对该区域弱的模型，或假设仅英文的插件上。

`language` 将检测到的区域变成可复用路由输入。

## 何时使用

在以下情况使用 `language`：

- 不同语言需要不同模型族
- 多语言支持是分层的或部分的
- 下游工具或提示依赖区域
- 希望语言检测与路由结果清晰分离

## 配置

源片段族：`config/signal/language/`

```yaml
routing:
  signals:
    language:
      - name: zh
        description: Chinese-language requests.
      - name: es
        description: Spanish-language requests.
```

规则名应与决策要引用的语言代码一致，如 `zh`、`es`、`en`。

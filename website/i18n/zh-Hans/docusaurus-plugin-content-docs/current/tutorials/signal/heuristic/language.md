---
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/tutorials/signal/heuristic/language.md"
  outdated: false
---

# 语言信号 {#language-signal}

## 概览 {#overview}

`language` 检测请求语言并将其暴露为路由信号。在 `routing.signals.language` 下定义语言规则。

它使用轻量语言检测器，而不是通用分类器模型。

## 主要优势 {#key-advantages}

- 多语言流量无需按区域复制决策即可路由。
- 语言处理在路由图中保持显式。
- 与模态、上下文和模型族约束配合良好。
- 仅关心区域时，不必为领域分类器付费。

## 解决什么问题？ {#what-problem-does-it-solve}

若忽略语言，多语言流量可能落到对检测到的区域较弱的模型上，或落到假定仅英语行为的插件上。

`language` 将检测到的区域变成可复用的路由输入。

## 何时使用 {#when-to-use}

在以下情况使用 `language`：

- 不同语言需要不同模型族
- 多语言支持是部分或分档的
- 下游工具或提示依赖区域
- 希望在语言检测与路由结果之间划清边界

## 配置 {#configuration}

```yaml
routing:
  signals:
    language:
      - name: zh
        description: Chinese-language requests.
        threshold: 0.6
      - name: es
        description: Spanish-language requests.
```

规则名应与决策要引用的语言代码一致，例如 `zh`、`es` 或 `en`。`threshold` 是检测器的最低置信度，范围从 `0` 到 `1`；省略（或设为 `0`）则使用运行时默认值 `0.3`。更高阈值会减少误报，但可能让更多请求落到回退路由。

## 依赖与限制 {#dependencies-and-limitations}

短、混语以及代码密集的提示可能模糊。始终提供回退路由，并在你的流量上评估检测器。完整示例见：
[`config/fragments/signal/language/multilingual.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/language/multilingual.yaml)。

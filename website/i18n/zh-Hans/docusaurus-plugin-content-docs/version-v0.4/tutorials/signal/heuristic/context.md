---
translation:
  source_commit: "e56591a9cb24f073bf159927e87116ba6d278741"
  source_file: "docs/tutorials/signal/heuristic/context.md"
  outdated: false
---

# 上下文信号 {#context-signal}

## 概览 {#overview}

`context` 检测需要更大有效上下文窗口的请求。在 `routing.signals.context` 下定义上下文规则。

该族为启发式：按 token 窗口需求路由，而非分类器推理。

## 主要优势 {#key-advantages}

- 长上下文路由显式化，而非埋在模型默认里。
- 短提示不必承担超大上下文模型成本。
- 同一上下文阈值可被多条决策复用。
- 与领域或复杂度信号配合良好。

## 解决什么问题？ {#what-problem-does-it-solve}

两个提示可能主题相同，但所需上下文窗口差异很大。若只按领域路由，长文档可能落到会截断或失败的模型上。

`context` 将上下文窗口需求作为一等路由输入。

## 何时使用 {#when-to-use}

在以下情况使用 `context`：

- 部分路由需要 32K、128K 或更大上下文支持
- 长文档流量应使用不同模型族
- 希望短请求留在更便宜或更快的模型上
- 路由依赖上下文规模而非仅主题

## 配置 {#configuration}

```yaml
routing:
  signals:
    context:
      - name: long_context
        min_tokens: 32K
        max_tokens: 256K
        description: Requests that need a larger effective context window.
```

当 Router 应根据提示长度或预期上下文需求切换候选时，使用 `context`。

## 区间语义 {#range-semantics}

每条规则是一个闭区间 token 档：当 `min_tokens <= token_count <= max_tokens` 时匹配。

- 两个上限都可选，但至少要设一个。缺少 `min_tokens` 表示 0。
- 省略 `max_tokens` 使该档开口向上。所有大于等于 `min_tokens` 的请求都匹配，没有上限。把开口档放在最后，这样超出最大有界档的溢出仍带有上下文信号。
- 将 `min_tokens` 设成与 `max_tokens` 相等，即该 token 数的精确匹配档。
- 每条匹配规则都会按配置顺序报告。允许重叠档，两个名称都会出现在 `x-vsr-matched-context` 中。
- 档之间的空隙与重叠会在配置加载时记为警告。落在空隙中的请求不匹配任何上下文规则。
- 校验会拒绝既无上限、无法解析、负数或过大的值，以及 `min_tokens` 大于 `max_tokens` 的规则。Router、`vllm-sr` CLI 与控制面板使用同一套规则，因此通过 `vllm-sr config validate` 的档也会在 Router 中加载。

数值接受 `K` 与 `M` 后缀（`1.5K`、`0.5M`）。

未加引号的上限会先由 Router 的 YAML 解码器按 YAML 1.1 规则定类型再解析：`0123` 是八进制 83，`0x10` 是 16，`1_000` 是 1000，`1:30` 不是数字。加引号可保持字面值，因此 `'0123'` 是 123。`vllm-sr` CLI 使用同一套定类型规则，因此其校验与 Router 从转发文件加载的结果一致。

上限可以引用环境变量，例如 `${CTX_MIN}`。Router 在配置加载时展开它，因此 CLI 会带警告接受该档，而不是检查它。

```yaml
routing:
  signals:
    context:
      - name: short_context
        min_tokens: 0
        max_tokens: 8K
      - name: medium_context
        min_tokens: 8001
        max_tokens: 64K
      - name: long_context
        min_tokens: 64001
        description: Open-ended band; matches everything above 64K tokens.
```

上下文档只是路由信号。它们不改变模型上下文窗口上限；Router 仍会在筛选候选时单独强制这些上限。

## 依赖与限制 {#dependencies-and-limitations}

token 估计取决于请求表示，并不保证后端会接受得到的提示。请保持模型卡上下文窗口准确，并为生成输出留出余量。选择前，Router 会移除已配置且为正、但上下文窗口小于估计请求的决策候选。缺少上下文元数据的候选仍可入选，以保持向后兼容；若每个候选的已知窗口都不足，Router 会拒绝请求，而不是转发给不合格后端。完整示例见：
[`config/fragments/signal/context/long-context.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/context/long-context.yaml)。

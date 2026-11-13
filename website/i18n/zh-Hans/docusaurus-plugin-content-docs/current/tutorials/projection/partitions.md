---
is_mtpe: false
sidebar_position: 2
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/tutorials/projection/partitions.md"
  outdated: false
---

# 分区

## 概览

`recipes[].projections.partitions` 协调竞争的 `domain` 信号，并只保留一个胜者。

## 解决什么问题？

没有分区时，一个请求可能同时匹配若干相近的领域。这对路由通常不理想：

- 一个请求通常应有一个主要领域胜者，而不是四个部分匹配的领域
- 当每个决策都必须防御重叠匹配时，重复的路由规则更难推理

分区通过在信号提取之后、决策评估之前协调检测器结果来解决该问题。

## 分区在运行时如何表现 {#how-partitions-behave-at-runtime}

分区遵循这些规则：

- 分区只接受 `domain` 成员
- `default` 是必需的，并且也必须出现在 `members` 中
- 只要有决策或投影读取任一成员，运行时就评估所有成员，因此胜者不取决于决策读取了哪些成员
- 若多个成员匹配，运行时只保留一个胜者
- 若没有成员匹配，运行时把 `default` 成员合成进匹配集合
- 若没有成员匹配且某个成员的信号失败，运行时不合成 `default`，并把 `default` 成员也标记为失败：读取失败成员或 `default` 的决策按其 `on_unknown` 策略处理

支持的语义：

- `exclusive`：按原样保留最高置信度胜者
- `softmax_exclusive`：保留同一胜者，但用带 `temperature` 的 softmax 重新归一化竞争者置信度

两个实际后果：

- 决策仍以 `type: domain` 引用获胜成员
- 决策不引用分区名本身

因此分区不是映射那种“命名投影输出”。它们是对现有信号名的协调。

## 配置

```yaml
recipes:
  - name: default
    projections:
      partitions:
        - name: balance_domain_partition
          semantics: softmax_exclusive
          temperature: 0.10
          members: [law, business, health, history, other]
          default: other
```

## DSL {#dsl}

```dsl
PROJECTION partition balance_domain_partition {
  semantics: "softmax_exclusive"
  temperature: 0.10
  members: ["law", "business", "health", "history", "other"]
  default: "other"
}
```

## 配置字段 {#config-fields}

| 字段 | 含义 |
|-------|---------|
| `name` | 用于配置和 DSL 的分区标识符 |
| `semantics` | 胜者选择模式：`exclusive` 或 `softmax_exclusive` |
| `temperature` | 仅对 `softmax_exclusive` 有意义；更低的值让胜者更果断 |
| `members` | 要协调的现有 `domain` 信号名 |
| `default` | 当所有成员都已评估且没有任何成员匹配时合成的回退成员 |

## 何时使用

在以下情况使用分区：

- 一个请求在路由前应有一个主导领域
- 希望下游决策保持简单，并直接读取获胜的原始信号

## 何时不使用 {#when-not-to-use}

不要在以下情况使用分区：

- 多个成员应对决策保持独立可见
- 该组混合了不应互相竞争的无关概念
- 互相竞争的通道是意图而不是学科领域；用一个 [`systemone`](../signal/systemone) `choice` 问题从中选出一个
- 需要可复用的命名档位，例如 `balance_reasoning`；那属于映射，而不是分区

分区不提出自己的问题；它们协调成员信号的结果，只要有决策读取任一成员，所有成员信号都会运行。配置的默认值是路由回退，不是默认值实际匹配的证据。完整示例见
[`config/config.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml)。

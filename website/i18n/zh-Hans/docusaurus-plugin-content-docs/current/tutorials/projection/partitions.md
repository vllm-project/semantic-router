---
translation:
  source_commit: "45bfd49e"
  source_file: "docs/tutorials/projection/partitions.md"
  outdated: false
sidebar_position: 2
---

# 分区（Partitions）

## 概览

`routing.projections.partitions` 协调竞争的 `domain` 或 `embedding` 信号，并保留**一个胜者**。

在以下情况使用分区：

- 多个相关领域或嵌入信号可能对同一请求同时匹配
- 下游路由应基于**单一解析胜者**，而非多个重叠匹配
- 需要在无任何成员命中时提供回退行为

## 主要优势

- 在决策运行前将竞争的领域或嵌入匹配收敛为单一胜者。
- 无成员清晰胜出时提供稳定默认回退。
- 下游决策保持简单——读取解析后的原始信号，而非分区逻辑。
- 支持基于 softmax 的置信度感知胜者选择。

## 解决什么问题？

没有分区时，请求可能同时匹配多个相近领域或嵌入通道，对路由往往不理想：

- 通常应有一个主领域胜者，而非四个部分匹配的领域
- 意图通道通常应先收敛为最佳嵌入类别，再供决策评估
- 若每个决策都要防御重叠匹配，路由规则会变难推理

分区在信号提取之后、决策评估之前协调检测器结果。

## 运行时行为

当前实现中：

- 分区仅接受 `domain` 或 `embedding` 成员
- 同一分区内所有成员必须类型相同
- `default` 必填，且必须出现在 `members` 中
- 若多个成员匹配，运行时保留一个胜者
- 若无成员匹配，运行时将 `default` 合成进匹配集合

支持的语义：

- `exclusive`：保留最高置信胜者不变
- `softmax_exclusive`：胜者相同，但用 `temperature` 对候选置信度做 softmax 重归一

两条实际后果：

- 决策仍用原生类型引用胜者，如 `type: domain` 或 `type: embedding`
- 决策不引用分区名本身

因此分区与映射意义上的「命名投影输出」不同：它们是对已有信号名的协调。

## 规范 YAML

```yaml
routing:
  projections:
    partitions:
      - name: balance_domain_partition
        semantics: softmax_exclusive
        temperature: 0.10
        members: [law, business, health, history, other]
        default: other

      - name: balance_intent_partition
        semantics: softmax_exclusive
        temperature: 0.18
        members: [code_general, architecture_design, research_synthesis, general_chat_fallback]
        default: general_chat_fallback
```

## DSL

```dsl
PROJECTION partition balance_intent_partition {
  semantics: "softmax_exclusive"
  temperature: 0.18
  members: ["code_general", "architecture_design", "research_synthesis", "general_chat_fallback"]
  default: "general_chat_fallback"
}
```

## 配置字段

| 字段 | 含义 |
|------|------|
| `name` | 配置与 DSL 中的分区标识 |
| `semantics` | 胜者选择模式：`exclusive` 或 `softmax_exclusive` |
| `temperature` | 仅对 `softmax_exclusive` 有意义；值越低胜者越「果断」 |
| `members` | 要协调的已有 `domain` 或 `embedding` 信号名 |
| `default` | 无成员匹配时的回退成员（会被合成进匹配集合） |

## 配置

分区位于 `routing.projections.partitions`。每个分区需要 `name`、`semantics`（`exclusive` 或 `softmax_exclusive`）、`members` 列表与 `default`。完整字段说明见上文 [规范 YAML](#规范-yaml) 与 [配置字段](#配置字段)。

## 何时使用

在以下情况使用分区：

- 路由前请求应有一个主导领域
- 若干嵌入通道代表互斥意图，应收敛为单一胜者
- 希望下游决策保持简单，直接读取胜出的原始信号

## 何时不用

在以下情况不要使用分区：

- 多个成员应对决策**独立**可见
- 分组混合了不应相互竞争的不相关概念
- 需要可复用的命名档位如 `balance_reasoning`——那属于映射，而非分区

## 设计说明

- 原始检测器定义放在 `routing.signals`；分区只协调它们。
- 将属于同一路由问题的成员归组，例如同一领域族或同一嵌入族。
- 当无成员清晰胜出时，若下游仍需稳定回退，请添加 `default`。
- 若在嵌入分区上使用 `softmax_exclusive`，当成员质心过于接近难以区分时，原生 DSL 校验可能发出警告。

## 下一步

- 解析出的信号仍需参与加权路由分数时，配合 [Scores](./scores)。
- 决策应读取命名档位而非原始信号胜者时，使用 [Mappings](./mappings)。

---
translation:
  source_commit: "707b84d7"
  source_file: "docs/tutorials/signal/heuristic/structure.md"
  outdated: false
---

# Structure 信号

## 概览

`structure` 检测请求形态相关事实，例如多个显式问题、有序工作流标记或密集的约束措辞。映射到 `config/signal/structure/`，在 `routing.signals.structure` 中声明。

该族为启发式：保持基于规则，但与 `keyword` 不同，可在发出命名信号前对类型化结构特征计数、归一化与比较。

## 主要优势

- 请求形态路由显式，而非藏在临时关键词列表里。
- 单个检测器可使用计数、密度或有序标记序列，而无需改决策 DSL。
- 产生可复用命名信号，投影与决策可像其他族一样消费。
- 保持仓库分层：检测阈值在信号中，路由策略在决策中。

## 解决什么问题？

部分路由取决于提示**写法**，而非仅主题。含五个问题的提示，或含「先…再…」的提示，往往与单句短问需要不同路由，即使领域相同。

`structure` 将请求形态特征变成稳定命名信号。

## 何时使用

在以下情况使用 `structure`：

- 路由依赖问题数量、列表形态或有序工作流标记
- 需要归一化计数或阈值，而非仅原始关键词是否存在
- 检测器仍为规则型，不需要学习模型
- 希望投影用 `type: structure` 消费结构事实

## 配置

源片段族：`config/signal/structure/`

```yaml
routing:
  signals:
    structure:
      - name: many_questions
        description: Prompts with many explicit questions.
        feature:
          type: count
          source:
            type: regex
            pattern: '[?？]'
        predicate:
          gte: 4
      - name: at_most_one_question
        description: Prompts with one or fewer explicit questions.
        feature:
          type: count
          source:
            type: regex
            pattern: '[?？]'
        predicate:
          lte: 1
      - name: numbered_steps
        description: Prompts that contain numbered list items such as "1. ..."
        feature:
          type: exists
          source:
            type: regex
            pattern: '(?m)^\s*\d+\.\s+'
      - name: first_then_flow
        description: Prompts that express an ordered workflow.
        feature:
          type: sequence
          source:
            type: sequence
            case_sensitive: false
            sequences:
              - ["first", "then"]
              - ["first", "next", "finally"]
              - ["首先", "然后"]
              - ["先", "再"]
      - name: constraint_dense
        description: Constraint language is dense relative to multilingual text units.
        feature:
          type: density
          source:
            type: keyword_set
            case_sensitive: false
            keywords:
              - under
              - at most
              - at least
              - within
              - no more than
              - 不超过
              - 至少
              - 最多
        predicate:
          gt: 0.08
      - name: format_directive_dense
        description: Output-format directives are dense relative to multilingual text units.
        feature:
          type: density
          source:
            type: keyword_set
            keywords:
              - table
              - bullet
              - json
              - markdown
              - 表格
              - 列表
              - JSON
        predicate:
          gt: 0.08
      - name: low_question_density
        description: Prompts with very low question density relative to multilingual text units.
        feature:
          type: density
          source:
            type: regex
            pattern: '[?？]'
        predicate:
          lt: 0.05
```

`feature.type` 定义如何计算值。`feature.source` 定义扫描对象。`predicate` 将数值或布尔结果转为命名匹配信号。

当前支持的约定：

- `feature.type`：`exists`、`count`、`density`、`sequence`
- `feature.source.type`：`regex`、`keyword_set`、`sequence`
- `predicate`：`gt`、`gte`、`lt`、`lte`

说明：

- `exists` 不接受 predicate；源存在即发出匹配。
- `density` 按多语言文本单位自动归一化。CJK 字符单独计数，连续非 CJK 字母数字算一个单位，标点忽略。
- `sequence` 要求 `feature.source.type=sequence`。
- `keyword_set` 使用脚本感知匹配，使连续 CJK 与混写提示仍能正确命中。
- 本族中 `regex` 为真正的正则源。

示例含义：

- `many_questions`：统计 `?` 或 `？` 个数，至少四个则匹配。
- `at_most_one_question`：零个或一个问号则匹配。
- `numbered_steps`：提示中已有如 `1. ...` 的编号列表项则匹配。
- `first_then_flow`：出现有序工作流标记序列，如 `first ... then ...` 或 `先 ... 再 ...`。
- `constraint_dense`：统计约束标记并按多语言单位归一，捕捉英中混写下约束异常密集的提示。

当路由依赖请求形态，但仍希望路由器约定保持类型化、声明式时，使用 `structure`。

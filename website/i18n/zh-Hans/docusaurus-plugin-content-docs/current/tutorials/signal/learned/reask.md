---
translation:
  source_commit: "c904264a"
  source_file: "docs/tutorials/signal/learned/reask.md"
  outdated: false
---

# Reask 信号

## 概览

`reask` 用于检测当前用户轮次是否在语义上重复同一对话中最近的用户轮次。它对应 `config/signal/reask/`，并在 `routing.signals.reasks` 下声明。

该信号族属于学习型信号：它使用路由器共享的语义 embedding 路径，将当前用户轮次与之前的用户轮次进行比较。

## 主要优势

- 无需用户明确说出“这是错的”等措辞，也能捕获隐含的不满。
- 区分一次重复提问和连续多轮重复所形成的不满趋势。
- 允许 decision 根据最近的对话历史升级，而不是只依据单条消息。
- 复用现有的语义相似度技术栈，不引入第二套模型能力面。

## 解决什么问题？

当上一次回答没有帮助时，用户通常会重新表述同一个问题。单轮分类器可能无法识别这种模式，因为用户的不满是隐含的，而非明确表达的。

`reask` 通过比较最新用户轮次与最近的用户轮次来解决这一问题；当连续的近期轮次在语义上保持相似时，它会产生可配置的不满信号。

## 何时使用

在以下场景使用 `reask`：

- 重复的问题应升级到更强的模型
- 一次重复提问与多次重复提问需要采用不同的处理方式
- 显式反馈较少，但用户轮次的重复仍具有意义
- 路由 decision 应依赖同一对话中的用户历史

## 配置

源片段信号族：`config/signal/reask/`

```yaml
routing:
  signals:
    reasks:
      - name: likely_dissatisfied
        description: Current user turn closely repeats the immediately previous user turn.
        threshold: 0.8
        lookback_turns: 1
      - name: persistently_dissatisfied
        description: Current user turn repeats the last two user turns in a row.
        threshold: 0.8
        lookback_turns: 2
```

每条规则都会将当前用户轮次与最近的 `lookback_turns` 个历史用户轮次进行比较。只有该连续近期序列中的每个轮次都高于配置的相似度阈值时，规则才会匹配。

---
translation:
  source_commit: "9649d02f2582471642196bde2e086c34ccb9c88c"
  source_file: "docs/tutorials/plugin/prompt-cache.md"
  outdated: false
---

# 提示词缓存

## 概览 {#overview}

`prompt_cache` 是路由本地的插件，在匹配的路由上插入 Anthropic 提示词缓存断点。它从不声称命中缓存、也不报告省了多少；它只是请 provider 考虑把标过的内容缓存起来。

## 关键优势 {#key-advantages}

- 缓存断点策略挂在受益的那条路由上，不用每个调用方自己管标记。
- 调用方已经在管自己的标记，它绝不覆盖。
- 默认关闭，在一条路由上打开，不会动别的路由。

## 解决什么问题 {#what-problem-does-it-solve}

Anthropic 支持在可重用的提示词块上打显式缓存断点。用中性请求格式的客户端，没法自己放这些 Anthropic 专有的标记。`prompt_cache` 让路由能标出重复的指令和工具定义，不用每个客户端都懂 Anthropic 的线上格式。它不开启 Anthropic 另有一套的顶层自动缓存模式。

## 什么时候用 {#when-to-use}

- 走 Anthropic 的路由，多数轮次发同一套系统指令或工具定义
- 这条路由上的调用方没有自己在发 `cache_control` 标记
- 路由能接受固定的 5 分钟或 1 小时缓存寿命

路由上的流量已经在打自己的标记，或所选模型不会 Anthropic Messages 线上格式，就跳过。

## 配置 {#configuration}

插件加在 `routing.decisions[].plugins` 下：

```yaml
plugins:
  - type: prompt_cache
    configuration:
      enabled: true
      ttl: 1h
      targets: [instructions, tools]
      on_unsupported: skip
```

`ttl` 收 `5m`（默认）或 `1h`，对应 Anthropic 支持的两种寿命。`targets` 收 `instructions`、`tools`，或两个都要（默认两个）。`on_unsupported` 管所选后端不会 Anthropic Messages 线上格式时怎么办：`skip`（默认）请求原样放过，`reject` 返带类型错误码 `prompt_cache_target_unsupported`。

## 标记怎么放 {#marker-placement}

启用后，只要中性请求里没有任何调用方提供的缓存标记，router 就确定性地在最后一个合格的文本指令块加至多一个标记、在最后一个合格的工具上加至多一个标记——一条路由一个请求至多两个 router 插入的标记。请求里只要已有任何调用方标记，调用方的意图说了算：现有标记全保留，router 一个不加。用户消息、助手消息、工具调用和工具结果，这个插件不标。

Anthropic 一个请求支持至多四个显式缓存断点。这个插件故意远在限额之下：它只盯最可能跨轮稳定的两块，而且调用方一旦接管缓存位置，它绝不再加。provider 侧的完整契约见 Anthropic 的[提示词缓存文档](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)。

## 运行时次序 {#runtime-order}

标记插入跑在路由之后、请求/上下文/工具变换之后，紧挨着请求编码上 Anthropic 线上格式之前。它只碰发出去的 Anthropic 形状请求；路由和其他插件评估的那个中性请求，它永远不改。

## 可观测性 {#observability}

每次评估记一条 `llm_plugin_execution_total`，`plugin_type=prompt_cache`，`status` 是 `inserted`、`preserved`、`skipped` 或 `rejected`。请求带 `x-vsr-debug: true` 时，同一个结果内联出现在 `x-vsr-prompt-cache-action`、`x-vsr-prompt-cache-reason`、`x-vsr-prompt-cache-inserted` 和 `x-vsr-prompt-cache-preserved`；见 [VSR 路由头](../../troubleshooting/vsr-headers)。这个插件不碰 Router Replay 持久化。

完整示例见 [`config/fragments/plugin/prompt-cache/anthropic-agent.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/prompt-cache/anthropic-agent.yaml)。

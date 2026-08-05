---
translation:
  source_commit: "15c6c555"
  source_file: "docs/tutorials/plugin/request-params.md"
  outdated: false
---

# 请求参数（Request Parameters）

## 概览

`request_params` 是一个路由局部插件，用于在 OpenAI Chat Completions 请求体转发到后端之前对其进行校验和裁剪。

它对应 `config/plugin/request-params/budget-tier.yaml`。

## 主要优势

- 按路由限制高成本参数（`max_tokens`、`n`）。
- 对不应暴露 token 分布的层级，阻止 `logprobs` / `top_logprobs` 等敏感参数。
- 可以选择移除未知的顶层 JSON 字段，减少透传行为带来的意外。

## 解决什么问题？

模型路由可以限制由哪个后端处理请求，但客户端仍然可以通过请求参数放大成本或提取 logits。该插件会在 decision 匹配后，对请求体执行按 decision 配置的限制。

## 何时使用

- 某个层级或路由不得请求 logprobs 或多条 completion
- 需要对 `max_tokens` 或 `n` 设置独立于客户端输入的硬上限
- 不应将未知 JSON 字段转发到后端

## 配置

在 `routing.decisions[].plugins` 下使用此片段（插件条目列表）：

```yaml
plugins:
  - type: request_params
    configuration:
      blocked_params:
        - logprobs
        - top_logprobs
      max_tokens_limit: 500
      max_n: 1
      strip_unknown: true
```

在 DSL 中，同一个插件可以写成：

```dsl
PLUGIN request_params {
  blocked_params: ["logprobs", "top_logprobs"]
  max_tokens_limit: 500
  max_n: 1
  strip_unknown: true
}
```

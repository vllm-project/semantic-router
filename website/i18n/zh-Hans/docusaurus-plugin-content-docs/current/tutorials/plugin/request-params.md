---
translation:
  source_commit: "bef43a20a96e7b329d314473083be6aa4bad0f03"
  source_file: "docs/tutorials/plugin/request-params.md"
  outdated: false
---

# 请求参数

## 概览

`request_params` 是一个路由局部插件，在将 OpenAI Chat Completions 请求正文转发给后端之前进行校验和裁剪。

## 主要优势

- 按路由限制昂贵参数（`max_tokens`、`n`）。`max_tokens_limit` 是补全 token 上限：客户端未设置上限时注入该值，否则与 `modelRefs[].max_completion_tokens` 以及任何 Looper 阶段上限按最严格值合成。
- 对不应暴露 token 分布的层级阻止敏感参数，例如 `logprobs` / `top_logprobs`。
- 可选剥离未知的顶层 JSON 字段，以减少意外透传。

## 解决什么问题？

模型路由可以限制由哪个后端服务请求，但客户端仍可通过请求参数放大成本或提取 logits。该插件在决策匹配后对请求正文强制按决策的限制。

## 何时使用

- 某个层级或路由不得请求 logprobs 或多次补全
- 需要对 `max_tokens` 或 `n` 设置与客户端输入无关的硬上限。当模型应使用不同上限时，将 `max_tokens_limit` 与 `routing.decisions[].modelRefs[].max_completion_tokens` 配对；派发保留最严格的上限。
- 未知 JSON 字段不应转发给后端

## 配置

在 `routing.decisions[].plugins` 下添加该插件：

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

在 DSL 中，同一插件可以写成：

```dsl
PLUGIN request_params {
  blocked_params: ["logprobs", "top_logprobs"]
  max_tokens_limit: 500
  max_n: 1
  strip_unknown: true
}
```

该插件强制一组有界的 OpenAI Chat Completions 字段。它不是通用 JSON schema 防火墙，也不授权调用方。`max_tokens_limit` 不能抬高更小的客户端 `max_completion_tokens` / `max_tokens` 值，并且阻止这些字段会在合成前移除客户端贡献。在启用 `strip_unknown` 之前，请针对会添加提供商专用字段的客户端进行测试。完整示例见：
[`config/fragments/plugin/request-params/budget-tier.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/request-params/budget-tier.yaml)。

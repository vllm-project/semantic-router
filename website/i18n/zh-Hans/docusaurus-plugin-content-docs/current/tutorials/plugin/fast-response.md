---
translation:
  source_commit: "7c874be2"
  source_file: "docs/tutorials/plugin/fast-response.md"
  outdated: false
---

# 快速响应（Fast Response）

## 概览

`fast_response` 是一个路由局部插件，用于立即返回确定性的回退消息。

## 主要优势

- 当轻量级回退已经足够时，短路高成本路由。
- 将过载处理行为限制在需要它的路由内。
- 在配置中明确声明回退消息。

## 解决什么问题？

有些路由应优雅降级，而不是等待完整的模型处理路径。`fast_response` 为这些路由提供即时响应路径，同时不改变全局行为。

## 何时使用

- 路由在过载或维护期间需要低成本回退
- 该流量类别可以接受确定性响应
- 回退行为应仅作用于单个路由

## 配置

在 `routing.decisions[].plugins` 下添加该插件：

```yaml
plugins:
  - type: fast_response
    configuration:
      message: The primary model is unavailable. Try again shortly.
```

该插件不会调用模型，也不会生成响应内容。消息中不得包含请求数据，并且不要将该插件用作身份验证或速率限制控制。插件本身不会测量过载状态；decision 必须匹配应接收回退响应的流量。完整示例见：
[`config/fragments/plugin/fast-response/busy.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/fast-response/busy.yaml)。

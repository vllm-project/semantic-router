---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/plugin/fast-response.md"
  outdated: false
---

# Fast Response

## 概览

`fast_response` 是路由局部插件：立即返回确定性回退消息。

对应 `config/plugin/fast-response/busy.yaml`。

## 主要优势

- 轻量回退足够时短路昂贵路由。
- 过载行为局部在需要的路由。
- 回退消息在配置中显式。

## 解决什么问题？

部分路由应优雅降级，而非等待完整模型路径。`fast_response` 为这些路由提供立即响应路径而不改全局行为。

## 何时使用

- 路由在过载或维护条件下需要廉价回退
- 对该流量类别可接受确定性响应
- 回退行为应仅局部在一条路由

## 配置

在 `routing.decisions[].plugins` 下使用：

```yaml
plugin:
  type: fast_response
  configuration:
    message: The primary model is saturated, so a lightweight response was returned immediately.
```

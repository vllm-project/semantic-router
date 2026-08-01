---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/plugin/router-replay.md"
  outdated: true
---

# Router Replay

## 概览

`router_replay` 是路由局部插件：捕获重放/调试产物。

对应 `config/plugin/router-replay/debug.yaml`。

## 主要优势

- 重放捕获局部在活跃调试或审计的路由上。
- 支持请求与响应体控制。
- 存储限制显式而非隐式。

## 解决什么问题？

重放有用，但并非每条路由应记录相同数据量。`router_replay` 让单条路由选用重放行为而不全局化存储成本。

## 何时使用

- 一条路由处于调试、审计或受控重放分析
- 捕获限制应对每条路由显式
- 仅对选中流量启用重放

## 配置

在 `routing.decisions[].plugins` 下使用：

```yaml
plugin:
  type: router_replay
  configuration:
    enabled: true
    max_records: 5000
    capture_request_body: true
    capture_response_body: false
    max_body_bytes: 65536
```

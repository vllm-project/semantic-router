---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/global/api-and-observability.md"
  outdated: true
---

# API 与可观测性

## 概览

本页介绍暴露接口与遥测的共享运行时块。

这些设置为全路由器级，属于 `global:`，而非路由局部插件片段。

## 主要优势

- 跨路由保持可观测性与接口控制一致。
- 避免在路由局部配置中重复指标或 API 设置。
- 将重放与响应 API 显式为共享服务。
- 运维控制集中在全路由器一层。

## 解决什么问题？

若 API 与遥测按路由配置，运维面会碎片化、难推理。

`global:` 的这部分将共享接口与监控设置集中在一处。

## 何时使用

在以下情况使用这些块：

- 路由器应暴露共享 API
- 响应 API 应对整台路由器启用
- 指标与追踪应一次配置
- 重放捕获作为共享运维服务保留

## 配置

### API

```yaml
global:
  services:
    api:
      enabled: true
```

### 响应 API

```yaml
global:
  services:
    response_api:
      enabled: true
      store_backend: memory
```

### 可观测性

```yaml
global:
  services:
    observability:
      metrics:
        enabled: true
```

### Router Replay

```yaml
global:
  services:
    router_replay:
      async_writes: true
```

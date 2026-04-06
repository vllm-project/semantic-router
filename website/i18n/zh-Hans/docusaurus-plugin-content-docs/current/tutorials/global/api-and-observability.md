---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/global/api-and-observability.md"
  outdated: false
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
      store_backend: redis        # 默认值；仅在本地开发时使用 "memory"
      redis:
        address: "redis:6379"
```

`store_backend` 用来控制响应与会话历史保存到哪里。可用后端如下：

| 后端 | 持久性 | 适用场景 |
| --- | --- | --- |
| `redis` | 路由器重启后仍保留，且可跨副本共享 | 生产默认 |
| `memory` | 路由器重启后丢失 | 仅本地开发 |

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
      store_backend: postgres     # 默认值；适合 SQL 可查询的审计留存
      enabled: true
      async_writes: true
      postgres:
        host: postgres
        port: 5432
        database: vsr
        user: router
        password: router-secret
```

`global.services.router_replay.enabled` 是全路由器默认值。启用后，除非某个 decision 在本地 `router_replay` 插件里显式写 `enabled: false`，否则该 decision 会采集 replay 记录。

`store_backend` 用来控制路由决策 replay 记录保存到哪里。可用后端如下：

| 后端 | 持久性 | 适用场景 |
| --- | --- | --- |
| `postgres` | 提供完整 SQL 查询能力，适合长期审计留存 | 生产默认 |
| `redis` | 路由器重启后仍保留，且可跨副本共享 | 已有 Redis 的轻量部署 |
| `milvus` | replay 记录可做向量搜索 | 需要语义 replay 搜索 |
| `memory` | 路由器重启后丢失 | 仅本地开发 |

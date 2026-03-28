---
translation:
  source_commit: "0ee41b5f"
  source_file: "docs/tutorials/global/stores-and-tools.md"
  outdated: false
---

# 存储与工具

## 概览

本页介绍 `global:` 内的共享存储与工具块。

这些设置支撑路由局部插件与全路由器工具行为。

## 主要优势

- 集中共享后端存储，而非每条路由重复。
- 语义缓存、内存、检索与工具目录保持一致。
- 路由局部插件保持小而专注。
- 共享基础设施依赖显式。

## 解决什么问题？

路由局部插件常依赖共享存储或工具状态。若依赖在各路由内随意配置，系统不一致且难运维。

这些 `global:` 块将共享后端服务定义一次。

## 何时使用

在以下情况使用这些块：

- 多条路由依赖同一语义缓存或内存后端
- 检索功能需要单一共享向量库
- 路由器应暴露共享工具目录
- 后端存储配置属于整台路由器而非单条路由

## 配置

### 语义缓存

```yaml
global:
  stores:
    semantic_cache:
      similarity_threshold: 0.8
```

### Memory

```yaml
global:
  stores:
    memory:
      enabled: true
```

### 向量库

```yaml
global:
  stores:
    vector_store:
      provider: milvus
```

### 工具

```yaml
global:
  integrations:
    tools:
      enabled: true
      top_k: 3
      tools_db_path: deploy/examples/runtime/tools/tools_db.json
```

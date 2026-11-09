---
is_mtpe: false
title: 数据与存储
description: 为缓存、记忆、回放、响应历史和向量数据选择支撑服务。
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/installation/storage-overview.md"
  outdated: false
---

# 数据与存储

Semantic Router 可以在没有外部向量数据库的情况下运行，但若干可选能力会持久化或共享数据。按能力、持久性、规模和数据处理策略选择存储，而不是仅按产品名称选择。

## 可能存储的内容

| 能力 | 典型数据 | 为何存储 |
| --- | --- | --- |
| 响应缓存 | 请求表示和先前响应 | 为等价请求复用先前答案。 |
| Agentic memory | 已嵌入的记忆和元数据 | 跨轮次或会话检索相关信息。 |
| 向量存储 | 文档、分块、嵌入和文件元数据 | 支持检索和已上传知识。 |
| 路由回放 | 路由元数据，以及（启用时）有界的请求和响应正文 | 检查并评估路由行为。 |
| Response API | 近期响应记录 | 通过管理界面检索已生成的响应。 |

这些存储具有不同的隐私和保留含义。如果提示词或响应被写入未经批准的共享存储，本地模型路由就不是端到端私密的。

## 可用指南

### Valkey

[Valkey 智能体记忆](valkey-memory) 介绍了一种兼容 Redis、带向量搜索的单服务选项。它适用于较小部署，或已经在运维 Valkey 的团队。

### Qdrant 向量库

[Qdrant](qdrant) 覆盖 Docker 和 Kubernetes 部署以及 Router 绑定。它可以支撑语义缓存、记忆、向量存储和回放配置。

### Milvus 向量库

[Milvus](milvus) 覆盖持久向量存储、Kubernetes 部署、监控、迁移和恢复。它适合较大或共享的向量集合，但比单服务存储有更多运维依赖。

### 内存存储

内存后端适用于本地实验和短期缓存。它们不提供重启持久性或跨副本共享。不要推断“临时”就意味着不敏感：内容在运行进程内仍可读，并且可能出现在日志或诊断中。

## 选择问题

在启用存储之前，先决定：

1. 哪个 Router 能力会使用它？
2. 数据是否必须在 Router 重启后保留？
3. 多个 Router 副本是否必须共享同一状态？
4. 可以捕获哪些请求或响应内容？
5. 保留、删除、加密和访问要求是什么？
6. 谁负责备份和恢复该服务？

然后在 `global.stores` 或相关服务块下配置共享存储，并仅启用需要它的路由插件。canonical 配置表面见[存储与工具](../tutorials/global/stores-and-tools)。

## Router 本地磁盘上的文件 {#router-local-files}

有两项能力把文件保存在 Router 自己的磁盘上，而不是保存在某个服务中：

| 能力 | 设置 | 未设置时 |
| --- | --- | --- |
| 向量存储上传的文件 | `global.stores.vector_store.file_storage_dir` | `/var/lib/vsr/data` |
| Router Flow 工具状态（`file` 后端） | `global.integrations.looper.flow.state.file.directory` | 用户缓存目录 |

`/var/lib/vsr` 是 Router 的数据目录，下面每种部署方式都会让它可写。本地存储路径都应放在它下面：

```yaml title="vector-store-files"
global:
  stores:
    vector_store:
      enabled: true
      backend_type: memory
      file_storage_dir: /var/lib/vsr/data
```

Router 启动时，在下载任何模型之前，会检查配置用到的每个目录是否可写。不可写时，它会停止启动，并给出指明该设置的错误：

```text
check local storage: global.stores.vector_store.file_storage_dir: /var/lib/vsr/data is not writable (/var/lib: read-only file system); mount a writable volume at that path, or set global.stores.vector_store.file_storage_dir to a writable directory
```

使用此类目录的配置更新会以原因 `storage_unavailable` 被拒绝，正在运行的配置保持生效。

### Helm

Chart 以只读根文件系统运行 Router，并在 `/var/lib/vsr` 挂载一个可写卷。默认情况下该卷是 `emptyDir`，新 Pod 启动时它是空的：即使 `metadata_store: postgres` 保留了文件记录，Pod 被替换后上传的文件也会丢失。若要保留它们，把数据目录放到 PersistentVolumeClaim 上：

```yaml title="values.yaml"
data:
  persistence:
    enabled: true
    size: 20Gi
```

`storageClassName`、`accessMode`、`annotations` 和 `existingClaim` 的用法与模型卷 `persistence` 相同。除非 claim 是 `ReadWriteMany`，每个 Router 副本都有自己的数据目录；因此在多副本时，通过一个副本上传的文件无法通过另一个副本读取。用 `extraVolumes` 和 `extraVolumeMounts` 挂载到 `/var/lib/vsr` 的卷会替换 Chart 自带的卷。

Router 的其他可写状态位于模型卷 `/app/models`（已下载的模型、运行时缓存，以及单副本的配置历史）和 `/tmp`（运行时套接字、启动状态和默认的 Flow 状态目录）。

### 本地 Docker 栈

`vllm-sr serve` 在可写文件系统上运行 Router，因此默认值可以直接使用；但 `/var/lib/vsr` 位于 Router 容器内部，新容器启动时没有其中的内容。若要在重启后保留上传的文件，把 `file_storage_dir` 指向 `/app/.vllm-sr` 下的目录，即该栈在主机上的状态目录；`vela-amd` 配方就使用了 `/app/.vllm-sr/files`。

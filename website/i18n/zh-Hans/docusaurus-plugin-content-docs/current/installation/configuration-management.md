---
title: 配置管理
description: 运行中的 Router 如何激活一次配置变更、变更会重建什么、被拒绝的变更如何报告，以及如何列出和回滚版本。
translation:
  source_commit: "2e7e0775e88b0fc9426daf7c4fa0fe6e0ec20654"
  source_file: "docs/installation/configuration-management.md"
  outdated: false
---

# 配置管理

修改运行中 Router 配置的每一种方式都走同一套生命周期：编辑它监听的文件、管理 API、控制面板，以及 Kubernetes 资源。任何一步失败的变更都会被拒绝，正在服务的版本继续服务。通过所有步骤的变更取得下一个版本号，并一次性替换上一个版本。

## 变更如何激活

Router 把文档转换为不可变的快照，并依次经过以下阶段：

| 阶段 | 发生什么 |
| --- | --- |
| `parse` | 按 canonical 布局读取文档。 |
| `compile` | listeners、providers、recipes、entrypoints 和全局设置变成按名字互相引用的类型化资源。重复的名字或引用不存在的资源在这里被拒绝。 |
| `validate` | 对照运行中的 Router 检查候选配置：需要重启的设置、当前网关模式缺少的能力，以及所需的模型制品。 |
| `warm` | 准备模型并构建路由流水线。在 standalone 模式下，还会先构建上游连接池，并让开启健康检查的后端完成第一次检查，然后才有请求使用它们。 |
| `activate` | 新快照一步替换旧快照。 |

版本激活时正在处理的请求在它们开始时的版本上完成，最后一个请求结束后旧版本才被释放。在 standalone 模式下，一个请求从路由、fallback 链直到响应的最后一个字节都由同一个版本服务。每个经过路由的响应都在 `x-vsr-config-version` 头中给出服务它的版本。

如果候选配置激活之前被监听的文件又发生变化，该候选会作为被取代（superseded）的变更丢弃，由更新的文档重新走这套生命周期。

## 变更会重建什么

Router 只重建变更涉及的部分：

- 只修改 provider 的后端或 listener 时，已加载的分类器和 embedding 模型保持不变。
- 在 standalone 模式下，修改 recipes 或 signals 时上游连接池保持不变；修改某个 provider 时，它未变化的后端的连接池也保持不变。
- 在 standalone 模式下，Router 在启动时绑定 listeners。增删 listener，或修改 listener 的 `address`、`port`、`timeout` 或 `tls`，会以 `restart_required` 被拒绝；重启 Router 才能生效。listener 的 `api_keys` 无需重启即可修改。

## 修改本地栈的配置

在 docker 目标上，`vllm-sr serve` 会把 `config.yaml` 复制成生效的文档 `.vllm-sr/runtime-config.yaml`，Router 监视的是这个文件。之后修改 `config.yaml` 不会产生任何效果，直到你应用它：

```bash
vllm-sr config plan --config config.yaml    # 对照运行中的 Router 检查这次变更
vllm-sr config apply --config config.yaml   # 激活它
```

`vllm-sr config apply` 最多等待 120 秒让变更生效，足以加载新模型；需要更久时传入 `--timeout`。超时不会取消变更：`vllm-sr config versions` 会显示它是否已激活。

需要重启的变更（例如 listener 换了端口）会像控制面板那样被保存下来：`config apply` 提示“Restart required: run `vllm-sr serve` to apply.”，`vllm-sr status` 会报告这次保存的变更，下一次 `vllm-sr serve` 会应用它。

要改为用 `config.yaml` 替换生效的文档（包括控制面板中的编辑）并重启：

```bash
vllm-sr serve --config config.yaml --replace-active-config
```

## 版本

版本号统计的是激活次数。Router 服务的第一份配置是版本 1，之后每次激活取下一个号。Router 以历史中最新的文档重启时保持原版本号；换成其他文档则取下一个号。回滚同样会激活一个新版本。

| 位置 | 显示什么 |
| --- | --- |
| 响应头 `x-vsr-config-version` | 服务该请求的版本。 |
| `GET /api/v1/config` | 正在服务的版本和文档哈希，分别在 `x-vsr-config-version` 和 `x-vsr-config-hash` 头中。 |
| `GET /api/v1/config/hash` | `active_version`，以及有变更被拒绝后的 `last_rejection`。 |
| `llm_config_active_version` 和 `llm_config_active_info` | 正在服务的版本，以及以标签给出的版本和哈希。 |

## 被拒绝的变更

被拒绝的变更不会影响正在服务的版本。它的原因说明失败在哪里：每条原因都有 `stage`、`code`、在已知时指向文档的 `path`，以及 `message`。

| 代码 | 含义 |
| --- | --- |
| `invalid_document` | 文档无法解析，或不是 canonical 布局。 |
| `duplicate_name` | 同一类资源中有两个同名资源。 |
| `unresolved_reference` | 资源引用了不存在的资源。 |
| `invalid_resource` | 资源本身无效。 |
| `restart_required` | 变更需要重启，例如 standalone listener 的新地址或端口。 |
| `unsupported` | 变更需要当前网关模式缺少的能力；消息会指出具备该能力的 `--gateway` 模式。 |
| `artifact_unavailable` | 变更所需的模型制品不可用。 |
| `model_unavailable` | 变更所需的模型无法准备好。 |
| `build_failed` | 路由流水线或上游层无法构建。 |
| `warmup_failed` | 某个组件预热失败。 |
| `activation_failed` | 新版本无法接管。 |
| `shutting_down` | Router 正在关闭。 |
| `canceled` | 变更在完成前被取消。 |

通过管理 API 提交的配置变更会等待结果。变更激活时返回 `config_version`。变更被拒绝时返回 `status: activation_failed` 和 `activation.reasons`；文档已经持久化，请用 `PUT` 修正它，或者回滚。两者都与正在服务的版本比较，而不是与被拒绝的文档比较，因此即使该文档无法解析也能成功。`GET /api/v1/config/hash` 保留最近一次拒绝。`llm_config_updates_total{result="failed"}` 按来源和阶段统计拒绝次数，`llm_config_last_rejection_timestamp_seconds` 记录最近一次拒绝的时间。

管理审计（`GET /api/v1/observability/audit`）把每次激活、拒绝和被取代的变更记录为 `config.activate`、`config.reject` 和 `config.supersede`，并带有版本、文档哈希、来源，以及引发它的管理请求。

## 历史与回滚

Router 把每次激活记录在最近 10 个版本的历史中（`-config-history-limit` 可以修改）。每个配置文件都有自己的历史，因此配置文件位于同一目录的多个 Router 不会混用历史：

- 工作区的 `config.yaml` 把历史保存在文件旁边的 `.vllm-sr/config-backups` 中，控制面板的备份也在这里。
- 其他文件把历史保存在 `.vllm-sr/config-backups` 中以该文件命名的目录里。
- `VLLM_SR_CONFIG_BACKUP_DIR` 可以改变它的位置。

按从新到旧列出历史，并回滚到某个已记录的版本：

```bash
vllm-sr config versions
vllm-sr config rollback 3
```

`vllm-sr config rollback` 接受版本号，或更早备份的时间戳。通过 API 时，用 `GET /api/v1/config/versions` 列出历史，再把版本提交给 `POST /api/v1/config/rollback`。和每一次配置变更一样，回滚必须在 `If-Match` 中带上已持久化文档的 `ETag`，否则会以 `428` 被拒绝（见[读取和更改 Router 配置](../api/apiserver#read-and-change-router-configuration)）：

```bash
etag=$(curl -s -o /dev/null -D - http://localhost:8080/api/v1/config \
  -H "Authorization: Bearer ${VSR_MGMT_TOKEN}" | awk 'tolower($1) == "etag:" {print $2}' | tr -d '\r')
curl -X POST http://localhost:8080/api/v1/config/rollback \
  -H "Authorization: Bearer ${VSR_MGMT_TOKEN}" \
  -H "If-Match: ${etag}" \
  -H "Content-Type: application/json" \
  -d '{"config_version": 3}'
```

回滚从不修改已记录的版本：它写入已记录的文档，并把它激活为一个新版本，其 `rollback_of` 指向被恢复的版本。来自 Kubernetes 资源的版本不记录文档；请在其来源处恢复。

在 Kubernetes 上，Helm chart 把单副本的历史保存在模型卷上，因此它能在 ConfigMap 变更所需的滚动更新后保留下来。在多副本、自动扩缩容或未开启持久化时，每个 Pod 在运行期间保留自己的历史；这时请通过 ConfigMap 恢复更早的文档。

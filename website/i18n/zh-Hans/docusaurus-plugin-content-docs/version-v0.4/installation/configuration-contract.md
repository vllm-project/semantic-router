---
title: 配置契约
description: 发现、校验并扩展 canonical vLLM Semantic Router 配置，而无需重复 schema。
translation:
  source_commit: "12c2aa4feb5c5d40d90104d8b09cade1facf0bf5"
  source_file: "docs/installation/configuration-contract.md"
  outdated: false
---

# 配置契约

Go 配置类型和路由注册表是 canonical 文档的事实来源。签入的 JSON Schema 从该来源生成，并由 Router API、CLI 和控制面板消费。

```text
Go config types + routing registries
                 │
                 ├── one checked-in JSON Schema + surface catalog
                 │      ├── embedded in Router builds
                 │      ├── staged into CLI and Dashboard builds
                 │      └── consumed directly by Website and Dashboard UI
                 │
                 └── Router semantic validators
                        └── validate API used before apply
```

这种划分是有意的：

- 生成的 schema 拥有字段名称、形状、描述，以及受支持的信号、投影、算法和插件清单；
- Router 校验器拥有跨字段约束、引用、安全规则、文件系统检查、默认值和运行时可行性；
- 控制面板可以添加标签或专用控件，但会将它们合并到生成的字段上，而不是维护另一套 schema；
- API 和运行时只接受 canonical 字段名称；schema 消费者不维护别名或替代载荷。

## 发现契约

从 CLI 附带的紧凑章节和路由表面索引开始：

```bash
vllm-sr config schema
```

只跟随当前任务所需的字段目录：

```bash
vllm-sr config schema --section global.router.learning
vllm-sr config schema --section global.router.learning --expanded
vllm-sr config schema --surface signal:keyword
vllm-sr config schema --surface algorithm:multi_factor
vllm-sr config schema --full
```

查询正在运行的 Router 暴露的精确契约：

```bash
vllm-sr config schema \
  --endpoint http://localhost:8080
```

`--endpoint` 适用于每个渐进选项。Router 端点是 `GET /api/v1/config/schema`：省略 `view` 返回紧凑索引；使用 `view=section&path=...` 获取紧凑字段目录，添加 `expanded=true` 获取该章节的自包含 schema，使用 `view=surface&kind=...&name=...` 获取一个已注册的路由表面，使用 `view=full` 获取完整 JSON Schema。

控制面板在 `GET /api/router/config/schema` 代理已部署的 Router 契约。如果该 Router 端点不可用，它会回退到控制面板构建中嵌入的 schema，并用 `X-Vllm-Sr-Schema-Source: bundled` 标记响应。**Operate → Platform & Access → Schema Reference** 页面显示此来源，并在已部署契约与捆绑契约不同时发出警告。网站的[配置 Schema 参考](../api/configuration-schema)可视化当前文档发行版，而不声称代表某个部署。

Agent 不需要控制面板。在 Router 上查询 `GET /api/v1` 获取紧凑端点清单，然后用 `GET /openapi.json?path=...&method=...` 请求一项操作；仅在需要完整 API 文档时使用 `GET /openapi.json`。控制面板的 **Router API Docs** 链接是指向同一运行时文档的面向人工的代理。

每种表示都有自己的 `ETag`；当 Agent 或编辑器缓存它时，使用 `If-None-Match`。

标准 JSON Schema 描述完整的 canonical 文档。其 `x-vllm-sr` 节添加路由专用发现元数据：

- `signals`：判别器、YAML 集合、运行时观察键、决策引用能力和限定，以及条目 schema；
- `algorithms`：判别器、支持层级、执行模式和载荷 schema；
- `plugins`：判别器、描述和配置 schema；
- `projections` 和 `projection_input_types`：受支持的派生路由表面；
- `global_sections`：用于构建管理表面的 canonical 全局路径；
- `schema_endpoint` 和 `validation.endpoint`：发现和语义校验路径。

## 分两阶段校验

JSON Schema 校验尽早捕获结构错误。在应用配置之前，将编写的 YAML 发送到 `POST /api/v1/config/validate`：

```json
{
  "yaml": "version: v0.3\nrouting: {}\nglobal: {}\n"
}
```

当文档有效时，Router 返回规范化、已脱敏的 YAML。它不会改动活动配置。仅 schema 校验不是应用时保证，因为它无法证明引用、部署连通性、本地资产或跨字段策略。

校验有三个所有者。生成的 schema 拥有文档结构；Go Router 拥有路由语义；CLI 或控制面板部署代码拥有文件系统访问和进程启动等环境特定检查。

仅允许消费者在其拥有的边界上进行检查：

- CLI 离线预检可以在 Router 未运行时复现 Router 诊断，但必须从生成的契约派生字段和判别器值。Router 在启动和应用时仍是权威。
- 控制面板表单可以在保存前检查不完整的交互状态，但管理后端和 Router 校验决定结果文档是否有效。
- 迁移代码可以仅识别已退役名称以生成 canonical 配置；这些别名不是被接受的稳态字段。

不要添加消费者字段允许列表、信号/algorithm/插件清单副本，或仅属于消费者的语义规则。决定 Router 能否运行的规则应首先放在 Go 中。

## Agent 编写循环

自动化或部署 Agent 应当：

1. 获取正在运行的 Router schema 索引，回退到其捆绑的 CLI 索引；
2. 在编写时只获取相关的章节和表面 schema；
3. 使用 `schema_id`、`contract_version` 和 `ETag` 作为契约身份；
4. 从 schema 字段和路由表面引用构建最小的 canonical 文档；
5. 省略仅用于引导的 `setup` 块，并调用语义校验端点；
6. 规划变更；使用返回的 `current_etag` 在 `If-Match` 中应用可热重载的变更，或在 listener 或 provider 拓扑返回 `RESTART_REQUIRED` 时使用部署工作流（对于本地 Docker，在批准后使用显式的 `vllm-sr serve --config <candidate> --replace-active-config` 操作）；
7. 轮询 `activation_status`，并在保留变更之前探测 Envoy 数据平面。

当目标 Router 的 schema 可用时，Agent 绝不应该从示例推断字段，或发送未知键。

## 添加或更改字段

对于稳态字段，更新其 Go 类型、YAML 标签和源注释。仅当存在在结构上强制时才添加 `jsonschema:"required"`；默认值和跨字段要求仍属于语义校验。对于带判别器的路由表面，同时更新匹配的 Go 注册表。注册表覆盖会对照反射的 Go 字段检查，因此如果只在一侧添加信号、投影或 algorithm 载荷，生成会失败。

将语义校验保留在 Router 中，将平台校验放在拥有它的部署代码旁边。控制面板展示元数据可以改进生成的控件，但未经整理的新字段或全局章节仍必须可通过通用 schema 渲染器编辑。然后重新生成并运行契约检查：

```bash
make config-schema-generate
make config-schema-check
make check
```

仓库恰好跟踪一份完整 schema，位于 `src/semantic-router/pkg/configschema/router-config-v0.3.schema.json`。生成器还会发出一个小型 TypeScript 导入/类型适配器，但没有第二份 JSON 副本。打包会将 canonical 产物暂存到 wheel 和容器镜像中，而不会把生成文件写回工作树。当 canonical 产物或适配器过期时，CI 会失败。

`setup.mode` 是仅用于引导的控制面板控制平面元数据，因此 Router schema 不会发布它。控制面板在激活时移除它；活动 Router 文档和对 `/api/v1/config/validate` 的调用不得包含它。

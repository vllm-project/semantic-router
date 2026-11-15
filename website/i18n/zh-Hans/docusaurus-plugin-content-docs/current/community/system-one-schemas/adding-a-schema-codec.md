---
title: 添加 schema codec
description: 如何向 Router 的 codec 注册表添加一个 System One schema，例如标准的新版本。
is_mtpe: false
translation:
  source_commit: "6b782822edead98397acc6401112e6ec1e45af96"
  source_file: "docs/community/system-one-schemas/adding-a-schema-codec.md"
  outdated: false
---

# 添加 schema codec {#adding-a-schema-codec}

一个 schema codec 使用 System One API 的一个版本。添加一个 codec（例如把标准的新版本作为 `systemone/0.3`）不会影响任何调用方：配置通过注册表解析 `schema`，Router 的客户端和 listener 路径都经过 codec 接口。

1. **描述规范。** 把规范文件放到 `config/schemas/systemone` 下，并在该目录的 README 中列出。对于标准，用我们自己的话描述接口（字段名、类型和结构），附上 `source` 和核对时的 `source_sha256`；不要复制没有许可证的文档。运行 `make systemone-schemas-drift`，把描述与已发布的文档比较。
2. **重新生成表格。** `make systemone-schemas-generate` 把文件及其 sha256 记录在 `pkg/systemonecodec/zz_generated_specs.go` 中；注册表拒绝规范不匹配的 codec。
3. **编写 codec**，放在 `src/semantic-router/pkg/systemonecodec` 中：一个无状态类型，实现 `ID`、`Spec`、`Capabilities`、`Plan`、`DecodeReply`、`ListingPath` 和 `DecodeListing`。`Plan` 决定一个批量的每次传输调用：一次批量调用，或每个请求一次调用；只读写 IR（`systemoneprotocol`），从不读写其他 codec 的传输形式。listener 可以使用的 schema 还需要一个 `ListenerCodec`（见 `listener.go`）。
4. **注册** 到 `Builtin()`（`registry.go`），并把它的 ID 加入 `pkg/apiregistry` 的 `systemone` 项（上游 schema；listener 可以使用时还要加入 `ListenerSchemas`）。
5. **测试：** 黄金计划和回复、加载时的拒绝、它的错误和模型列表，以及针对由规范构建的 mock 服务器的一致性测试。
6. **文档：** 更新 [System One schema](../../system-one/use/schemas.md) 页面和由规范生成的兼容性矩阵。

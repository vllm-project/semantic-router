---
translation:
  source_commit: "e86e1ac69ece8f9921cddbbfa12a4c2d8f50b66b"
  source_file: "docs/api/session-identification.md"
  outdated: false
---

# 会话标识 {#session-identification}

路由学习保护要求客户端为相关轮次提供稳定、显式的身份。回放和遥测可以使用派生的回退身份，但这些回退身份不会启用保护。

## 选择所需身份 {#choose-the-identity-you-need}

使用默认的 `scope: conversation` 保护时，请同时发送两个请求头：

```http
x-session-id: tenant-42:session-7
x-conversation-id: conversation-3
```

在整个 session 内保持 `x-session-id` 稳定，并在其中每个 conversation 内保持 `x-conversation-id` 稳定。conversation 保护需要两者；`scope: session` 只需要 session 请求头。若自定义了请求头名称，请使用配置中的名称。缺少必需身份时，请求仍会正常路由，但保护不会保持模型。

Replay 在具有显式身份时使用相同配置的会话和对话请求头名称。记录对话 ID 不会改变保护范围，也不会在 session 范围下重置模型归属。

Responses API 将显式对话成员关系与响应链路分开。请求中的 `conversation` 值标识成员关系；`previous_response_id` 读取保留历史并提供内部链路跟踪键，不会加入或创建对话。两者都不存在时，Router 生成内部跟踪身份。这些遥测身份不能替代路由学习保护所要求的配置身份请求头。

## Chat 与 Messages API 优先级 {#chat-and-messages-api-priority}

当请求不是 Responses API 请求时，按以下顺序取第一个可用来源作为 Router 会话 id：

1. 应用或网关提供的 `x-session-id`。
2. Anthropic Messages 请求上的 `x-claude-code-session-id`。
3. Anthropic `metadata.user_id`，存储时加 `ant-md-` 前缀。
4. 由消息历史和已认证用户身份生成的指纹。
5. 没有用户身份时，由消息结构生成的指纹。
6. 由 `x-request-id` 派生的哈希，作为最终回退。

该顺序使显式会话键保持稳定，同时仍为只发送消息历史的客户端提供可用回退。派生指纹不应视为持久的应用标识符：编辑历史或更改身份上下文都可能改变它们。

## 隐私与稳定性 {#privacy-and-stability}

`x-session-id` 和 `x-claude-code-session-id` 在去除首尾空白后原样传递；Router 不会对其做哈希或加命名空间。不要在这两个头中发送密钥或原始个人数据。

若标识符必须按租户隔离或匿名化，请在客户端或受信任网关中转换，再将结果写入 `x-session-id`。该头具有最高的客户端提供优先级，因此下游 Router 功能会一致使用转换后的值。

在会话生命周期内保持所选 id 稳定。将同一个 id 复用于无关用户或对话，会混会话感知路由状态、遥测或记忆范围。

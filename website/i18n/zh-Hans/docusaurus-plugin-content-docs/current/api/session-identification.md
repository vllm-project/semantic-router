---
is_mtpe: false
translation:
  source_commit: "7f1b814c97035e96a3780a3b8780ea5d3b6a6b24"
  source_file: "docs/api/session-identification.md"
  outdated: false
---

# 会话标识 {#session-identification}

路由学习保护要求客户端为相关轮次提供稳定、显式的 session 身份。对于没有提供身份的请求，回放和遥测也会按其对话开头的指纹分组，但该指纹永远不会启用保护。

推理连接与工具循环验证见[接入 Agent Harness](../installation/agent-harness)。

## 选择所需身份 {#choose-the-identity-you-need}

使用默认的 `scope: conversation` 保护时，请同时发送两个请求头：

```http
x-session-id: tenant-42:session-7
x-conversation-id: conversation-3
```

在整个 session 内保持 `x-session-id` 稳定，并在其中每个 conversation 内保持 `x-conversation-id` 稳定。`scope: session` 只读取 session 请求头。若自定义了请求头名称，请使用配置中的名称。

在 `scope: conversation` 下，带有 session 请求头但没有 conversation 请求头的请求仍按 conversation 保持模型。Router 按对话的开头为其命名，使用的是[Chat 与 Messages API 优先级](#chat-and-messages-api-priority)中描述的指纹。agent 的主线程与其旁路调用（例如生成标题的请求）会保持各自的模型。如果 harness 自己跟踪对话，请发送 `x-conversation-id`：它会取代指纹，并且在对话开头发生变化（例如压缩上下文）后仍保持不变。Replay 会把通过指纹识别的对话的身份来源记录为 `message_hash`。

没有 session 请求头时，请求仍会正常路由，但保护不会保持模型。

Replay 在具有显式身份时使用相同配置的会话和对话请求头名称。记录对话 ID 不会改变保护范围，也不会在 session 范围下重置模型归属。

Responses API 将显式对话成员关系与响应链路分开。请求中的 `conversation` 值标识成员关系；`previous_response_id` 读取保留历史并提供内部链路跟踪键，不会加入或创建对话。两者都不存在时，Router 生成内部跟踪身份。这些遥测身份不能替代路由学习保护所要求的配置身份请求头。

## Chat 与 Messages API 优先级 {#chat-and-messages-api-priority}

当请求不是 Responses API 请求时，按以下顺序取第一个可用来源作为 Router 会话 id：

1. Agent Harness、其他客户端或受信任网关提供的 `x-session-id`。
2. Anthropic Messages 请求上的 `x-claude-code-session-id`。
3. Anthropic `metadata.user_id`，存储时加 `ant-md-` 前缀。
4. 由第一条用户消息和已认证用户身份生成的指纹。
5. 没有已认证用户时，由对话开头生成的指纹，存储时加 `cc-start-` 前缀：取系统与开发者指令以及第一条用户消息各自的前 512 个字符；网关设置了已认证租户（`x-authz-tenant-id`）时也计入该租户。
6. 由 `x-request-id` 派生的哈希，作为最终回退。

该顺序使显式会话键保持稳定，同时仍为只发送消息历史的客户端提供可用回退。同一对话的每个请求都会重复其开头，因此没有 session 请求头的客户端在整个对话中获得同一个会话 id：Replay 将其显示为一条轨迹，会话遥测在各轮之间累积，每一轮都能看到上一轮使用的模型。Router 只保存哈希。

开头相同的两个对话（相同的指令和第一条消息，且属于同一租户）会共享这个 id。它们的 Replay、遥测和上一轮模型查询会因此合并；保护、粘性工具状态和 KV 放置则不会，因为派生指纹永远不会启用它们。派生指纹也不是持久的应用标识符：客户端改写历史开头（例如压缩历史）后会得到新的指纹。

## 隐私与稳定性 {#privacy-and-stability}

`x-session-id` 和 `x-claude-code-session-id` 在去除首尾空白后原样传递；Router 不会对其做哈希或加命名空间。不要在这两个头中发送密钥或原始个人数据。

若标识符必须按租户隔离或匿名化，请在客户端或受信任网关中转换，再将结果写入 `x-session-id`。该头具有最高的客户端提供优先级，因此下游 Router 功能会一致使用转换后的值。

在会话生命周期内保持所选 id 稳定。将同一个 id 复用于无关用户或对话，会混会话感知路由状态、遥测或记忆范围。

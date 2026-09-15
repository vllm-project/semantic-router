---
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/api/session-identification.md"
  outdated: false
---

# 会话标识 {#session-identification}

会话感知路由、记忆、回放和遥测需要为相关轮次提供稳定身份。Router 在可用时使用显式的客户端身份，否则派生回退身份。

## 选择所需身份 {#choose-the-identity-you-need}

对于 Chat Completions 和 Messages API 客户端，若应用已有稳定会话键，请发送 `x-session-id`：

```http
x-session-id: tenant-42:conversation-7
```

对于路由学习保护，`x-conversation-id` 可以标识该会话内更窄的对话。保护策略在 `scope: conversation` 时使用对话身份；`scope: session` 时使用更宽的会话身份。

Responses API 管理自己的对话链。请求中的 `conversation` 值优先；否则由 `previous_response_id` 链继承原始对话；两者都不存在时会生成新的 conversation id。

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

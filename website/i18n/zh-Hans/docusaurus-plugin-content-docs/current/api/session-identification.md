---
translation:
  source_commit: "e52d6023"
  source_file: "docs/api/session-identification.md"
  outdated: false
---

# 会话标识

路由器为每个请求派生一个会话标识符（`RequestContext.SessionID`），使会话感知插件、记忆操作和遥测使用同一个键。本文介绍优先级顺序和稳定性契约。

## 优先级顺序

`populateSessionTransitionFields` 按从上到下的顺序评估以下来源；第一个非空值优先。

| # | 来源 | 来源说明 |
| --- | --- | --- |
| 1 | `ResponseAPICtx.ConversationID` | 仅限 Response API。由 Response API 提取器写入。 |
| 2 | `x-session-id` 标头 | 运维方/SDK 显式指定的固定值。客户端提供的最高优先级来源。 |
| 3 | `x-claude-code-session-id` 标头 | Claude Code CLI 在 `/v1/messages` 请求中发出的不透明会话级令牌。路由器按原样传递，且不验证格式。 |
| 4 | `metadata.user_id`（Anthropic 请求体） | Anthropic 入站解析器将其镜像到 `IRExtensions.MetadataUserID`。添加 `ant-md-` 前缀。 |
| 5 | `deriveSessionIDFromMessages` | 对消息线程及已解析的 authz 用户计算指纹。 |
| 6 | `deriveSessionIDFromMessagesStructure` | 没有可用用户身份时，对消息结构计算指纹。 |
| 7 | `deriveSessionIDFromRequestID` | 对 `x-request-id` 计算 SHA-256。最后的回退方案。 |

### 为何采用此顺序

- (1) 和 (2) 是由部署或客户端控制的固定值来源；它们必须覆盖路由器派生的任何值。
- (3) 位于 (4) 之前，因为 Claude Code 标头按会话生成，而 `metadata.user_id` 按安装实例生成（不相关的会话也使用相同值）。
- (4) 在启用消息指纹回退之前，为设置了 `metadata.user_id` 的非 Claude Code Anthropic SDK 用户提供稳定种子。
- (5)–(7) 是原有的 chat-completion 回退方案，保持不变。

## 稳定性契约

(2) 和 (3) 中的标头值按原样传递。路由器不会对它们进行哈希、加盐或添加命名空间。除空白字符会被修剪外，插件收到的就是客户端发送的内容。

如果部署出于隐私原因需要为会话 ID 添加命名空间（例如，将 Claude Code UUID 映射为内部的租户级键），请在路由器前运行哈希插件，并让该插件在请求到达路由器之前将转换后的值写入 `x-session-id`。由于 `x-session-id` 的优先级高于 `x-claude-code-session-id`，路由器随后会提供哈希后的值。

(4) 中的 `ant-md-` 前缀是传输契约的一部分：会话感知代码可借此区分以 `metadata.user_id` 为种子的值和以标头为种子的值。对于任何按命名空间进行模式匹配的插件，重命名前缀都属于破坏性变更。

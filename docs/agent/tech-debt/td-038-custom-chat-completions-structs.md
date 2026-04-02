# TD038: Custom Chat Completions Structs Duplicate Official OpenAI SDK Types

## Status

Resolved — all packages migrated to official `openai-go` SDK types.

## Scope

`src/semantic-router/pkg/responseapi`, `pkg/classification`, `pkg/modelselection`,
`pkg/mcp`, `pkg/extproc`, `pkg/anthropic`, `pkg/cache`, `pkg/memory`

## Summary

Several packages defined their own `ChatCompletionRequest`, `ChatCompletionResponse`,
`ChatMessage`, `Choice`, `Usage`, and similar types instead of using the official
`openai-go` SDK types. This duplication risks schema drift against the upstream
OpenAI API and creates maintenance burden when fields are added or changed.

## Evidence

All packages have been migrated:

| Package | Removed Structs | Now Uses |
|---------|----------------|----------|
| `pkg/responseapi` | 8 structs (ChatCompletionRequest, ChatMessage, ToolCall, FunctionCall, ChatTool, ChatCompletionResponse, Choice, CompletionUsage) | `openai.ChatCompletionNewParams`, `openai.ChatCompletion` |
| `pkg/classification` | 6 structs (ChatCompletionRequest, ChatMessage, ChatCompletionResponse, Choice, Message, Usage) | `openai.ChatCompletion`, `openai.ChatCompletionMessageParamUnion` with composition for `ExtraBody` |
| `pkg/modelselection` | 3 structs (ChatCompletionRequest, ChatMessage, ChatCompletionResponse) | `openai.ChatCompletion`, `openai.ChatCompletionMessageParamUnion` with composition for error fields |
| `pkg/mcp` | 4 structs (OpenAITool, OpenAIToolFunction, OpenAIToolCall, OpenAIToolCallFunction) | `openai.ChatCompletionToolParam`, `openai.ChatCompletionMessageToolCall` |
| `pkg/cache` | 2 structs (ChatMessage, OpenAIRequest) | `openai.ChatCompletionNewParams` for request parsing; `extractUserContent` uses SDK content union |
| `pkg/memory` | 1 struct (Message) | `openai.ChatCompletionMessageParamUnion` directly; `SDKMessageRole`/`SDKMessageContent` helpers for extraction |

**Key finding**: Neither `pkg/cache` nor `pkg/memory` custom types were serialized
to persistent storage — `cache.ChatMessage`/`OpenAIRequest` were only used for
parsing incoming request JSON, and `memory.Message` was an in-memory representation.
No cache format migration strategy was needed.

## Why It Matters

- Schema drift: custom structs miss new fields added to the OpenAI API
- Maintenance burden: changes must be replicated across multiple struct definitions
- Testing gap: custom types can silently diverge from what clients actually send
- Earlier review feedback explicitly flagged this consolidation as required work

## Desired End State

All OpenAI-shaped request/response handling uses `openai-go` SDK types. Custom
structs exist only where composition is needed for router-specific extensions
(e.g., vLLM `extra_body`, provider error wrapping), documented with a comment
explaining why the extension is necessary.

## Exit Criteria

- [x] `pkg/cache` types migrated (PR #1685 — no serialization, pure code replacement)
- [x] `pkg/memory` types migrated (PR #1685 — no serialization, pure code replacement)
- [x] Zero custom `ChatCompletion*` type definitions remain outside documented exceptions
- [x] Compatibility tests cover all conversion paths (`pkg/cache/sdk_compat_test.go`)

## Tracking

- Initial migration: PR #1550 (issue #1517)
- Completion: PR for issue #1685

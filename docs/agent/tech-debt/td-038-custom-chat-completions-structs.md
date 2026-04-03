# TD038: Custom Chat Completions Structs Duplicate Official OpenAI SDK Types

## Status

Partially resolved — major packages migrated, remaining surfaces tracked below.

## Scope

`src/semantic-router/pkg/responseapi`, `pkg/classification`, `pkg/modelselection`,
`pkg/mcp`, `pkg/extproc`, `pkg/anthropic`, `pkg/cache`, `pkg/memory`

## Summary

Several packages defined their own `ChatCompletionRequest`, `ChatCompletionResponse`,
`ChatMessage`, `Choice`, `Usage`, and similar types instead of using the official
`openai-go` SDK types. This duplication risks schema drift against the upstream
OpenAI API and creates maintenance burden when fields are added or changed.

## Evidence

Packages that **have already been migrated** in earlier loops:

| Package | Removed Structs | Now Uses |
|---------|----------------|----------|
| `pkg/responseapi` | 8 structs (ChatCompletionRequest, ChatMessage, ToolCall, FunctionCall, ChatTool, ChatCompletionResponse, Choice, CompletionUsage) | `openai.ChatCompletionNewParams`, `openai.ChatCompletion` |
| `pkg/classification` | 6 structs (ChatCompletionRequest, ChatMessage, ChatCompletionResponse, Choice, Message, Usage) | `openai.ChatCompletion`, `openai.ChatCompletionMessageParamUnion` with composition for `ExtraBody` |
| `pkg/modelselection` | 3 structs (ChatCompletionRequest, ChatMessage, ChatCompletionResponse) | `openai.ChatCompletion`, `openai.ChatCompletionMessageParamUnion` with composition for error fields |
| `pkg/mcp` | 4 structs (OpenAITool, OpenAIToolFunction, OpenAIToolCall, OpenAIToolCallFunction) | `openai.ChatCompletionToolParam`, `openai.ChatCompletionMessageToolCall` |

Packages with **remaining custom types** (lower priority):

| Package | Custom Types | Notes |
|---------|-------------|-------|
| `pkg/cache` | Message, Usage structs | Used for cache serialization; migration deferred to avoid cache format break |
| `pkg/memory` | Message struct | Used for memory store serialization |

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

- [ ] `pkg/cache` serialization types migrated or documented as intentional exceptions
- [ ] `pkg/memory` serialization types migrated or documented as intentional exceptions
- [ ] Zero custom `ChatCompletion*` type definitions remain outside documented exceptions
- [ ] Compatibility tests cover all conversion paths

## Tracking

Follow-up issue: #1685

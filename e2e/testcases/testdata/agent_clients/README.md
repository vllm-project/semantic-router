# Agent-client captures

Each file is a tool loop that a real agent client sent through its normal
request path. The `protocol-codec-*-backend-agent-client-replay` cases in the
`response-api` and `provider-protocols` profiles replay these files through
Envoy and the Router to the provider-mocker.

| Field | Meaning |
| --- | --- |
| `client`, `version` | The client and the exact version that produced the capture. |
| `path` | The client endpoint: `/v1/chat/completions`, `/v1/messages` or `/v1/responses`. |
| `headers` | Non-credential headers the client sent, such as `anthropic-beta`. |
| `backends` | The backend formats this capture passes through today, each with the request fields that must reach the provider. |
| `turns` | Two request bodies: the turn that asks for a tool call, then the follow-up that carries the client's call and its result. |

The replay changes only the model, the `stream` flag, the text of the last user
message and, in the follow-up, the tool call's ID and name, which it takes from
the call the mocker returned on the first turn, as the client would. Buffered
replays drop `stream_options`, which Chat Completions accepts only on streams. It
sends each turn streamed and buffered, adds the mocker's `__mock_tool_call__` or
`__mock_provider_error__` marker to the user text, and checks the status, the
client's own response envelope or event order, tool-call identity across stream
deltas, usage, the fields the provider received, that the follow-up reaches the
provider as the answer to the call it returned, and the client's error shape for
a provider 429.

## Current captures

| File | Captured with |
| --- | --- |
| `claude-code-2.1.281-messages-tool-loop.json` | `claude -p --bare --strict-mcp-config --model vllm-sr/auto`, with `ANTHROPIC_BASE_URL` at a loopback recorder |
| `copilot-cli-1.0.88-chat-tool-loop.json` | Copilot CLI BYOK, `COPILOT_PROVIDER_TYPE=openai`, `COPILOT_MODEL=vllm-sr/auto` |
| `copilot-cli-1.0.88-messages-tool-loop.json` | Copilot CLI BYOK, `COPILOT_PROVIDER_TYPE=anthropic`, `COPILOT_MODEL=vllm-sr/auto` |

## Adding a capture

1. Point the client at a loopback server that records each request and answers
   with a tool call and then a final message. No provider key is needed.
2. Keep every field, tool type and content-block type the client sent. Shorten
   system prompts, reminders and tool descriptions, and replace IDs, dates and
   working directories with fixed placeholders. Remove keys, local paths,
   hostnames and personal data. `TestAgentClientCapturesAreSanitized` rejects the
   common leaks and any file over 16 KiB.
3. List only the backends the capture passes on the current main branch. A backend
   that fails is a bug to file; add it to `backends` together with its fix.

Captures from live-provider runs use the same format. Only their client requests
are replayed, because the provider-mocker supplies every response.

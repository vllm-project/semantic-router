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
| `identity` | Protection identity headers. Each entry has `name`, a `placeholder-*` value, and `on`: `turn-0`, `turn-1`, or `subagent`. An empty `headers` list means this capture recorded no identity header. |
| `backends` | The backend formats this capture passes through today, each with the request fields that must reach the provider. |
| `turns` | Two request bodies: the turn that asks for a tool call, then the follow-up that carries the client's call and its result. |
| `subagent` | Optional child-agent request. It is not a third tool-loop turn. Include it when an identity header lists `subagent`. |

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
4. Set `identity`. Use `placeholder-session`, `placeholder-conversation`, or
   `placeholder-agent` for header values. `TestAgentClientCapturesAreSanitized`
   rejects a real session id in an identity header.

Captures from live-provider runs use the same format. Only their client requests
are replayed, because the provider-mocker supplies every response.

## Verify a harness version

`harness-support.yaml` is the list of exact versions, identity headers, and
the highest protection level those captures support. The protocol compatibility
guide renders it between the harness support markers.
`tools/ci/check_harness_support.py` fails when the guide, this file, the
captures, or the replay cases disagree.

CI replays captures. It does not run a harness.

1. Record the client with a loopback recorder. No provider key is needed. When
   the level depends on a subagent, include one subagent request in the same
   run. A live Terminal-Bench or Harbor trial can be named in `live_lane`; the
   capture is what CI replays.
2. Sanitize the capture, then fill `identity` with placeholder values and the
   requests that carry each header.
3. Add an exact version with `status: verified`, `last_verified`, the capture
   files, and the replay cases. Give each identity header a `role` of
   `session`, `conversation`, or `agent`. Do not write a range. Claim only the
   level the recorded headers and the current protection config can hold. R3
   also requires `identity.headers.agent` in the protection config.
4. Update the marked table in the protocol compatibility guide.

When a version breaks, keep the last verified row. Add the broken version with
`status: broken` and `known_broken_since` set to the day the capture or live
run failed. Do not replace either row with a range. File protocol bugs
separately. Do not choose the default protection level for a coding-agent
entrypoint here.

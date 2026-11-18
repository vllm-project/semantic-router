# Session identification

Router Learning protection requires a stable, explicit session identity for
related turns. Replay and telemetry also group requests that carry none, by a
fingerprint of how their conversation starts, but that fingerprint never
enables protection.

For the inference connection and a tool-loop check, start with
[Connect an agent harness](../installation/agent-harness).

## Choose the identity you need

For the default `scope: conversation` protection, send both headers:

```http
x-session-id: tenant-42:session-7
x-conversation-id: conversation-3
```

Keep `x-session-id` stable for the session and `x-conversation-id` stable for
each conversation inside it. A policy with `scope: session` reads only the
session header. Use your configured header names if customized.

Under `scope: conversation`, a request with the session header but no
conversation header is still held per conversation. The Router names the
conversation by how it starts, with the fingerprint described under
[Chat and Messages API priority](#chat-and-messages-api-priority). An agent's
main thread and its side calls, such as a title request, keep separate models.
Send `x-conversation-id` when your harness tracks its conversations: it
replaces the fingerprint and survives a change to the conversation's opening,
such as compaction. Replay reports a fingerprinted conversation with the
identity source `message_hash`.

Without the session header the request still routes, but protection does not
retain a model.

An agent harness that delegates work to subagents can also name each agent.
Protection then holds each named agent on its own model inside the session,
and requests without an agent keep the session or conversation key. The agent
is never required. The harnesses the Router supports need no identity headers
configured, because protection reads what they send:

| Harness | Session | Agent |
| --- | --- | --- |
| Claude Code | `x-claude-code-session-id`, a `session_id_headers` default | `x-claude-code-agent-id`, sent on subagent requests only |
| Codex CLI | `session-id` | `thread-id`; on a root thread it equals `session-id` |
| OpenCode | `x-session-id`, a `session_id_headers` default; on a subagent's calls `x-parent-session-id` | on a subagent's calls, its own session in `x-session-id` |
| Pi, with the provider `vllm-sr connect pi` writes, and the VS Code extension | `x-session-id`, a `session_id_headers` default | none |
| DeepSeek Harness | none | none |
| Cursor, with a custom OpenAI base URL | none | none |
| OpenClaw | none | none |

- **Session.** With `identity.headers.session` unset, protection reads the
  session the request pins, `x-vsr-session-id` or a `session_id_headers`
  header, and without one Codex's `session-id`. An OpenCode subagent runs in
  a child session of its own and sends the session that spawned it in
  `x-parent-session-id`; unless the request pins a session in
  `x-vsr-session-id`, protection keys the call by that parent session.
- **Agent.** With `identity.headers.agent` unset, protection reads
  `x-claude-code-agent-id`, then `thread-id`. A `thread-id` equal to the
  request's `session-id` marks Codex's root thread, the main agent, and names no
  agent, also when the request pins another session, as
  `vllm-sr connect codex --launch` does. An OpenCode subagent's own session,
  its `x-session-id`, is its agent. OpenCode is known by its User-Agent,
  `opencode/<version>`, and DeepSeek Harness by its User-Agent,
  `deepseek-harness/<version>`; DeepSeek Harness sends no session or agent
  header, so its calls keep the session the Router derives unless the request
  pins one in `x-vsr-session-id`. Cursor is known by a User-Agent that starts
  with `Cursor/`, and OpenClaw by the `<!-- openclaw:` comment its system
  prompt opens with; neither sends a session or agent header.
- **Replay's conversation id for Cursor and OpenClaw.** The conversation-start
  fingerprint reads only the first 512 characters of the first user message,
  and Cursor opens that message with the same workspace preamble in every
  conversation, so its conversations would share one fingerprint. Replay
  therefore files a Cursor or OpenClaw call that pins no session under
  `cursor-start-<hash>` or `openclaw-start-<hash>`: a hash of the whole first
  user message, the first 512 characters of the system instructions, the
  request's `user` field and the authenticated tenant, keyed with
  `USER_SCOPE_NAMESPACE_SECRET` as the fingerprint is. Every call of one
  conversation resends them, and only the hash is kept. Replay and tracing
  read it; protection and routing never do. A compaction that rewrites the
  conversation's opening starts a new id.
- **Codex's turn metadata.** Without those headers, protection reads the
  `session_id` and `thread_id` of Codex's `x-codex-turn-metadata`, or of a
  Responses request's `client_metadata`.
- **A configured header replaces the defaults** for its part: set
  `identity.headers.agent` to a header your harness does not send to hold its
  subagents on their parent's model.

Replay records the session and conversation protection reads, from the
configured headers or the defaults above, when that identity is available.
When protection cannot key a call, Replay still files it under the session
the harness names: the configured session header, the pinned session, or
Codex's `session-id`. A Codex Responses call under `scope: conversation`
without `x-conversation-id`, which has no conversation start to name its
conversation by, or under an `identity.headers.session` Codex does not send,
is filed under its `session-id` that way, so a run is one session rather than
one per call. An OpenCode subagent's calls are filed under the session that
spawned it. Replay also records the agent of each call from the harness
defaults above, whichever agent header protection reads, and the agent that
spawned it, Codex's `x-codex-parent-thread-id`: see
[Agent runs](../tutorials/plugin/router-replay#agent-runs). Recording a
conversation ID does not change the protection scope or reset model ownership
under session scope.

The Responses API keeps explicit conversation membership separate from response
lineage. A request's `conversation` value identifies that membership;
`previous_response_id` retrieves retained history and provides an internal
lineage tracking key, without joining or creating a conversation. With neither,
the router generates an internal tracking identity. These telemetry identities
do not replace the configured identity headers required by Router Learning
protection.

## Chat and Messages API priority

When the request is not a Responses API request, the first available source in
this order becomes the router session id:

1. `x-session-id` supplied by the agent harness, another client, or a trusted gateway.
2. `x-claude-code-session-id` on Anthropic Messages requests.
3. Anthropic `metadata.user_id`, stored with an `ant-md-` prefix.
4. A fingerprint of the first user message and the authenticated user identity.
5. Without an authenticated user, a fingerprint of how the conversation
   starts, stored with a `cc-start-` prefix: the first 512 characters of the
   system and developer instructions and of the first user message, with the
   authenticated tenant (`x-authz-tenant-id`) when the gateway sets one.
6. A hash derived from `x-request-id` as the final fallback.

This order keeps explicit conversation keys stable while still giving clients
that send only message history a usable fallback. Every request of a
conversation repeats its start, so a client without a session header gets one
session id for the whole conversation: Replay shows it as one trajectory,
session telemetry accumulates across its turns, and each turn sees the model
the previous turn used. The Router keeps only the hash. With
`USER_SCOPE_NAMESPACE_SECRET` set, the hash is an HMAC-SHA256 under that
secret, as response-cache scoping is: whoever lacks the secret cannot confirm
a guessed conversation start from its id, and another deployment gives the
same start another id. Without the secret it is a plain SHA-256.

Two conversations that start the same way, with the same instructions and
first message in the same tenant, share that id. Their Replay, telemetry and
previous-model lookups then merge; protection, sticky tool state and KV
placement do not, because a derived fingerprint never enables them. Derived
fingerprints are not durable application identifiers either: a client that
rewrites the start of its history, for example by compacting it, starts a new
one.

## Privacy and stability

`x-session-id` and `x-claude-code-session-id` pass through after whitespace is
trimmed; the router does not hash or namespace them. Do not send secrets or raw
personal data in either header.

If identifiers must be tenant-scoped or pseudonymous, transform them in the
client or trusted gateway and write the result to `x-session-id`. Because that
header has the highest client-supplied priority, downstream router features use
the transformed value consistently.

Keep the chosen id stable for the lifetime of the session. Reusing one id for
unrelated users or conversations can mix session-aware routing state,
telemetry, or memory scope.

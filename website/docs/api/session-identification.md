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
Set `identity.headers.agent` to the header that carries that name, such as
`x-claude-code-agent-id`, which Claude Code sends on subagent requests only.
For Codex CLI, set `session: session-id` and `agent: thread-id`. Codex sends
both on every request; on a root thread the two values are equal.
Protection then holds each named agent on its own model inside the session,
and requests without the header keep the session or conversation key. The
agent header is optional, has no default, and is never required.

Replay uses the same configured session and conversation header names when that
explicit identity is available. Recording a conversation ID does not change the
protection scope or reset model ownership under session scope.

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

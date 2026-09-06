# Session identification

Session-aware routing, memory, replay, and telemetry need a stable identity for
related turns. The router uses an explicit client identity when one is
available and derives a fallback otherwise.

## Choose the identity you need

For Chat Completions and Messages API clients, send `x-session-id` when your
application already has a stable session key:

```http
x-session-id: tenant-42:conversation-7
```

For Router Learning protection, `x-conversation-id` can identify a narrower
conversation inside that session. A protection policy with
`scope: conversation` uses the conversation identity; `scope: session` uses
the broader session identity.

The Responses API manages its own conversation chain. Its request
`conversation` value wins, otherwise a `previous_response_id` chain inherits
the original conversation, and a new conversation id is generated when neither
is present.

## Chat and Messages API priority

When the request is not a Responses API request, the first available source in
this order becomes the router session id:

1. `x-session-id` supplied by the application or gateway.
2. `x-claude-code-session-id` on Anthropic Messages requests.
3. Anthropic `metadata.user_id`, stored with an `ant-md-` prefix.
4. A fingerprint of the message history and authenticated user identity.
5. A fingerprint of the message structure when no user identity is available.
6. A hash derived from `x-request-id` as the final fallback.

This order keeps explicit conversation keys stable while still giving clients
that send only message history a usable fallback. Derived fingerprints should
not be treated as durable application identifiers: editing history or changing
identity context can change them.

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

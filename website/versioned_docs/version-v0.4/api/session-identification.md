# Session identification

Router Learning protection requires stable, explicit client identities for
related turns. Replay and telemetry can use derived fallback identities, but
those fallbacks do not enable protection.

## Choose the identity you need

For the default `scope: conversation` protection, send both headers:

```http
x-session-id: tenant-42:session-7
x-conversation-id: conversation-3
```

Keep `x-session-id` stable for the session and `x-conversation-id` stable for
each conversation inside it. Conversation protection requires both; a policy
with `scope: session` requires only the session header. Use your configured
header names if customized. If a required identity is missing, the request
still routes, but protection does not retain a model.

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

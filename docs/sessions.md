# Session identification

The router derives a session identifier (`RequestContext.SessionID`) for
every request so session-aware plugins, memory operations, and telemetry
agree on a single key. This document describes the priority order and
the stability contract.

## Priority order

`populateSessionTransitionFields` evaluates the following sources
top-to-bottom; the first non-empty value wins.

| # | Source                                  | Provenance                                                                                       |
|---|-----------------------------------------|--------------------------------------------------------------------------------------------------|
| 1 | `ResponseAPICtx.ConversationID`         | Response API only. Stamped by the Response API extractor.                                        |
| 2 | `x-session-id` header                   | Explicit operator/SDK pin. Highest-priority client-supplied source.                              |
| 3 | `x-claude-code-session-id` header       | Per-conversation UUID emitted by the Claude Code CLI on `/v1/messages` requests.                 |
| 4 | `metadata.user_id` (Anthropic body)     | Mirrored by the Anthropic inbound parser into `IRExtensions.MetadataUserID`. Prefixed `ant-md-`. |
| 5 | `deriveSessionIDFromMessages`           | Fingerprint over the message thread plus the resolved authz user.                                |
| 6 | `deriveSessionIDFromMessagesStructure`  | Fingerprint over message structure when no user identity is available.                           |
| 7 | `deriveSessionIDFromRequestID`          | SHA-256 of `x-request-id`. Last resort.                                                          |

### Why this order

- (1) and (2) are pinning sources controlled by the deployment or the
  client; they must override any router-derived value.
- (3) sits above (4) because the Claude Code header is per-conversation
  whereas `metadata.user_id` is per-installation (the same value across
  unrelated conversations).
- (4) gives non-Claude-Code Anthropic SDK users that set
  `metadata.user_id` a stable seed before the message-fingerprint
  fallbacks engage.
- (5)–(7) are the pre-existing chat-completion fallbacks and remain
  unchanged.

## Stability contract

The header values at (2) and (3) pass through verbatim. The router does
not hash, salt, or namespace them. Plugins receive whatever the client
sent, modulo whitespace trimming.

If a deployment needs to namespace session IDs for privacy reasons (for
example to map a Claude Code UUID into an internal per-tenant key), run
a hashing plugin in front of the router and have it write the
transformed value into `x-session-id` before the request reaches the
router. The router will then surface the hashed value because
`x-session-id` outranks `x-claude-code-session-id`.

The `ant-md-` prefix at (4) is part of the wire contract: it lets
session-aware code distinguish a value seeded from `metadata.user_id`
from a value seeded by a header. Renaming the prefix is a breaking
change for any plugin pattern-matching on the namespace.

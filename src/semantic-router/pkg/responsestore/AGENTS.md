# Response Store Package Notes

## Scope

- `src/semantic-router/pkg/responsestore/**`

## Responsibilities

- Keep shared response or conversation persistence contracts separate from backend-specific storage implementations.
- Treat TTL, keying, indexing, and serialization as backend support details, not reasons to widen the shared interface.
- Keep memory-backed and Redis-backed storage paths behind the same narrow store contract.

## Change Rules

- `interface.go` is a ratcheted hotspot. Avoid growing the shared interface when a backend-local helper or optional support seam can own the behavior.
- Do not put Redis TLS, connection bootstrap, serialization, and interface evolution into the same change path without extracting helpers first.
- Keep response or conversation indexing logic behind backend-specific helpers instead of turning the shared contract into a feature matrix.

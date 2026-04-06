# Memory Package Notes

## Scope

- `src/semantic-router/pkg/memory/**`

## Responsibilities

- Keep memory extraction, sanitization, consolidation, reflection, filtering, metrics, and backend storage on separate seams.
- Treat Milvus-backed storage as a schema and query adapter, not as the owner for shared collection lifecycle policy.
- Keep embedding generation and backend persistence separate from memory-type semantics and retention logic.

## Change Rules

- `milvus_store.go` is a ratcheted hotspot. Shared collection bootstrap, retry, or index lifecycle belongs in shared helpers such as `pkg/milvuslifecycle`, while this file should stay focused on memory-specific schema and query behavior.
- Do not merge extractor, consolidation, and persistence concerns back into one file when a narrower helper can own the change.
- Keep `caching_store.go` thin; backend-specific caching or persistence behavior should grow in the backend or helper seam that already owns it.

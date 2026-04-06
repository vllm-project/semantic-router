# Cache Package Notes

## Scope

- `src/semantic-router/pkg/cache/**`

## Responsibilities

- Keep cache contracts, backend adapters, lookup or scoring helpers, and factory wiring on separate seams.
- Treat Redis, Valkey, in-memory, Milvus, and hybrid cache implementations as backend-specific adapters rather than one shared dumping ground.
- Keep HNSW, SIMD distance, and lookup-path helpers separate from backend connection or TTL policy.

## Change Rules

- `inmemory_cache.go` and `milvus_cache.go` are ratcheted hotspots. New search, reconstruction, or embedding-path helpers belong in adjacent support files instead of widening those backends.
- Keep `cache_interface.go` narrow. New backend-specific behavior should not widen the shared interface when a backend-local helper or optional capability seam can own it.
- Keep `hybrid_cache.go` and its support files focused on orchestration; backend-specific connection, query, or reconstruction details belong in the backend files or dedicated helpers.

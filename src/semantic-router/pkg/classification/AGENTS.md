# Classification Package Notes

## Scope

- `src/semantic-router/pkg/classification/**`

## Responsibilities

- Keep request-time classification orchestration separate from model discovery and bootstrap.
- Keep per-family concerns such as category, jailbreak, and PII inference behind narrow seams instead of one giant shared state table.
- Keep the unified batch path and any legacy or fallback path composable behind explicit interfaces or dedicated helpers.
- Treat the package as classification runtime code, not as a dumping ground for service assembly, config migration, or unrelated routing policy.

## Change Rules

- Do not add new backend-selection matrices, mapping loaders, or bootstrap branches directly into `classifier.go` when a family adapter, backend helper, or discovery file can own them.
- Do not mix model discovery, mapping ownership, metrics wiring, and request-time inference in the same new file.
- When adding a new classifier backend or signal family, prefer dedicated support files for backend, mapping, or family-specific behavior and keep constructors thin.
- Touching `classifier.go` should be extraction-first: the file may remain a hotspot temporarily, but it should not gain a new major responsibility.

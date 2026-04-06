# DSL Package Notes

## Scope

- `src/semantic-router/pkg/dsl/**`

## Responsibilities

- Keep grammar and AST ownership separate from compile, decompile, and format or emission helpers.
- Keep authoring sugar such as `DECISION_TREE`, `IF ELSE`, and `TEST` support separate from the canonical runtime routing contract.
- Keep YAML or JSON value translation on narrower seams instead of rejoining it with parser or decompiler orchestration.

## Change Rules

- `parser.go` and `decompiler.go` are ratcheted hotspots. Extend them through sibling helpers for family-specific parsing, routing export, or template extraction instead of widening one giant switch path.
- `dsl_test.go` is a legacy regression hotspot. Prefer focused fixture helpers and new targeted tests over appending another large inline matrix to the same file.
- Do not turn the DSL package into a second config-runtime owner. Runtime schema defaults and canonical contract behavior belong in `pkg/config` or public authoring seams, while this package owns syntax, AST, and authoring translation.
